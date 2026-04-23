from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from models import DiagnosticState
from tools import query_knowledge_base, restart_container, search_logs, send_alert

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None

logger = logging.getLogger("ai_devops.agent")

class ErrorClassification(BaseModel):
    category: Literal["OOM", "5XX", "SQL", "NETWORK", "DISK", "UNKNOWN"]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class RootCauseAnalysis(BaseModel):
    root_cause: str = Field(..., description="最可能的根本原因")
    impact_analysis: str = Field(..., description="影响范围和潜在后果")


class SuggestionResult(BaseModel):
    recommendations: List[str]
    auto_fix_action: str | None = None


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
    if api_key and ChatGoogleGenerativeAI is not None:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
    raise RuntimeError("未配置 GOOGLE_API_KEY")


def has_real_llm() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY")) and ChatGoogleGenerativeAI is not None


def fallback_classification(state: DiagnosticState) -> ErrorClassification:
    text = f"{state.event.log_event.error_type} {state.event.log_event.message}".lower()
    memory_percent = state.event.container_stats.memory_percent
    if "oom" in text or "outofmemory" in text or memory_percent >= 90:
        return ErrorClassification(category="OOM", severity="CRITICAL" if memory_percent >= 95 else "HIGH")
    if "sql" in text or "deadlock" in text:
        return ErrorClassification(category="SQL", severity="HIGH")
    if "timeout" in text or "5xx" in text:
        return ErrorClassification(category="5XX", severity="HIGH")
    if "network" in text or "dns" in text or "connection refused" in text:
        return ErrorClassification(category="NETWORK", severity="HIGH")
    if "disk" in text or "no space" in text:
        return ErrorClassification(category="DISK", severity="CRITICAL")
    return ErrorClassification(category="UNKNOWN", severity="MEDIUM")


def fallback_root_cause(state: DiagnosticState) -> RootCauseAnalysis:
    if state.error_category == "OOM":
        return RootCauseAnalysis(
            root_cause="实例内存接近上限，应用存在堆空间不足或短时对象激增，导致 OOM 与 Full GC 抖动。",
            impact_analysis="该实例会持续超时甚至重启，当前请求会受到影响；如果没有隔离流量，故障可能扩散到上游调用。",
        )
    if state.error_category == "5XX":
        return RootCauseAnalysis(
            root_cause="服务实例或其下游依赖出现异常，导致请求在应用层返回 5XX。",
            impact_analysis="用户请求直接失败，若流量集中到异常实例会迅速放大错误率。",
        )
    if state.error_category == "NETWORK":
        return RootCauseAnalysis(
            root_cause="服务间网络连通性或 DNS 解析存在异常，导致调用失败。",
            impact_analysis="依赖链上的请求可能批量超时，并拖慢调用线程池。",
        )
    return RootCauseAnalysis(
        root_cause="当前信息不足以定位单一根因，需要继续结合日志与指标排查。",
        impact_analysis="若不及时处理，可能持续影响部分请求成功率和延迟。",
    )


def fallback_suggestions(state: DiagnosticState) -> SuggestionResult:
    if state.error_category == "OOM":
        return SuggestionResult(
            recommendations=[
                "1. [紧急] 先摘流或限流，再重启异常容器，尽快恢复可用性。",
                "2. [短期] 调整 JVM 堆上限并校验容器 memory limit，避免再次触发 OOM。",
                "3. [排查] 导出 heap dump，检查缓存膨胀、连接未释放或批处理堆积。",
            ],
            auto_fix_action="restart_container",
        )
    return SuggestionResult(
        recommendations=[
            "1. [紧急] 先确认故障实例和影响范围，必要时做流量切换。",
            "2. [排查] 补充最近 15 分钟日志和关键运行指标，缩小根因范围。",
            "3. [修复] 根据根因调整配置或重启异常实例，再持续观察错误率。",
        ],
        auto_fix_action=None,
    )


def classify_error(state: DiagnosticState) -> DiagnosticState:
    logger.info("agent.node.start node=classify session_id=%s", state.event.id)
    if has_real_llm():
        llm = get_llm()
        event = state.event
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是专业的 DevOps 故障分类专家。"),
                (
                    "user",
                    "请分类以下错误。\n"
                    f"错误类型: {event.log_event.error_type}\n"
                    f"日志消息: {event.log_event.message}\n"
                    f"容器内存使用: {event.container_stats.memory_percent:.1f}%\n"
                    f"容器重启次数: {event.container_stats.restart_count}\n",
                ),
            ]
        )
        result: ErrorClassification = llm.with_structured_output(ErrorClassification).invoke(prompt.format_messages())
    else:
        result = fallback_classification(state)
    state.error_category = result.category
    state.severity = result.severity
    logger.info(
        "agent.node.end node=classify session_id=%s error_category=%s severity=%s",
        state.event.id,
        state.error_category,
        state.severity,
    )
    return state


def gather_context(state: DiagnosticState) -> DiagnosticState:
    logger.info("agent.node.start node=gather_context session_id=%s", state.event.id)
    event = state.event
    state.additional_logs = [
        search_logs.invoke(
            {
                "container_name": event.container_stats.container_name,
                "keyword": event.log_event.error_type,
                "lines": 20,
            }
        )
    ]
    state.knowledge_base_results = [query_knowledge_base.invoke({"error_type": event.log_event.error_type})]
    logger.info(
        "agent.node.end node=gather_context session_id=%s additional_logs=%s knowledge_hits=%s",
        state.event.id,
        len(state.additional_logs),
        len(state.knowledge_base_results),
    )
    return state


def analyze_root_cause(state: DiagnosticState) -> DiagnosticState:
    logger.info("agent.node.start node=analyze session_id=%s", state.event.id)
    if has_real_llm():
        llm = get_llm()
        event = state.event
        recent_logs = "\n".join(event.recent_logs[:5]) if event.recent_logs else "无"
        context_logs = "\n".join(state.additional_logs) if state.additional_logs else "无"
        kb = "\n".join(state.knowledge_base_results) if state.knowledge_base_results else "无"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是资深 SRE。请只输出结构化分析结果，不要附加额外说明。",
                ),
                (
                    "human",
                    "请分析根因。\n"
                    f"错误类别: {state.error_category}\n"
                    f"严重程度: {state.severity}\n"
                    f"原始错误: {event.log_event.message}\n"
                    f"CPU: {event.container_stats.cpu_percent:.1f}%\n"
                    f"内存: {event.container_stats.memory_percent:.1f}%\n"
                    f"重启次数: {event.container_stats.restart_count}\n"
                    f"最近日志:\n{recent_logs}\n"
                    f"扩展日志:\n{context_logs}\n"
                    f"知识库:\n{kb}\n",
                ),
            ]
        )
        result: RootCauseAnalysis = llm.with_structured_output(RootCauseAnalysis).invoke(prompt.format_messages())
    else:
        result = fallback_root_cause(state)
    state.root_cause = result.root_cause
    state.impact_analysis = result.impact_analysis
    logger.info(
        "agent.node.end node=analyze session_id=%s root_cause=%s",
        state.event.id,
        (state.root_cause or "")[:120],
    )
    return state


def generate_recommendations(state: DiagnosticState) -> DiagnosticState:
    logger.info("agent.node.start node=generate session_id=%s", state.event.id)
    if has_real_llm():
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是故障处理专家。输出 3-5 条可执行建议，按优先级排序，并判断是否适合自动修复。",
                ),
                (
                    "human",
                    "请生成建议。\n"
                    f"根因: {state.root_cause}\n"
                    f"影响: {state.impact_analysis}\n"
                    f"类别: {state.error_category}\n"
                    f"严重程度: {state.severity}\n",
                ),
            ]
        )
        result: SuggestionResult = llm.with_structured_output(SuggestionResult).invoke(prompt.format_messages())
    else:
        result = fallback_suggestions(state)
    state.recommendations = result.recommendations
    state.auto_fix_action = result.auto_fix_action
    state.awaiting_human_input = True
    logger.info(
        "agent.node.end node=generate session_id=%s recommendations=%s auto_fix_action=%s",
        state.event.id,
        len(state.recommendations),
        state.auto_fix_action,
    )
    return state


def human_decision_node(state: DiagnosticState) -> DiagnosticState:
    logger.info(
        "agent.node node=human_decision session_id=%s human_decision=%s awaiting_human_input=%s",
        state.event.id,
        state.human_decision,
        state.awaiting_human_input,
    )
    state.awaiting_human_input = True
    return state


def conversation_node(state: DiagnosticState) -> DiagnosticState:
    logger.info(
        "agent.node.start node=conversation session_id=%s has_question=%s",
        state.event.id,
        bool(state.human_question),
    )
    if not state.human_question:
        logger.info("agent.node.end node=conversation session_id=%s skipped=no_question", state.event.id)
        return state

    context = (
        f"错误类别: {state.error_category}\n"
        f"严重程度: {state.severity}\n"
        f"根本原因: {state.root_cause}\n"
        f"影响分析: {state.impact_analysis}\n"
        f"修复建议: {'; '.join(state.recommendations)}\n"
        f"容器: {state.event.container_stats.container_name}\n"
        f"CPU: {state.event.container_stats.cpu_percent:.1f}%\n"
        f"内存: {state.event.container_stats.memory_percent:.1f}%\n"
        f"最近日志: {' | '.join(state.event.recent_logs[:5])}\n"
    )
    if has_real_llm():
        llm = get_llm()
        history = state.messages[:]
        history.append(HumanMessage(content=state.human_question))
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是资深 DevOps 值班工程师。结合上下文回答追问，保持专业、准确、简洁，控制在 3-5 句话。\n"
                    "上下文如下:\n{context}",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        response = llm.invoke(prompt.format_messages(context=context, messages=history))
        answer = response.content
    else:
        answer = (
            f"基于当前诊断，{state.root_cause} "
            f"当前容器内存为 {state.event.container_stats.memory_percent:.1f}%，"
            f"建议优先执行: {state.recommendations[0] if state.recommendations else '补充更多日志后再判断'}"
        )
    state.messages.append(HumanMessage(content=state.human_question))
    state.messages.append(AIMessage(content=answer))
    state.conversation_count += 1
    state.human_question = None
    state.awaiting_human_input = True
    logger.info(
        "agent.node.end node=conversation session_id=%s conversation_count=%s",
        state.event.id,
        state.conversation_count,
    )
    return state


def decide_action(state: DiagnosticState) -> DiagnosticState:
    logger.info(
        "agent.node.start node=decide session_id=%s human_decision=%s severity=%s auto_fix_action=%s",
        state.event.id,
        state.human_decision,
        state.severity,
        state.auto_fix_action,
    )
    if state.human_decision == "approve":
        state.should_auto_fix = state.auto_fix_action is not None
        state.should_alert = True
    elif state.human_decision == "reject":
        state.should_auto_fix = False
        state.should_alert = True
    else:
        state.should_auto_fix = bool(
            state.severity == "CRITICAL" and state.auto_fix_action == "restart_container"
        )
        state.should_alert = state.severity in {"CRITICAL", "HIGH", "MEDIUM"}

    state.awaiting_human_input = False
    state.diagnostic_report = build_report(state)
    logger.info(
        "agent.node.end node=decide session_id=%s should_auto_fix=%s should_alert=%s",
        state.event.id,
        state.should_auto_fix,
        state.should_alert,
    )
    return state


def execute_auto_fix(state: DiagnosticState) -> DiagnosticState:
    logger.info(
        "agent.node.start node=auto_fix session_id=%s should_auto_fix=%s auto_fix_action=%s",
        state.event.id,
        state.should_auto_fix,
        state.auto_fix_action,
    )
    if state.should_auto_fix and state.auto_fix_action == "restart_container":
        result = restart_container.invoke(
            {
                "container_name": state.event.container_stats.container_name,
                "reason": state.root_cause or "AI diagnostic decision",
            }
        )
        state.execution_log.append(result)
    logger.info(
        "agent.node.end node=auto_fix session_id=%s execution_log_entries=%s",
        state.event.id,
        len(state.execution_log),
    )
    return state


def send_alert_node(state: DiagnosticState) -> DiagnosticState:
    logger.info(
        "agent.node.start node=alert session_id=%s should_alert=%s",
        state.event.id,
        state.should_alert,
    )
    if state.should_alert:
        result = send_alert.invoke(
            {
                "severity": state.severity or "MEDIUM",
                "message": state.diagnostic_report or "",
                "channels": ["slack", "dingtalk"],
            }
        )
        state.execution_log.append(result)
    logger.info(
        "agent.node.end node=alert session_id=%s execution_log_entries=%s",
        state.event.id,
        len(state.execution_log),
    )
    return state


def should_gather_more_context(state: DiagnosticState) -> str:
    if state.severity in {"CRITICAL", "HIGH"}:
        route = "gather"
    else:
        route = "analyze"
    logger.info(
        "agent.route route=should_gather_more_context session_id=%s severity=%s next=%s",
        state.event.id,
        state.severity,
        route,
    )
    return route


def should_continue_conversation(state: DiagnosticState) -> str:
    if state.conversation_count >= 10:
        route = "decide"
    elif state.human_decision == "continue":
        route = "conversation"
    elif state.human_decision in {"approve", "reject"}:
        route = "decide"
    else:
        route = "wait"
    logger.info(
        "agent.route route=should_continue_conversation session_id=%s human_decision=%s conversation_count=%s next=%s",
        state.event.id,
        state.human_decision,
        state.conversation_count,
        route,
    )
    return route


def should_auto_fix_route(state: DiagnosticState) -> str:
    route = "fix" if state.should_auto_fix else "alert"
    logger.info(
        "agent.route route=should_auto_fix_route session_id=%s should_auto_fix=%s next=%s",
        state.event.id,
        state.should_auto_fix,
        route,
    )
    return route


def build_report(state: DiagnosticState) -> str:
    return (
        "# 故障诊断报告\n\n"
        f"- 事件 ID: {state.event.id}\n"
        f"- 来源: {state.event.log_event.source}\n"
        f"- 错误类型: {state.error_category}\n"
        f"- 严重程度: {state.severity}\n"
        f"- 根本原因: {state.root_cause}\n"
        f"- 影响分析: {state.impact_analysis}\n"
        f"- 修复建议: {' | '.join(state.recommendations)}\n"
        f"- 人工对话轮数: {state.conversation_count}\n"
        f"- 最终决策: {'执行自动修复' if state.should_auto_fix else '人工介入'}\n"
    )


def build_conversation_workflow():
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("classify", classify_error)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("analyze", analyze_root_cause)
    workflow.add_node("generate", generate_recommendations)
    workflow.add_node("human_decision", human_decision_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("decide", decide_action)
    workflow.add_node("auto_fix", execute_auto_fix)
    workflow.add_node("alert", send_alert_node)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        should_gather_more_context,
        {"gather": "gather_context", "analyze": "analyze"},
    )
    workflow.add_edge("gather_context", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "human_decision")
    workflow.add_conditional_edges(
        "human_decision",
        should_continue_conversation,
        {"conversation": "conversation", "decide": "decide", "wait": "human_decision"},
    )
    workflow.add_edge("conversation", "human_decision")
    workflow.add_conditional_edges(
        "decide",
        should_auto_fix_route,
        {"fix": "auto_fix", "alert": "alert"},
    )
    workflow.add_edge("auto_fix", "alert")
    workflow.add_edge("alert", END)
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("models", "DiagnosticEvent"),
            ("models", "ContainerStats"),
            ("models", "LogEvent"),
        ]
    )
    return workflow.compile(
        checkpointer=MemorySaver(serde=serde),
        interrupt_before=["human_decision"],
    )


app = build_conversation_workflow()
