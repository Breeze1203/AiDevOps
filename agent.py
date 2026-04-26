from __future__ import annotations
"""LangGraph 诊断工作流。

这里实现的是项目最核心的部分：
1. 首轮诊断阶段：LLM 自主决定是否调工具，再汇总成诊断结论。
2. 人工交互阶段：支持追问、批准、拒绝。
3. 安全约束阶段：自动修复必须经过人工批准。
"""

import logging
import os
from functools import lru_cache
from typing import List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from llm_runtime import SUPPORTED_PROVIDERS
from models import DiagnosticState
from tools import TOOLS, query_knowledge_base, restart_container, search_logs, send_alert

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None

logger = logging.getLogger("ai_devops.agent")
TOOL_REGISTRY = {tool.name: tool for tool in TOOLS}
HARD_APPROVAL_TOOLS = {"restart_container"}


class ErrorClassification(BaseModel):
    """错误分类结果。"""
    category: Literal["OOM", "5XX", "SQL", "NETWORK", "DISK", "UNKNOWN"]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class RootCauseAnalysis(BaseModel):
    """根因分析结果。"""
    root_cause: str = Field(..., description="最可能的根本原因")
    impact_analysis: str = Field(..., description="影响范围和潜在后果")


class SuggestionResult(BaseModel):
    """修复建议结果。"""
    recommendations: List[str]
    auto_fix_action: str | None = None


class DiagnosticConclusion(BaseModel):
    """首轮诊断最终产出的结构化结论。"""
    error_category: Literal["OOM", "5XX", "SQL", "NETWORK", "DISK", "UNKNOWN"]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    root_cause: str
    impact_analysis: str
    recommendations: List[str]
    auto_fix_action: str | None = None


@lru_cache(maxsize=16)
def get_llm(provider: str, model_name: str) -> BaseChatModel:
    """按 provider / model_name 创建实际的聊天模型实例。"""
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and ChatGoogleGenerativeAI is not None:
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        raise RuntimeError("未配置 GOOGLE_API_KEY 或缺少 langchain-google-genai")

    if provider == "chatgpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and ChatOpenAI is not None:
            return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        raise RuntimeError("未配置 OPENAI_API_KEY 或缺少 langchain-openai")

    if provider == "cloudecode":
        api_key = os.getenv("CLOUDECODE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if api_key and ChatAnthropic is not None:
            return ChatAnthropic(model=model_name, temperature=0, api_key=api_key)
        raise RuntimeError("未配置 CLOUDECODE_API_KEY/ANTHROPIC_API_KEY 或缺少 langchain-anthropic")

    raise RuntimeError(f"不支持的模型提供商: {provider}")


def has_real_llm(state: DiagnosticState) -> bool:
    """判断当前 session 是否具备调用真实 LLM 的条件。"""
    enabled, reason = get_llm_status(state)
    state.llm_enabled = enabled
    state.llm_status = reason
    return enabled


def get_llm_status(state: DiagnosticState) -> tuple[bool, str]:
    """给出“模型是否可用”的布尔值和解释原因。"""
    provider = state.model_provider
    if provider not in SUPPORTED_PROVIDERS:
        return False, f"不支持的模型提供商: {provider}"
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            return False, "未配置 GOOGLE_API_KEY"
        if ChatGoogleGenerativeAI is None:
            return False, "缺少 langchain-google-genai 依赖"
        return True, f"已启用 Gemini 模型 {state.model_name}"
    if provider == "chatgpt":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "未配置 OPENAI_API_KEY"
        if ChatOpenAI is None:
            return False, "缺少 langchain-openai 依赖"
        return True, f"已启用 ChatGPT 模型 {state.model_name}"
    if provider == "cloudecode":
        if not (os.getenv("CLOUDECODE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            return False, "未配置 CLOUDECODE_API_KEY 或 ANTHROPIC_API_KEY"
        if ChatAnthropic is None:
            return False, "缺少 langchain-anthropic 依赖"
        return True, f"已启用 CloudeCode 模型 {state.model_name}"
    return False, f"不支持的模型提供商: {provider}"


def update_token_usage(state: DiagnosticState, usage: dict | None) -> None:
    """把一次模型调用的 usage 累加到 session 总 usage。"""
    if not usage:
        return
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    state.token_usage.input_tokens += input_tokens
    state.token_usage.output_tokens += output_tokens
    state.token_usage.total_tokens += total_tokens


def update_token_usage_from_message(state: DiagnosticState, message: AIMessage | None) -> None:
    """从不同 provider 的消息对象里抽取 usage 并统一累计。"""
    if message is None:
        return
    usage = getattr(message, "usage_metadata", None) or getattr(message, "response_metadata", {}).get("usage_metadata")
    if not usage and getattr(message, "response_metadata", None):
        usage = message.response_metadata.get("token_usage")
    update_token_usage(state, usage)


def fallback_classification(state: DiagnosticState) -> ErrorClassification:
    """当没有真实模型时，用规则做基础错误分类。"""
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
    """规则版根因分析。"""
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
    """规则版修复建议。"""
    if state.error_category == "OOM":
        return SuggestionResult(
            recommendations=[
                "1. [紧急] 先摘流或限流，再准备重启异常容器，待人工批准后执行。",
                "2. [短期] 调整 JVM 堆上限并校验容器 memory limit，避免再次触发 OOM。",
                "3. [排查] 导出 heap dump，检查缓存膨胀、连接未释放或批处理堆积。",
            ],
            auto_fix_action="restart_container",
        )
    return SuggestionResult(
        recommendations=[
            "1. [紧急] 先确认故障实例和影响范围，必要时做流量切换。",
            "2. [排查] 补充最近 15 分钟日志和关键运行指标，缩小根因范围。",
            "3. [修复] 根据根因调整配置；涉及自动修复的动作先走人工审批。",
        ],
        auto_fix_action=None,
    )


def build_event_context(state: DiagnosticState) -> str:
    """把事件关键上下文拼成适合喂给 LLM 的文本。"""
    event = state.event
    recent_logs = "\n".join(event.recent_logs[:8]) if event.recent_logs else "无"
    return (
        f"事件 ID: {event.id}\n"
        f"错误类型: {event.log_event.error_type}\n"
        f"日志消息: {event.log_event.message}\n"
        f"容器: {event.container_stats.container_name}\n"
        f"CPU: {event.container_stats.cpu_percent:.1f}%\n"
        f"内存: {event.container_stats.memory_percent:.1f}%\n"
        f"重启次数: {event.container_stats.restart_count}\n"
        f"最近日志:\n{recent_logs}\n"
    )


def build_agent_transcript(state: DiagnosticState) -> str:
    """把最近的 tool-calling 轨迹压缩成文本摘要。"""
    lines: list[str] = []
    for message in state.agent_messages[-12:]:
        role = message.type.upper()
        if isinstance(message, ToolMessage):
            lines.append(f"{role} [{message.name}]: {message.content}")
        else:
            lines.append(f"{role}: {getattr(message, 'content', '')}")
    return "\n".join(lines) if lines else "无"


def apply_fallback_diagnosis(state: DiagnosticState) -> DiagnosticState:
    """在没有真实 LLM 时，直接生成可用的首轮诊断结果。"""
    has_real_llm(state)
    classification = fallback_classification(state)
    state.error_category = classification.category
    state.severity = classification.severity

    state.additional_logs = [
        search_logs.invoke(
            {
                "container_name": state.event.container_stats.container_name,
                "keyword": state.event.log_event.error_type,
                "lines": 20,
            }
        )
    ]
    state.knowledge_base_results = [query_knowledge_base.invoke({"error_type": state.event.log_event.error_type})]

    root_cause = fallback_root_cause(state)
    suggestions = fallback_suggestions(state)
    state.root_cause = root_cause.root_cause
    state.impact_analysis = root_cause.impact_analysis
    state.recommendations = suggestions.recommendations
    state.auto_fix_action = suggestions.auto_fix_action
    state.awaiting_human_input = True
    state.approval_required = suggestions.auto_fix_action is not None
    return state


def llm_diagnostic_node(state: DiagnosticState) -> DiagnosticState:
    """首轮诊断节点。

    这个节点会：
    - 在首次进入时构造 prompt。
    - 让 LLM 决定要不要调用工具。
    - 把 tool call 或普通回答写入 `agent_messages`。
    """
    logger.info("agent.node.start node=llm_diagnostic session_id=%s", state.event.id)
    if not has_real_llm(state):
        state = apply_fallback_diagnosis(state)
        logger.info(
            "agent.node.end node=llm_diagnostic session_id=%s mode=fallback severity=%s auto_fix_action=%s",
            state.event.id,
            state.severity,
            state.auto_fix_action,
        )
        return state

    if not state.agent_messages:
        state.agent_messages.append(
            HumanMessage(
                content=(
                    "请对以下故障做诊断。你可以自主决定是否调用工具来补充日志、查询知识库或获取上下文。"
                    "你可以提出自动修复建议，但不要把自动修复当成已执行事实。\n\n"
                    f"{build_event_context(state)}"
                )
            )
        )

    llm = get_llm(state.model_provider, state.model_name).bind_tools(TOOLS)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是资深 SRE。先自行判断是否需要调用工具，再输出诊断思考。"
                    "高风险动作如重启容器只能作为建议，不得假设自己已经获批。"
                )
            ),
            *state.agent_messages,
        ]
    )
    update_token_usage_from_message(state, response)
    state.agent_messages.append(response)
    logger.info(
        "agent.node.end node=llm_diagnostic session_id=%s tool_calls=%s",
        state.event.id,
        len(getattr(response, "tool_calls", []) or []),
    )
    return state


def execute_requested_tools(state: DiagnosticState) -> DiagnosticState:
    """执行上一轮 LLM 请求的工具调用。"""
    logger.info("agent.node.start node=execute_tools session_id=%s", state.event.id)
    last_message = state.agent_messages[-1] if state.agent_messages else None
    if not isinstance(last_message, AIMessage):
        return state

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        if tool_name in HARD_APPROVAL_TOOLS and state.human_decision != "approve":
            content = f"BLOCKED: 工具 {tool_name} 需要人工 approve 后才能执行。"
            state.execution_log.append(content)
            state.agent_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_name)
            )
            continue

        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            content = f"ERROR: 未找到工具 {tool_name}"
            state.execution_log.append(content)
            state.agent_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_name)
            )
            continue

        result = tool.invoke(tool_args)
        state.execution_log.append(f"tool:{tool_name} -> {result}")
        if tool_name == "search_logs":
            state.additional_logs.append(result)
        elif tool_name == "query_knowledge_base":
            state.knowledge_base_results.append(result)
        state.agent_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
        )

    logger.info(
        "agent.node.end node=execute_tools session_id=%s execution_log_entries=%s",
        state.event.id,
        len(state.execution_log),
    )
    return state


def finalize_diagnosis(state: DiagnosticState) -> DiagnosticState:
    """把前面的分析轨迹收束成结构化首轮诊断结论。"""
    logger.info("agent.node.start node=finalize_diagnosis session_id=%s", state.event.id)
    if not has_real_llm(state):
        state.awaiting_human_input = True
        state.approval_required = state.auto_fix_action is not None
        return state

    llm = get_llm(state.model_provider, state.model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是专业的 DevOps 故障诊断器。基于事件和工具调用结果，输出最终结构化诊断。"
                "如果建议自动修复，只能返回建议动作，不能假设已经执行。",
            ),
            (
                "human",
                "原始事件:\n{event_context}\n\n"
                "工具调用和分析过程:\n{transcript}\n",
            ),
        ]
    )
    result_bundle = llm.with_structured_output(DiagnosticConclusion, include_raw=True).invoke(
        prompt.format_messages(
            event_context=build_event_context(state),
            transcript=build_agent_transcript(state),
        )
    )
    result: DiagnosticConclusion = result_bundle["parsed"]
    update_token_usage_from_message(state, result_bundle.get("raw"))
    state.error_category = result.error_category
    state.severity = result.severity
    state.root_cause = result.root_cause
    state.impact_analysis = result.impact_analysis
    state.recommendations = result.recommendations
    state.auto_fix_action = result.auto_fix_action
    state.awaiting_human_input = True
    state.approval_required = result.auto_fix_action is not None
    logger.info(
        "agent.node.end node=finalize_diagnosis session_id=%s severity=%s auto_fix_action=%s",
        state.event.id,
        state.severity,
        state.auto_fix_action,
    )
    return state


def human_decision_node(state: DiagnosticState) -> DiagnosticState:
    """人工决策占位节点。

    LangGraph 会在进入这个节点前中断，等待 API 收到新的人工输入后再恢复。
    """
    logger.info(
        "agent.node node=human_decision session_id=%s human_decision=%s awaiting_human_input=%s",
        state.event.id,
        state.human_decision,
        state.awaiting_human_input,
    )
    state.awaiting_human_input = True
    return state


def conversation_node(state: DiagnosticState) -> DiagnosticState:
    """处理人工追问。"""
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
    if has_real_llm(state):
        llm = get_llm(state.model_provider, state.model_name)
        history = state.messages[:]
        history.append(HumanMessage(content=state.human_question))
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是资深 DevOps 值班工程师。结合上下文回答追问，保持专业、准确、简洁，控制在 3-5 句话。\n"
                    "如果涉及自动修复，明确说明仍需人工批准。\n"
                    "上下文如下:\n{context}",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        response = llm.invoke(prompt.format_messages(context=context, messages=history))
        update_token_usage_from_message(state, response)
        answer = response.content
    else:
        answer = (
            f"当前会话没有在调用真实大模型，原因: {state.llm_status or '模型不可用'}。"
            f"所以现在不能根据你的追问生成语义化回答，只能返回本地规则诊断结果。"
            f"当前根因判断: {state.root_cause or '暂无'}；"
            f"建议: {state.recommendations[0] if state.recommendations else '补充更多日志后再判断'}。"
        )
    state.messages.append(HumanMessage(content=state.human_question))
    state.messages.append(AIMessage(content=answer))
    state.conversation_count += 1
    state.human_question = None
    # 等待人工输入为true
    state.awaiting_human_input = True
    logger.info(
        "agent.node.end node=conversation session_id=%s conversation_count=%s",
        state.event.id,
        state.conversation_count,
    )
    return state


def decide_action(state: DiagnosticState) -> DiagnosticState:
    """把人工决策翻译成后续执行策略。"""
    logger.info(
        "agent.node.start node=decide session_id=%s human_decision=%s severity=%s auto_fix_action=%s",
        state.event.id,
        state.human_decision,
        state.severity,
        state.auto_fix_action,
    )
    state.should_auto_fix = state.human_decision == "approve" and state.auto_fix_action is not None
    state.should_alert = state.severity in {"CRITICAL", "HIGH", "MEDIUM"} or state.human_decision in {
        "approve",
        "reject",
    }
    state.awaiting_human_input = False
    state.approval_required = False
    state.diagnostic_report = build_report(state)
    logger.info(
        "agent.node.end node=decide session_id=%s should_auto_fix=%s should_alert=%s",
        state.event.id,
        state.should_auto_fix,
        state.should_alert,
    )
    return state


def execute_auto_fix(state: DiagnosticState) -> DiagnosticState:
    """执行自动修复动作。

    注意：真正是否允许执行，前面已经由审批逻辑和硬规则一起决定。
    """
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
    """发送通知告警。"""
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


def should_continue_tool_loop(state: DiagnosticState) -> str:
    """判断首轮诊断要继续调工具还是直接收束结果。"""
    last_message = state.agent_messages[-1] if state.agent_messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        route = "tools"
    else:
        route = "finalize"
    logger.info(
        "agent.route route=should_continue_tool_loop session_id=%s next=%s",
        state.event.id,
        route,
    )
    return route


def should_continue_conversation(state: DiagnosticState) -> str:
    """判断人工阶段下一步是继续追问、进入决策还是继续等待。"""
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
    """决定后续走自动修复还是只发告警。"""
    route = "fix" if state.should_auto_fix else "alert"
    logger.info(
        "agent.route route=should_auto_fix_route session_id=%s should_auto_fix=%s next=%s",
        state.event.id,
        state.should_auto_fix,
        route,
    )
    return route


def build_report(state: DiagnosticState) -> str:
    """把最终诊断结果整理成可展示的文本报告。"""
    decision = "执行自动修复" if state.should_auto_fix else "人工介入/仅告警"
    return (
        "# 故障诊断报告\n\n"
        f"- 事件 ID: {state.event.id}\n"
        f"- 来源: {state.event.log_event.source}\n"
        f"- 错误类型: {state.error_category}\n"
        f"- 严重程度: {state.severity}\n"
        f"- 根本原因: {state.root_cause}\n"
        f"- 影响分析: {state.impact_analysis}\n"
        f"- 修复建议: {' | '.join(state.recommendations)}\n"
        f"- 建议自动修复: {state.auto_fix_action or '无'}\n"
        f"- 安全规则: 自动修复必须人工批准后执行\n"
        f"- 人工对话轮数: {state.conversation_count}\n"
        f"- 最终决策: {decision}\n"
    )


def build_conversation_workflow():
    """组装 LangGraph 工作流并配置人工中断点。"""
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("llm_diagnostic", llm_diagnostic_node)
    workflow.add_node("execute_tools", execute_requested_tools)
    workflow.add_node("finalize_diagnosis", finalize_diagnosis)
    workflow.add_node("human_decision", human_decision_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("decide", decide_action)
    workflow.add_node("auto_fix", execute_auto_fix)
    workflow.add_node("alert", send_alert_node)

    workflow.set_entry_point("llm_diagnostic")
    # 从 llm_diagnostic 节点执行完后，把当前 state 交给 should_continue_tool_loop 做决策，
    # 然后根据返回值，在第三个参数的映射表中找到对应的下一节点执行
    workflow.add_conditional_edges(
        "llm_diagnostic",
        should_continue_tool_loop,
        {"tools": "execute_tools", "finalize": "finalize_diagnosis"},
    )
    # 如果调用了工具，则在工具执行后回到 LLM 节点，继续基于新信息推理
    workflow.add_edge("execute_tools", "llm_diagnostic")
    # 当 LLM 判断无需再调用工具并生成最终结果后，进入人工决策环节
    workflow.add_edge("finalize_diagnosis", "human_decision")
    # === Human-in-the-loop 路由 ===
    # 在 human_decision 节点恢复执行（拿到人工输入）后，
    # 将当前 state 交给 should_continue_conversation 做决策，
    # 并根据返回值从映射表中选择下一步执行的节点
    workflow.add_conditional_edges(
        "human_decision",
        should_continue_conversation,
        {
            "conversation": "conversation",  # 进入对话节点，继续回答用户问题
            "decide": "decide",  # 进入最终决策/收尾逻辑
            "wait": "human_decision",  # 保持在当前节点（通常表示继续等待人工输入）
        },
    )

    # 对话完成后，回到 human_decision，形成“对话 ↔ 人工决策”的循环
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
            ("models", "TokenUsage"),
        ]
    )
    # - checkpointer：通过 MemorySaver 保存执行状态（使用 JsonPlusSerializer 处理自定义类型） 生产最好采用PostgresSaver 或 MongoDB 或者继承BaseCheckpointSaver
    # - interrupt_before：在进入 human_decision 前暂停，等待外部输入后从该节点恢复执行
    return workflow.compile(
        checkpointer=MemorySaver(serde=serde),
        interrupt_before=["human_decision"],
    )


app = build_conversation_workflow()
