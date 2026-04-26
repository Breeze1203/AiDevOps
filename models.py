from __future__ import annotations
"""项目核心数据模型。

这个文件定义了三类对象：
1. 输入事件模型：Kafka / API 进入系统的诊断事件。
2. 会话摘要模型：前端列表页需要的轻量信息。
3. Agent 状态模型：LangGraph 在节点之间传递和持久化的完整状态。
"""

from datetime import datetime
from typing import Any, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from llm_runtime import get_runtime_model


class TokenUsage(BaseModel):
    """记录一个 session 累积消耗的 token。"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ContainerStats(BaseModel):
    """故障发生时容器的资源使用快照。"""
    container_id: str = ""
    container_name: str
    cpu_percent: float = 0.0
    memory_usage: int = 0
    memory_limit: int = 0
    memory_percent: float = 0.0
    restart_count: int = 0
    status: str = "unknown"


class LogEvent(BaseModel):
    """单条原始日志事件。"""
    id: str
    timestamp: datetime
    source: str
    level: str = "ERROR"
    message: str
    error_type: str
    container_id: Optional[str] = None


class DiagnosticEvent(BaseModel):
    """进入诊断系统的标准事件结构。"""
    id: str
    log_event: LogEvent
    container_stats: ContainerStats
    recent_logs: list[str] = Field(default_factory=list)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionDecision(BaseModel):
    """人工审批接口里允许的决策动作。"""
    decision: Literal["approve", "reject", "continue"]


class DiagnosticSummary(BaseModel):
    """前端会话列表和详情头部使用的摘要信息。"""
    session_id: str
    status: str
    error_message: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    llm_enabled: bool = False
    llm_status: Optional[str] = None
    root_cause: Optional[str] = None
    severity: Optional[str] = None
    recommendations: list[str] = Field(default_factory=list)
    auto_fix_action: Optional[str] = None
    approval_required: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    should_auto_fix: bool = False
    should_alert: bool = False


class DiagnosticState(BaseModel):
    """LangGraph 在整个诊断流程中流转的完整状态。"""
    event: DiagnosticEvent
    model_provider: str = "gemini"
    model_name: str = ""
    llm_enabled: bool = False
    llm_status: Optional[str] = None
    error_category: Optional[str] = None
    severity: Optional[str] = None
    additional_logs: list[str] = Field(default_factory=list)
    similar_cases: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_base_results: list[str] = Field(default_factory=list)
    root_cause: Optional[str] = None
    impact_analysis: Optional[str] = None
    recommendations: list[str] = Field(default_factory=list)
    auto_fix_action: Optional[str] = None
    diagnostic_report: Optional[str] = None
    should_auto_fix: bool = False
    should_alert: bool = False
    approval_required: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    agent_messages: list[BaseMessage] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    human_question: Optional[str] = None
    conversation_count: int = 0
    awaiting_human_input: bool = False
    human_decision: Optional[Literal["approve", "reject", "continue"]] = None
    execution_log: list[str] = Field(default_factory=list)


def build_initial_state(
    event: DiagnosticEvent,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> DiagnosticState:
    """把输入事件包装成初始 agent 状态。"""
    runtime = get_runtime_model()
    return DiagnosticState(
        event=event,
        model_provider=model_provider or runtime["provider"],
        model_name=model_name or runtime["model_name"],
    )


def coerce_state(value: DiagnosticState | dict) -> DiagnosticState:
    """把 dict / model 两种输入统一收敛成 DiagnosticState。"""
    if isinstance(value, DiagnosticState):
        return value
    return DiagnosticState.model_validate(value)
