from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ContainerStats(BaseModel):
    container_id: str = ""
    container_name: str
    cpu_percent: float = 0.0
    memory_usage: int = 0
    memory_limit: int = 0
    memory_percent: float = 0.0
    restart_count: int = 0
    status: str = "unknown"


class LogEvent(BaseModel):
    id: str
    timestamp: datetime
    source: str
    level: str = "ERROR"
    message: str
    error_type: str
    container_id: Optional[str] = None


class DiagnosticEvent(BaseModel):
    id: str
    log_event: LogEvent
    container_stats: ContainerStats
    recent_logs: list[str] = Field(default_factory=list)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionDecision(BaseModel):
    decision: Literal["approve", "reject", "continue"]


class DiagnosticSummary(BaseModel):
    session_id: str
    status: str
    error_message: Optional[str] = None
    root_cause: Optional[str] = None
    severity: Optional[str] = None
    recommendations: list[str] = Field(default_factory=list)
    auto_fix_action: Optional[str] = None
    should_auto_fix: bool = False
    should_alert: bool = False


class DiagnosticState(BaseModel):
    event: DiagnosticEvent
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
    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)
    human_question: Optional[str] = None
    conversation_count: int = 0
    awaiting_human_input: bool = False
    human_decision: Optional[Literal["approve", "reject", "continue"]] = None
    execution_log: list[str] = Field(default_factory=list)


def build_initial_state(event: DiagnosticEvent) -> DiagnosticState:
    return DiagnosticState(event=event)


def coerce_state(value: DiagnosticState | dict) -> DiagnosticState:
    if isinstance(value, DiagnosticState):
        return value
    return DiagnosticState.model_validate(value)
