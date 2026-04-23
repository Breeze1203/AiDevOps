from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import app
from main import main as kafka_forwarder_main
from models import DiagnosticEvent, DiagnosticSummary, build_initial_state, coerce_state
from session_store import build_session_store

api_app = FastAPI(title="AI DevOps Diagnostic API")
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
EXECUTOR = ThreadPoolExecutor(max_workers=8)
API_LOOP: asyncio.AbstractEventLoop | None = None
KAFKA_FORWARDER_THREAD: threading.Thread | None = None

SESSION_STORE = build_session_store()
SESSION_LOCKS: dict[str, threading.Lock] = {}
SESSION_LOCKS_GUARD = threading.Lock()

SESSION_QUEUED = "queued"
SESSION_RUNNING = "running"
SESSION_AWAITING_DECISION = "awaiting_decision"
SESSION_COMPLETED = "completed"
SESSION_FAILED = "failed"

logger = logging.getLogger("ai_devops.api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] [thread=%(threadName)s] %(message)s",
    )

class ConnectionManager:
    def __init__(self):
        self.dashboard_connections: list[WebSocket] = []
        self.session_connections: dict[str, list[WebSocket]] = {}

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_connections.append(websocket)

    async def connect_session(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.session_connections.setdefault(session_id, []).append(websocket)

    def disconnect_dashboard(self, websocket: WebSocket):
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)

    def disconnect_session(self, session_id: str, websocket: WebSocket):
        sockets = self.session_connections.get(session_id, [])
        if websocket in sockets:
            sockets.remove(websocket)
        if not sockets and session_id in self.session_connections:
            del self.session_connections[session_id]

    async def _safe_send(self, websocket: WebSocket, payload: dict):
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    async def broadcast_dashboard(self, payload: dict):
        for websocket in list(self.dashboard_connections):
            await self._safe_send(websocket, payload)

    async def broadcast_session(self, session_id: str, payload: dict):
        for websocket in list(self.session_connections.get(session_id, [])):
            await self._safe_send(websocket, payload)


manager = ConnectionManager()


class StartDiagnosticRequest(BaseModel):
    event: DiagnosticEvent


class AskQuestionRequest(BaseModel):
    session_id: str
    question: str


class MakeDecisionRequest(BaseModel):
    session_id: str
    decision: Literal["approve", "reject", "continue"]


def _start_kafka_forwarder_once() -> None:
    global KAFKA_FORWARDER_THREAD
    if os.getenv("DISABLE_EMBEDDED_KAFKA_FORWARDER", "0") == "1":
        logger.info("kafka_forwarder.skip embedded forwarder disabled by env")
        return
    if KAFKA_FORWARDER_THREAD is not None and KAFKA_FORWARDER_THREAD.is_alive():
        logger.info("kafka_forwarder.skip already running")
        return

    def run_kafka_forwarder():
        try:
            logger.info("kafka_forwarder.thread_start")
            kafka_forwarder_main()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            logger.exception("kafka_forwarder.thread_exit error=%s", exc)

    KAFKA_FORWARDER_THREAD = threading.Thread(
        target=run_kafka_forwarder,
        name="kafka-forwarder",
        daemon=True,
    )
    KAFKA_FORWARDER_THREAD.start()
    logger.info("kafka_forwarder.started")


def _get_session_lock(session_id: str) -> threading.Lock:
    with SESSION_LOCKS_GUARD:
        if session_id not in SESSION_LOCKS:
            SESSION_LOCKS[session_id] = threading.Lock()
        return SESSION_LOCKS[session_id]


def _session_payload(session_id: str, session: dict) -> dict:
    state = coerce_state(session["state"])
    return {
        "type": "session_update",
        "session_id": session_id,
        "summary": _to_summary(session_id, session).model_dump(),
        "messages": [getattr(msg, "content", "") for msg in state.messages],
        "execution_log": state.execution_log,
        "report": state.diagnostic_report,
    }


def _get_session(session_id: str) -> dict:
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


def _save_session(session_id: str, session: dict) -> None:
    SESSION_STORE.set(session_id, session)


def _to_summary(session_id: str, session: dict) -> DiagnosticSummary:
    state = coerce_state(session["state"])
    return DiagnosticSummary(
        session_id=session_id,
        status=session["status"],
        error_message=session.get("error_message"),
        root_cause=state.root_cause,
        severity=state.severity,
        recommendations=state.recommendations,
        auto_fix_action=state.auto_fix_action,
        should_auto_fix=state.should_auto_fix,
        should_alert=state.should_alert,
    )


async def _publish_session_update(session_id: str) -> None:
    session = _get_session(session_id)
    payload = _session_payload(session_id, session)
    await manager.broadcast_session(session_id, payload)
    await manager.broadcast_dashboard(payload)


def _notify_session_update(session_id: str):
    if API_LOOP is None or API_LOOP.is_closed() or not API_LOOP.is_running():
        return
    coro = _publish_session_update(session_id)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, API_LOOP)
        future.result(timeout=5)
    except Exception:
        coro.close()


def _invoke_graph(input_value, config: dict):
    thread_id = config.get("configurable", {}).get("thread_id")
    phase = "resume" if input_value is None else "initial"
    logger.info("langgraph.invoke.start session_id=%s phase=%s", thread_id, phase)
    result = app.invoke(input_value, config)
    result_state = coerce_state(result)
    logger.info(
        "langgraph.invoke.end session_id=%s phase=%s awaiting_human_input=%s conversation_count=%s",
        thread_id,
        phase,
        result_state.awaiting_human_input,
        result_state.conversation_count,
    )
    return result


def _set_session_status_sync(session_id: str, status: str, error_message: str | None = None):
    session = _get_session(session_id)
    previous_status = session["status"]
    session["status"] = status
    session["error_message"] = error_message
    _save_session(session_id, session)
    logger.info(
        "session.status_change session_id=%s from=%s to=%s error=%s",
        session_id,
        previous_status,
        status,
        error_message,
    )
    _notify_session_update(session_id)


def _run_initial_diagnostic_sync(session_id: str):
    with _get_session_lock(session_id):
        try:
            logger.info("session.initial.begin session_id=%s", session_id)
            _set_session_status_sync(session_id, SESSION_RUNNING)
            session = _get_session(session_id)
            result = _invoke_graph(session["initial_state"], session["config"])
            session["state"] = result
            result_state = coerce_state(result)
            if result_state.awaiting_human_input:
                session["status"] = SESSION_AWAITING_DECISION
            else:
                session["status"] = SESSION_COMPLETED
            session["error_message"] = None
            logger.info(
                "session.initial.end session_id=%s final_status=%s awaiting_human_input=%s",
                session_id,
                session["status"],
                result_state.awaiting_human_input,
            )
        except Exception as exc:
            session = _get_session(session_id)
            session["status"] = SESSION_FAILED
            session["error_message"] = str(exc)
            logger.exception("session.initial.fail session_id=%s error=%s", session_id, exc)
        _save_session(session_id, session)
        _notify_session_update(session_id)


def _continue_session_sync(session_id: str, state_update: dict):
    with _get_session_lock(session_id):
        session = _get_session(session_id)
        if session["status"] == SESSION_RUNNING:
            raise HTTPException(status_code=409, detail="该会话仍在执行中，请稍后重试")
        if session["status"] == SESSION_FAILED:
            raise HTTPException(status_code=409, detail="该会话已失败，无法继续")

        logger.info("session.continue.begin session_id=%s update=%s", session_id, state_update)
        session["status"] = SESSION_RUNNING
        session["error_message"] = None
        _save_session(session_id, session)
        _notify_session_update(session_id)

        try:
            app.update_state(session["config"], state_update)
            result = _invoke_graph(None, session["config"])
            session["state"] = result
            result_state = coerce_state(result)
            if result_state.awaiting_human_input:
                session["status"] = SESSION_AWAITING_DECISION
            else:
                session["status"] = SESSION_COMPLETED
            session["error_message"] = None
            logger.info(
                "session.continue.end session_id=%s final_status=%s awaiting_human_input=%s conversation_count=%s",
                session_id,
                session["status"],
                result_state.awaiting_human_input,
                result_state.conversation_count,
            )
        except Exception as exc:
            session["status"] = SESSION_FAILED
            session["error_message"] = str(exc)
            _save_session(session_id, session)
            _notify_session_update(session_id)
            logger.exception("session.continue.fail session_id=%s error=%s", session_id, exc)
            raise exc

        _save_session(session_id, session)
        _notify_session_update(session_id)
        return coerce_state(session["state"])



@api_app.on_event("startup")
async def on_startup():
    logger.info("api.startup.begin")
    _start_kafka_forwarder_once()
    logger.info("api.startup.end")


@api_app.get("/")
async def dashboard_page():
    return FileResponse(FRONTEND_FILE)


@api_app.get("/api/sessions")
async def list_sessions():
    items = []
    for session_id, session in SESSION_STORE.list():
        items.append(_to_summary(session_id, session).model_dump())
    return {"sessions": items}


@api_app.post("/api/diagnostic/start")
async def start_diagnostic(request: StartDiagnosticRequest):
    _start_kafka_forwarder_once()
    global API_LOOP
    if API_LOOP is None:
        API_LOOP = asyncio.get_running_loop()

    event = request.event
    session_id = event.id
    if SESSION_STORE.exists(session_id):
        raise HTTPException(status_code=409, detail="会话已存在，请勿重复创建")

    logger.info("session.create session_id=%s source=%s", session_id, event.log_event.source)
    config = {"configurable": {"thread_id": session_id}}
    session = {
        "state": build_initial_state(event),
        "initial_state": build_initial_state(event),
        "config": config,
        "status": SESSION_QUEUED,
        "error_message": None,
    }
    _save_session(session_id, session)
    await _publish_session_update(session_id)
    logger.info("session.submit_initial_to_executor session_id=%s", session_id)
    EXECUTOR.submit(_run_initial_diagnostic_sync, session_id)

    return {
        "session_id": session_id,
        "summary": _to_summary(session_id, session).model_dump(),
        "report": None,
    }


@api_app.post("/api/conversation/ask")
async def ask_question(request: AskQuestionRequest):
    session = _get_session(request.session_id)
    if session["status"] != SESSION_AWAITING_DECISION:
        raise HTTPException(status_code=409, detail=f"当前会话状态为 {session['status']}，不能追问")

    global API_LOOP
    if API_LOOP is None:
        API_LOOP = asyncio.get_running_loop()

    try:
        loop = asyncio.get_running_loop()
        logger.info("session.submit_continue_to_executor session_id=%s mode=ask", request.session_id)
        result_state = await loop.run_in_executor(
            EXECUTOR,
            partial(
                _continue_session_sync,
                request.session_id,
                {"human_question": request.question, "human_decision": "continue"},
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"会话执行失败: {exc}") from exc
    return {
        "session_id": request.session_id,
        "answer": result_state.messages[-1].content if result_state.messages else "",
        "conversation_count": result_state.conversation_count,
        "summary": _to_summary(request.session_id, _get_session(request.session_id)).model_dump(),
    }


@api_app.post("/api/decision/submit")
async def submit_decision(request: MakeDecisionRequest):
    if request.decision == "continue":
        raise HTTPException(status_code=400, detail="请使用 /api/conversation/ask 继续追问")

    session = _get_session(request.session_id)
    if session["status"] != SESSION_AWAITING_DECISION:
        raise HTTPException(status_code=409, detail=f"当前会话状态为 {session['status']}，不能审批")

    global API_LOOP
    if API_LOOP is None:
        API_LOOP = asyncio.get_running_loop()

    try:
        loop = asyncio.get_running_loop()
        logger.info(
            "session.submit_continue_to_executor session_id=%s mode=decision decision=%s",
            request.session_id,
            request.decision,
        )
        result_state = await loop.run_in_executor(
            EXECUTOR,
            partial(_continue_session_sync, request.session_id, {"human_decision": request.decision}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"会话执行失败: {exc}") from exc
    return {
        "session_id": request.session_id,
        "decision": request.decision,
        "summary": _to_summary(request.session_id, _get_session(request.session_id)).model_dump(),
        "execution_log": result_state.execution_log,
        "report": result_state.diagnostic_report,
    }


@api_app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = _get_session(session_id)
    state = coerce_state(session["state"])
    return {
        "session_id": session_id,
        "summary": _to_summary(session_id, session).model_dump(),
        "messages": [getattr(msg, "content", "") for msg in state.messages],
        "execution_log": state.execution_log,
        "report": state.diagnostic_report,
    }


@api_app.websocket("/ws/dashboard")
async def dashboard_socket(websocket: WebSocket):
    await manager.connect_dashboard(websocket)
    try:
        current = []
        for session_id, session in SESSION_STORE.list():
            current.append(_to_summary(session_id, session).model_dump())
        await websocket.send_json({"type": "snapshot", "sessions": current})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_dashboard(websocket)


@api_app.websocket("/ws/session/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await manager.connect_session(session_id, websocket)
    try:
        if SESSION_STORE.exists(session_id):
            await websocket.send_json(
                {
                    "type": "snapshot",
                    **(await get_session(session_id)),
                }
            )
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            action = payload.get("action")
            if action == "ask":
                response = await ask_question(
                    AskQuestionRequest(session_id=session_id, question=payload["question"])
                )
                await websocket.send_json({"type": "ask_result", **response})
            elif action == "approve":
                response = await submit_decision(
                    MakeDecisionRequest(session_id=session_id, decision="approve")
                )
                await websocket.send_json({"type": "decision_result", **response})
            elif action == "reject":
                response = await submit_decision(
                    MakeDecisionRequest(session_id=session_id, decision="reject")
                )
                await websocket.send_json({"type": "decision_result", **response})
            elif action == "get_state":
                response = await get_session(session_id)
                await websocket.send_json({"type": "state", **response})
            else:
                await websocket.send_json({"type": "error", "message": f"未知 action: {action}"})
    except WebSocketDisconnect:
        manager.disconnect_session(session_id, websocket)
