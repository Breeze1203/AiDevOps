"""项目级集成测试。

这些测试主要覆盖三类行为：
1. Kafka 转发与失败重试。
2. API + LangGraph 的首轮诊断 / 追问 / 审批流程。
3. 模型配置、消息去重和 fallback 行为等回归场景。
"""

import contextlib
import json
import os
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError

from fastapi.testclient import TestClient

from api import SESSION_LOCKS, SESSION_STORE, api_app
from llm_runtime import set_runtime_model
from main import forward_event_to_api, forward_event_with_retry
from run import main as run_main


def build_sample_event(event_id: str) -> dict:
    """构造一条固定的示例故障事件。"""
    return {
        "id": event_id,
        "timestamp": "2026-04-22T10:00:00",
        "log_event": {
            "id": f"log-{event_id}",
            "timestamp": "2026-04-22T10:00:00",
            "source": "app",
            "level": "ERROR",
            "message": "java.lang.OutOfMemoryError: GC overhead limit exceeded",
            "error_type": "oom_error",
            "container_id": "c-1",
        },
        "container_stats": {
            "container_id": "c-1",
            "container_name": "payment-service",
            "cpu_percent": 82.4,
            "memory_usage": 1835008000,
            "memory_limit": 2147483648,
            "memory_percent": 95.7,
            "restart_count": 2,
            "status": "running",
        },
        "recent_logs": [
            "ERROR OutOfMemoryError: GC overhead limit exceeded",
            "WARN Full GC took 8.1s",
            "ERROR Request timeout for order submit",
        ],
    }


def wait_for_status(client: TestClient, session_id: str, expected: str, timeout: float = 2.0):
    """轮询 session 状态，直到进入目标状态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/session/{session_id}")
        if response.status_code == 200 and response.json()["summary"]["status"] == expected:
            return response
        time.sleep(0.05)
    raise AssertionError(f"session {session_id} did not reach status {expected}")


class DiagnosticWorkflowTests(unittest.TestCase):
    """覆盖主要业务链路的测试集合。"""

    def setUp(self):
        SESSION_STORE.clear()
        SESSION_LOCKS.clear()
        set_runtime_model("gemini", "gemini-2.5-flash-lite")
        self.failed_queue_path = "/tmp/ai-devops-failed-events-test.jsonl"
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.failed_queue_path)

    def test_dashboard_page_is_served(self):
        client = TestClient(api_app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI DevOps 值班台", response.text)

    def test_run_module_is_importable(self):
        self.assertTrue(callable(run_main))

    def test_forward_event_calls_api(self):
        expected = {
            "session_id": "evt-test-001",
            "summary": {"status": "queued"},
        }
        with patch("main.build_opener") as mock_build_opener:
            mock_opener = mock_build_opener.return_value
            mock_response = mock_opener.open.return_value.__enter__.return_value
            mock_response.read.return_value = str(expected).replace("'", '"').encode("utf-8")
            result = forward_event_to_api(build_sample_event("evt-test-001"))
        self.assertEqual(result["session_id"], "evt-test-001")
        self.assertEqual(result["summary"]["status"], "queued")

    def test_forward_event_retries_then_succeeds(self):
        expected = {
            "session_id": "evt-test-retry",
            "summary": {"status": "queued"},
        }
        with patch("main.build_opener") as mock_build_opener, patch("main.time.sleep") as mock_sleep:
            first = URLError("temporary failure")
            mock_opener = mock_build_opener.return_value
            mock_response = mock_opener.open.return_value.__enter__.return_value
            mock_response.read.return_value = str(expected).replace("'", '"').encode("utf-8")
            mock_opener.open.side_effect = [first, mock_opener.open.return_value]
            with patch.dict(
                "os.environ",
                {"API_FORWARD_MAX_RETRIES": "2", "API_FORWARD_RETRY_DELAY_SECONDS": "0"},
                clear=False,
            ):
                result = forward_event_with_retry(build_sample_event("evt-test-retry"))
        self.assertEqual(result["session_id"], "evt-test-retry")
        self.assertEqual(mock_sleep.call_count, 1)

    def test_forward_event_writes_failed_queue_after_retries(self):
        with patch("main.build_opener") as mock_build_opener, patch(
            "main.time.sleep"
        ), patch.dict(
            "os.environ",
            {
                "API_FORWARD_MAX_RETRIES": "1",
                "API_FORWARD_RETRY_DELAY_SECONDS": "0",
                "FAILED_EVENT_QUEUE_PATH": self.failed_queue_path,
            },
            clear=False,
        ):
            mock_build_opener.return_value.open.side_effect = URLError("downstream unavailable")
            with self.assertRaises(RuntimeError):
                forward_event_with_retry(build_sample_event("evt-test-dead-letter"))

        with open(self.failed_queue_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["event"]["id"], "evt-test-dead-letter")
        self.assertIn("downstream unavailable", payload["error_message"])

    def test_api_flow_supports_question_and_approval(self):
        client = TestClient(api_app)

        start = client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-002")})
        self.assertEqual(start.status_code, 200)
        self.assertIn(start.json()["summary"]["status"], {"queued", "running"})

        wait_for_status(client, "evt-test-002", "awaiting_decision")
        waiting = client.get("/api/session/evt-test-002")
        self.assertTrue(waiting.json()["summary"]["approval_required"])

        ask = client.post(
            "/api/conversation/ask",
            json={"session_id": "evt-test-002", "question": "为什么会 OOM？"},
        )
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["conversation_count"], 1)
        self.assertEqual(ask.json()["summary"]["status"], "awaiting_decision")

        decision = client.post(
            "/api/decision/submit",
            json={"session_id": "evt-test-002", "decision": "approve"},
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["summary"]["status"], "completed")
        self.assertFalse(decision.json()["summary"]["approval_required"])
        self.assertEqual(decision.json()["summary"]["model_provider"], "gemini")
        self.assertEqual(decision.json()["summary"]["model_name"], "gemini-2.5-flash-lite")
        self.assertEqual(decision.json()["summary"]["token_usage"]["total_tokens"], 0)
        self.assertTrue(decision.json()["execution_log"])

    def test_runtime_model_can_be_switched(self):
        client = TestClient(api_app)
        current = client.get("/api/runtime/model")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["current"]["provider"], "gemini")

        updated = client.post(
            "/api/runtime/model",
            json={"provider": "chatgpt", "model_name": "gpt-4o-mini"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["current"]["provider"], "chatgpt")
        self.assertEqual(updated.json()["current"]["model_name"], "gpt-4o-mini")

        start = client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-runtime-switch")})
        self.assertEqual(start.status_code, 200)

        wait_for_status(client, "evt-test-runtime-switch", "awaiting_decision")
        session = client.get("/api/session/evt-test-runtime-switch")
        self.assertEqual(session.status_code, 200)
        self.assertEqual(session.json()["summary"]["model_provider"], "chatgpt")
        self.assertEqual(session.json()["summary"]["model_name"], "gpt-4o-mini")

    def test_auto_fix_stays_pending_until_manual_approval(self):
        client = TestClient(api_app)
        client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-approval-gate")})
        wait_for_status(client, "evt-test-approval-gate", "awaiting_decision")

        before = client.get("/api/session/evt-test-approval-gate")
        self.assertEqual(before.status_code, 200)
        self.assertFalse(before.json()["summary"]["should_auto_fix"])
        self.assertTrue(before.json()["summary"]["approval_required"])

        ask = client.post(
            "/api/conversation/ask",
            json={"session_id": "evt-test-approval-gate", "question": "现在可以直接自动修复吗？"},
        )
        self.assertEqual(ask.status_code, 200)
        self.assertEqual(ask.json()["summary"]["status"], "awaiting_decision")

        after = client.get("/api/session/evt-test-approval-gate")
        self.assertEqual(after.status_code, 200)
        self.assertFalse(after.json()["summary"]["should_auto_fix"])
        self.assertTrue(after.json()["summary"]["approval_required"])

    def test_conversation_messages_do_not_duplicate_after_multiple_questions(self):
        client = TestClient(api_app)
        client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-conversation-dedupe")})
        wait_for_status(client, "evt-test-conversation-dedupe", "awaiting_decision")

        first = client.post(
            "/api/conversation/ask",
            json={"session_id": "evt-test-conversation-dedupe", "question": "第一次追问"},
        )
        self.assertEqual(first.status_code, 200)

        second = client.post(
            "/api/conversation/ask",
            json={"session_id": "evt-test-conversation-dedupe", "question": "第二次追问"},
        )
        self.assertEqual(second.status_code, 200)

        session = client.get("/api/session/evt-test-conversation-dedupe")
        self.assertEqual(session.status_code, 200)
        messages = session.json()["messages"]
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], "第一次追问")
        self.assertEqual(messages[2], "第二次追问")
        self.assertEqual(messages.count("第一次追问"), 1)
        self.assertEqual(messages.count("第二次追问"), 1)

    def test_fallback_conversation_answer_explains_llm_is_not_enabled(self):
        client = TestClient(api_app)
        client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-fallback-reason")})
        wait_for_status(client, "evt-test-fallback-reason", "awaiting_decision")

        ask = client.post(
            "/api/conversation/ask",
            json={"session_id": "evt-test-fallback-reason", "question": "这个根因为什么成立？"},
        )
        self.assertEqual(ask.status_code, 200)
        self.assertIn("没有在调用真实大模型", ask.json()["answer"])
        self.assertIn("GOOGLE_API_KEY", ask.json()["answer"])

        session = client.get("/api/session/evt-test-fallback-reason")
        self.assertEqual(session.status_code, 200)
        self.assertFalse(session.json()["summary"]["llm_enabled"])
        self.assertIn("GOOGLE_API_KEY", session.json()["summary"]["llm_status"])

    def test_two_sessions_can_progress_independently(self):
        client = TestClient(api_app)
        first = client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-101")})
        second = client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-102")})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)

        wait_for_status(client, "evt-test-101", "awaiting_decision")
        wait_for_status(client, "evt-test-102", "awaiting_decision")

        decision = client.post(
            "/api/decision/submit",
            json={"session_id": "evt-test-101", "decision": "approve"},
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["summary"]["status"], "completed")

        untouched = client.get("/api/session/evt-test-102")
        self.assertEqual(untouched.status_code, 200)
        self.assertEqual(untouched.json()["summary"]["status"], "awaiting_decision")

    def test_websocket_flow_supports_question_and_approval(self):
        client = TestClient(api_app)
        client.post("/api/diagnostic/start", json={"event": build_sample_event("evt-test-003")})
        wait_for_status(client, "evt-test-003", "awaiting_decision")

        with client.websocket_connect("/ws/session/evt-test-003") as websocket:
            snapshot = websocket.receive_json()
            self.assertEqual(snapshot["type"], "snapshot")
            websocket.send_json({"action": "ask", "question": "为什么会 OOM？"})
            first_ask = websocket.receive_json()
            if first_ask["type"] != "ask_result":
                second_ask = websocket.receive_json()
                self.assertEqual(second_ask["type"], "ask_result")
            websocket.send_json({"action": "approve"})
            first_decision = websocket.receive_json()
            if first_decision["type"] != "decision_result":
                second_decision = websocket.receive_json()
                self.assertEqual(second_decision["type"], "decision_result")


if __name__ == "__main__":
    unittest.main()
