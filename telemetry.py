from __future__ import annotations
"""可选的 LangSmith tracing 初始化。

这个模块被 API 入口尽早导入，目的是在 agent / model 初始化之前就把
LangSmith tracing 的环境准备好。
"""

import logging
import os

logger = logging.getLogger("ai_devops.telemetry")
_INITIALIZED = False


def init_langsmith() -> None:
    """按环境变量决定是否启用 LangSmith tracing。"""
    global _INITIALIZED
    if _INITIALIZED:
        return

    api_key = os.getenv("LANGSMITH_API_KEY")
    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    project = os.getenv("LANGSMITH_PROJECT")

    if not api_key and not endpoint and not project:
        logger.info("telemetry.skip reason=no_langsmith_config")
        _INITIALIZED = True
        return

    # LangChain / LangGraph tracing integrates through env vars.
    # If the user supplied LangSmith config, enable tracing by default unless explicitly disabled.
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    try:
        import langsmith  # noqa: F401

        logger.info("telemetry.enabled provider=langsmith")
    except Exception as exc:  # pragma: no cover
        logger.warning("telemetry.skip reason=langsmith_sdk_unavailable error=%s", exc)
    finally:
        _INITIALIZED = True


init_langsmith()
