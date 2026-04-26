from __future__ import annotations
"""运行时模型配置。

这个模块解决两个问题：
1. 项目启动时，默认应该用哪个 provider / model。
2. 前端切换模型后，新的 session 应该使用什么配置。

注意：这里管理的是“当前项目默认模型”，不是覆盖已经创建好的旧 session。
"""

import os
import threading
from typing import Any

SUPPORTED_PROVIDERS = {
    # 每个 provider 都声明：
    # - 对前端展示的 label
    # - 需要的环境变量
    # - 默认模型名
    "chatgpt": {
        "label": "ChatGPT",
        "env_keys": ["OPENAI_API_KEY"],
        "default_model": lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    },
    "gemini": {
        "label": "Gemini",
        "env_keys": ["GOOGLE_API_KEY"],
        "default_model": lambda: os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
    },
    "cloudecode": {
        "label": "CloudeCode",
        "env_keys": ["CLOUDECODE_API_KEY", "ANTHROPIC_API_KEY"],
        "default_model": lambda: os.getenv("CLOUDECODE_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")),
    },
}

_RUNTIME_LOCK = threading.Lock()
_RUNTIME_MODEL = {
    "provider": os.getenv("LLM_PROVIDER", "").strip().lower() or None,
    "model_name": os.getenv("LLM_MODEL", "").strip() or None,
}


def _pick_default_provider() -> str:
    """优先选择当前环境里已经具备 API key 的 provider。"""
    for provider, meta in SUPPORTED_PROVIDERS.items():
        if any(os.getenv(key, "").strip() for key in meta["env_keys"]):
            return provider
    return "gemini"


def normalize_runtime_model(provider: str | None = None, model_name: str | None = None) -> dict[str, str]:
    """把外部输入标准化成合法的 provider / model 组合。"""
    normalized_provider = (provider or "").strip().lower() or _RUNTIME_MODEL.get("provider") or _pick_default_provider()
    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"不支持的模型提供商: {normalized_provider}")

    normalized_model_name = (model_name or "").strip()
    if not normalized_model_name:
        normalized_model_name = _RUNTIME_MODEL.get("model_name") or SUPPORTED_PROVIDERS[normalized_provider][
            "default_model"
        ]()

    return {"provider": normalized_provider, "model_name": normalized_model_name}


def get_runtime_model() -> dict[str, str]:
    """读取当前默认模型配置。"""
    with _RUNTIME_LOCK:
        current = normalize_runtime_model(_RUNTIME_MODEL.get("provider"), _RUNTIME_MODEL.get("model_name"))
        _RUNTIME_MODEL.update(current)
        return dict(current)


def set_runtime_model(provider: str, model_name: str) -> dict[str, str]:
    """更新当前默认模型配置。"""
    with _RUNTIME_LOCK:
        current = normalize_runtime_model(provider, model_name)
        _RUNTIME_MODEL.update(current)
        return dict(current)


def list_runtime_options() -> list[dict[str, Any]]:
    """返回给前端展示的 provider 选项列表。"""
    current = get_runtime_model()
    options = []
    for provider, meta in SUPPORTED_PROVIDERS.items():
        options.append(
            {
                "provider": provider,
                "label": meta["label"],
                "default_model": meta["default_model"](),
                "env_keys": meta["env_keys"],
                "active": current["provider"] == provider,
            }
        )
    return options
