from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

import redis

from models import coerce_state


class SessionStore(ABC):
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, session_id: str, session: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[tuple[str, dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


def _serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": coerce_state(session["state"]).model_dump(mode="json"),
        "initial_state": coerce_state(session["initial_state"]).model_dump(mode="json"),
        "config": session["config"],
        "status": session["status"],
        "error_message": session.get("error_message"),
    }


def _deserialize_session(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": data["state"],
        "initial_state": data["initial_state"],
        "config": data["config"],
        "status": data["status"],
        "error_message": data.get("error_message"),
    }


class MemorySessionStore(SessionStore):
    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._data

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._data.get(session_id)
            return None if value is None else _deserialize_session(_serialize_session(value))

    def set(self, session_id: str, session: dict[str, Any]) -> None:
        with self._lock:
            self._data[session_id] = _serialize_session(session)

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return [(session_id, _deserialize_session(data)) for session_id, data in self._data.items()]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str, prefix: str = "ai-devops:sessions"):
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._index_key = f"{prefix}:index"

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def exists(self, session_id: str) -> bool:
        return bool(self._client.exists(self._key(session_id)))

    def get(self, session_id: str) -> dict[str, Any] | None:
        value = self._client.get(self._key(session_id))
        if value is None:
            return None
        return _deserialize_session(json.loads(value))

    def set(self, session_id: str, session: dict[str, Any]) -> None:
        payload = json.dumps(_serialize_session(session), ensure_ascii=False)
        pipe = self._client.pipeline()
        pipe.set(self._key(session_id), payload)
        pipe.sadd(self._index_key, session_id)
        pipe.execute()

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        session_ids = sorted(self._client.smembers(self._index_key))
        if not session_ids:
            return []
        pipe = self._client.pipeline()
        for session_id in session_ids:
            pipe.get(self._key(session_id))
        values = pipe.execute()
        items: list[tuple[str, dict[str, Any]]] = []
        for session_id, value in zip(session_ids, values):
            if value is not None:
                items.append((session_id, _deserialize_session(json.loads(value))))
        return items

    def clear(self) -> None:
        session_ids = self._client.smembers(self._index_key)
        pipe = self._client.pipeline()
        for session_id in session_ids:
            pipe.delete(self._key(session_id))
        pipe.delete(self._index_key)
        pipe.execute()


def build_session_store() -> SessionStore:
    backend = os.getenv("SESSION_STORE_BACKEND", "memory").lower()
    if backend == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSessionStore(redis_url)
    return MemorySessionStore()
