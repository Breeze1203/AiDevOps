from __future__ import annotations

import importlib
import os


def load_dotenv_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def validate_runtime() -> None:
    missing_modules: list[str] = []
    for module_name in ("uvicorn", "fastapi", "kafka", "redis"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    if missing_modules:
        modules = ", ".join(missing_modules)
        raise RuntimeError(f"缺少运行依赖: {modules}。请先执行 `pip install -r requirements.txt`。")


def validate_kafka_env() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").strip()
    topic = os.getenv("KAFKA_TOPIC", "ai-diagnostics").strip()
    if not bootstrap:
        raise RuntimeError("缺少 KAFKA_BOOTSTRAP_SERVERS 配置。")
    if not topic:
        raise RuntimeError("缺少 KAFKA_TOPIC 配置。")


def validate_redis_if_needed() -> None:
    backend = os.getenv("SESSION_STORE_BACKEND", "memory").lower()
    if backend != "redis":
        return

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    import redis

    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:
        raise RuntimeError(f"Redis 自检失败，无法连接 {redis_url}: {exc}") from exc


def preflight_check() -> None:
    print("执行启动前自检...")
    validate_runtime()
    validate_kafka_env()
    validate_redis_if_needed()
    print("启动前自检通过。")


def main():
    load_dotenv_file()
    preflight_check()
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    os.environ.setdefault("API_BASE_URL", f"http://{host}:{port}")
    uvicorn.run("api:api_app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
