from __future__ import annotations
"""项目启动入口。

这个模块只做两件事：
1. 从 `.env` 读取基础配置。
2. 在启动 FastAPI 之前做最小运行时自检。

这样做的目的是把“配置问题/依赖问题”尽量提前暴露，而不是等到
API 已经启动后，后台线程或请求路径里才报错。
"""

import importlib
import os


def load_dotenv_file(path: str = ".env.example") -> None:
    """从项目根目录读取简单的 KEY=VALUE 配置文件。"""
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
    """检查最基本的运行依赖是否已经安装。"""
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
    """检查 Kafka 连接所需的关键环境变量。"""
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").strip()
    topic = os.getenv("KAFKA_TOPIC", "ai-diagnostics").strip()
    if not bootstrap:
        raise RuntimeError("缺少 KAFKA_BOOTSTRAP_SERVERS 配置。")
    if not topic:
        raise RuntimeError("缺少 KAFKA_TOPIC 配置。")


def validate_redis_if_needed() -> None:
    """当会话存储切到 Redis 时，启动前先验证 Redis 是否可连。"""
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
    """汇总启动前需要做的自检步骤。"""
    print("执行启动前自检...")
    validate_runtime()
    validate_kafka_env()
    validate_redis_if_needed()
    print("启动前自检通过。")


def main():
    """加载配置并启动 FastAPI 服务。"""
    load_dotenv_file()
    preflight_check()
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    os.environ.setdefault("API_BASE_URL", f"http://{host}:{port}")
    uvicorn.run("api:api_app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
