from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen
from typing import Iterable

from kafka import KafkaConsumer

from models import DiagnosticEvent


def build_kafka_consumer() -> KafkaConsumer:
    topic = os.getenv("KAFKA_TOPIC", "ai-diagnostics")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    group_id = os.getenv("KAFKA_GROUP_ID", "ai-diagnostic-group")
    offset = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset=offset,
    )


def forward_event_to_api(event_data: dict) -> dict:
    event = DiagnosticEvent.model_validate(event_data)
    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    url = f"{api_base_url.rstrip('/')}/api/diagnostic/start"
    payload = json.dumps({"event": event.model_dump(mode='json')}).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Explicitly bypass system HTTP proxies for local API forwarding.
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def retry_failed_queue_path() -> Path:
    path = os.getenv("FAILED_EVENT_QUEUE_PATH", "failed_events.jsonl")
    return Path(path)


def append_failed_event(event_data: dict, error_message: str) -> None:
    queue_path = retry_failed_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
        "event": event_data,
    }
    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def forward_event_with_retry(event_data: dict) -> dict:
    max_retries = int(os.getenv("API_FORWARD_MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("API_FORWARD_RETRY_DELAY_SECONDS", "1"))
    attempt = 0
    last_error: Exception | None = None
    while attempt <= max_retries:
        try:
            return forward_event_to_api(event_data)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_delay * (attempt + 1)
            print(f"转发失败，准备重试 {attempt + 1}/{max_retries}: {exc}")
            time.sleep(sleep_seconds)
            attempt += 1
        except Exception as exc:
            last_error = exc
            break

    message = str(last_error) if last_error is not None else "unknown forward error"
    append_failed_event(event_data, message)
    raise RuntimeError(
        f"事件转发失败，已写入本地失败队列: {retry_failed_queue_path()}，错误: {message}"
    ) from last_error


def consume_forever(messages: Iterable):
    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"Kafka forwarder started. Forwarding events to {api_base_url}")
    for message in messages:
        try:
            print("收到 Kafka 消息")
            event_data = message.value if hasattr(message, "value") else message
            print(f"准备转发事件: {event_data.get('id')}")
            result = forward_event_with_retry(event_data)
            print("=" * 60)
            print(f"已转发会话: {result['session_id']}")
            print(f"状态: {result['summary']['status']}")
            print("=" * 60)
        except HTTPError as exc:  # pragma: no cover
            print(f"API 返回错误: {exc.code} {exc.reason}")
        except URLError as exc:  # pragma: no cover
            print(f"无法连接 API: {exc.reason}")
        except Exception as exc:  # pragma: no cover
            print(f"处理事件失败: {exc}")


def main():
    consumer = build_kafka_consumer()
    consume_forever(consumer)


if __name__ == "__main__":
    main()
