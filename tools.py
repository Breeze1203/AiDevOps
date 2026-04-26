from __future__ import annotations
"""Agent 可调用的工具集合。

当前实现大多是占位/模拟工具，目的是先把“LLM 决策 + tool-calling”链路跑通。
后续接真实平台时，可以只替换工具内部实现，不必改 agent 编排。
"""

import json
from typing import List

from langchain.tools import tool


@tool
def get_container_stats(container_name: str) -> str:
    """获取指定容器的实时状态。"""
    stats = {
        "container_name": container_name,
        "cpu_percent": 45.2,
        "memory_percent": 78.5,
        "restart_count": 3,
        "status": "running",
    }
    return json.dumps(stats, ensure_ascii=False)


@tool
def search_logs(container_name: str, keyword: str, lines: int = 100) -> str:
    """搜索容器日志中包含特定关键词的行。"""
    logs = [
        "2026-04-07 10:30:23 ERROR OutOfMemoryError: Java heap space",
        "2026-04-07 10:30:24 ERROR Unable to create new native thread",
        "2026-04-07 10:30:25 ERROR GC overhead limit exceeded",
        "2026-04-07 10:30:28 WARN Container memory usage reached 96.3%",
    ]
    matched = [line for line in logs if keyword.lower() in line.lower()]
    return "\n".join((matched or logs)[:lines])


@tool
def query_knowledge_base(error_type: str) -> str:
    """从知识库查询已知错误模式和解决方案。"""
    knowledge = {
        "oom_error": (
            "常见原因: JVM 堆内存过小、连接未关闭导致泄漏、突发流量导致对象激增。"
            "常见处理: 增加 -Xmx、分析 heap dump、检查连接池和大对象缓存。"
        ),
        "5xx_error": (
            "常见原因: 后端实例异常、数据库连接池耗尽、依赖超时。"
            "常见处理: 检查健康探针、扩容实例、调整超时和重试。"
        ),
        "network_error": (
            "常见原因: DNS 异常、服务发现失效、网络抖动。"
            "常见处理: 检查连通性、域名解析和负载均衡配置。"
        ),
    }
    return knowledge.get(error_type, "未找到直接匹配案例，建议结合日志与运行指标进一步分析。")


@tool
def web_search(query: str) -> str:
    """搜索互联网获取补充背景。当前为占位实现。"""
    return f"外部搜索占位结果: 已为查询 `{query}` 预留接入点，可后续替换为真实搜索 API。"


@tool
def restart_container(container_name: str, reason: str) -> str:
    """重启指定容器。当前为占位实现，不会真正执行。"""
    return f"模拟执行: 已请求重启容器 {container_name}，原因: {reason}"


@tool
def send_alert(severity: str, message: str, channels: List[str]) -> str:
    """发送告警通知。当前为占位实现。"""
    return f"模拟告警: severity={severity}, channels={', '.join(channels)}, message={message[:120]}"


TOOLS = [
    # 统一导出给 agent 进行 bind_tools。
    get_container_stats,
    search_logs,
    query_knowledge_base,
    web_search,
    restart_container,
    send_alert,
]
