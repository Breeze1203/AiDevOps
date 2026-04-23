# AI DevOps Diagnostic Agent

一个最小可运行的 PoC，用于消费 Kafka 运行日志事件，将事件自动转发到 AI 诊断 API，并通过 WebSocket 支持人工追问和最终干预。

## 当前能力

- 从 Kafka 主题 `ai-diagnostics` 消费诊断事件并自动转发到 API
- 基于日志和容器指标做错误分类、根因分析和修复建议
- 在执行修复前进入人工审批节点
- 支持 REST 和 WebSocket 两种人工交互方式
- 每个事件创建独立 `session_id` 和独立 LangGraph `thread_id`
- 不同事件并发执行，人工挂起某个会话不会阻塞其他会话
- 自动修复和告警当前是占位实现，便于后续替换真实 Docker/K8s/消息平台接口

## 目录

- `main.py`: Kafka 转发器，消费消息后转发给 API
- `api.py`: FastAPI 服务，提供开始诊断、人工追问、人工审批和 WebSocket 会话接口
- `agent.py`: LangGraph 工作流与 AI 问诊逻辑
- `models.py`: 事件、状态、接口响应模型
- `tools.py`: 日志、知识库、修复、告警工具封装
- `agent_test.py`: 本地集成测试

## 安装依赖

```bash
pip install -r requirements.txt
```

如果缺少 `uvicorn`，单独安装：

```bash
pip install uvicorn
```

## 环境变量模板

项目根目录提供了模板文件：

[/Users/pt/Download/AiDevOps/.env.example](/Users/pt/Download/AiDevOps/.env.example:1)

你可以按这个文件准备自己的环境变量，再导出到当前 shell。

## 环境变量

可选：

```bash
export GOOGLE_API_KEY=your_google_api_key
export GOOGLE_MODEL=gemini-2.5-flash-lite
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_TOPIC=ai-diagnostics
export KAFKA_GROUP_ID=ai-diagnostic-group
export KAFKA_AUTO_OFFSET_RESET=latest
export API_BASE_URL=http://127.0.0.1:8000
export API_FORWARD_MAX_RETRIES=3
export API_FORWARD_RETRY_DELAY_SECONDS=1
export FAILED_EVENT_QUEUE_PATH=failed_events.jsonl
export SESSION_STORE_BACKEND=redis
export REDIS_URL=redis://localhost:6379/0
```

说明：

- 配置 `GOOGLE_API_KEY` 后走真实 Gemini 模型
- 未配置时走本地规则兜底，方便你先联调 Kafka 和人工干预链路

## 一条命令启动

推荐直接使用：

```bash
python run.py
```

这会同时启动：

- FastAPI 服务
- Kafka 转发器

说明：

- `run.py` 只负责启动 Uvicorn
- Kafka 转发器会在 FastAPI `startup` 生命周期里自动拉起
- 这样可以避免“API 已启动，但 Kafka 线程自己的健康检查先超时退出”的启动竞态

`run.py` 启动前会做自检：

- 检查 `uvicorn`、`fastapi`、`kafka-python`、`redis` 依赖是否可用
- 检查 `KAFKA_BOOTSTRAP_SERVERS` 和 `KAFKA_TOPIC` 是否已配置
- 当 `SESSION_STORE_BACKEND=redis` 时，检查 `REDIS_URL` 是否可连接

默认地址：

- API 与前端：`http://127.0.0.1:8000`
- Dashboard：`http://127.0.0.1:8000/`

可选环境变量：

```bash
export API_HOST=127.0.0.1
export API_PORT=8000
```

## 分开启动 API

```bash
uvicorn api:api_app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
```

## 分开启动顺序

### 1. 启动 API

```bash
uvicorn api:api_app --reload
```

### 2. 启动 Kafka 转发器

```bash
python main.py
```

说明：

- Kafka 进程只负责消费和转发，不负责本地保存会话
- 诊断会话由 API 进程维护，session 数据可持久化到 Redis
- 每收到一条 Kafka 日志事件，转发器会调用 `POST /api/diagnostic/start`
- API 会立刻创建 session，并把首轮诊断放到后台线程池执行
- 同一个 session 内串行恢复执行，不同 session 之间并发
- Kafka 转发失败时会自动重试；超过阈值后写入本地失败队列文件

## Kafka 转发失败重试与失败队列

- `API_FORWARD_MAX_RETRIES`
  转发到 API 的最大重试次数，默认 `3`
- `API_FORWARD_RETRY_DELAY_SECONDS`
  基础重试间隔秒数，默认 `1`
- `FAILED_EVENT_QUEUE_PATH`
  本地失败队列文件路径，默认 `failed_events.jsonl`

行为说明：

- 如果 API 临时不可用，Kafka 转发器会自动重试
- 如果重试后仍失败，事件会被追加写入本地 `jsonl` 文件
- 每条失败记录包含：
  - `failed_at`
  - `error_message`
  - 原始 `event`

## 会话存储后端

- 默认 `SESSION_STORE_BACKEND=memory`
- 生产建议设置 `SESSION_STORE_BACKEND=redis`
- Redis 中会持久化：
  - session 状态
  - 当前诊断 state
  - initial state
  - LangGraph `thread_id` 配置

注意：

- 当前版本替换掉了 `active_sessions`，会话数据不再只存在内存字典里
- 但 LangGraph checkpoint 仍使用进程内 `MemorySaver`
- 这意味着：
  - API 运行期间，多 session 并发和人工挂起是正常的
  - Redis 能保留 session 数据和诊断结果
  - 如果 API 进程重启，LangGraph 的中断点 checkpoint 不会自动恢复
- 如果你要做到“进程重启后还能从人工决策点继续恢复”，下一步需要把 LangGraph checkpoint 也换成外部持久化存储

## 会话状态

- `queued`: 已创建会话，等待后台线程开始执行
- `running`: 正在执行首轮诊断或恢复后的 LangGraph 节点
- `awaiting_decision`: 已停在人工决策点，可追问、批准或拒绝
- `completed`: 已执行完成
- `failed`: 执行失败，可查看 `error_message`

## API 示例

### 1. 开始诊断

```bash
curl -X POST http://127.0.0.1:8000/api/diagnostic/start \
  -H "Content-Type: application/json" \
  -d '{
    "event": {
      "id": "evt-001",
      "timestamp": "2026-04-22T10:00:00",
      "log_event": {
        "id": "log-001",
        "timestamp": "2026-04-22T10:00:00",
        "source": "app",
        "level": "ERROR",
        "message": "java.lang.OutOfMemoryError: GC overhead limit exceeded",
        "error_type": "oom_error",
        "container_id": "c-1"
      },
      "container_stats": {
        "container_id": "c-1",
        "container_name": "payment-service",
        "cpu_percent": 82.4,
        "memory_usage": 1835008000,
        "memory_limit": 2147483648,
        "memory_percent": 95.7,
        "restart_count": 2,
        "status": "running"
      },
      "recent_logs": [
        "ERROR OutOfMemoryError: GC overhead limit exceeded",
        "WARN Full GC took 8.1s",
        "ERROR Request timeout for order submit"
      ]
    }
  }'
```

### 2. 人工追问

```bash
curl -X POST http://127.0.0.1:8000/api/conversation/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "evt-001",
    "question": "为什么会 OOM？"
  }'
```

### 3. 最终审批

```bash
curl -X POST http://127.0.0.1:8000/api/decision/submit \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "evt-001",
    "decision": "approve"
  }'
```

`decision` 支持：

- `approve`: 批准执行自动修复
- `reject`: 拒绝自动修复，仅保留报告和告警

## WebSocket 交互

问答追问模式用 WebSocket 更合适，原因很直接：

- 首轮诊断是异步到达的，WebSocket 方便把新会话实时推给值班台
- 人工追问和批准/拒绝是多轮交互，WebSocket 比轮询更自然
- 会话状态更新、执行日志和最终报告可以实时推送

当前项目提供两个通道：

- `ws://127.0.0.1:8000/ws/dashboard`
  用于值班面板接收所有会话的实时更新
- `ws://127.0.0.1:8000/ws/session/{session_id}`
  用于某个具体会话的追问和审批

### Dashboard WebSocket

连接后会先收到当前会话快照：

```json
{
  "type": "snapshot",
  "sessions": []
}
```

后续每当 Kafka 新事件进入或会话被更新，会收到：

```json
{
  "type": "session_update",
  "session_id": "evt-001",
  "summary": {
    "session_id": "evt-001",
    "status": "awaiting_decision",
    "root_cause": "...",
    "severity": "CRITICAL",
    "recommendations": ["..."],
    "auto_fix_action": "restart_container",
    "should_auto_fix": false,
    "should_alert": false
  },
  "messages": [],
  "execution_log": [],
  "report": null
}
```

### Session WebSocket

连接：

```text
ws://127.0.0.1:8000/ws/session/evt-001
```

可发送的消息：

```json
{"action": "get_state"}
{"action": "ask", "question": "为什么会 OOM？"}
{"action": "approve"}
{"action": "reject"}
```

你会先收到 `snapshot` 或 `session_update`，随后收到对应动作结果：

```json
{"type": "ask_result", "session_id": "evt-001", "answer": "...", "conversation_count": 1, "summary": {...}}
```

```json
{"type": "decision_result", "session_id": "evt-001", "decision": "approve", "summary": {...}, "execution_log": ["..."], "report": "..."}
```

## 测试

```bash
python agent_test.py
```

