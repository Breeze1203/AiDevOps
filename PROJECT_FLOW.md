# AI DevOps 项目时序图与节点流转图

## 1. 整体时序图

```mermaid
sequenceDiagram
    autonumber
    participant K as Kafka
    participant F as main.py 转发器
    participant A as FastAPI(api.py)
    participant G as LangGraph(agent.py)
    participant S as active_sessions
    participant W as WebSocket
    participant U as 值班前端/人工

    K->>F: 推送日志/故障事件
    F->>A: POST /api/diagnostic/start
    A->>G: app.invoke(initial_state, config)

    G->>G: classify
    alt severity 为 CRITICAL/HIGH
        G->>G: gather_context
    end
    G->>G: analyze
    G->>G: generate
    G-->>A: 在 human_decision 前中断返回

    A->>S: 保存 session state + config
    A->>W: 广播 session_update
    W->>U: 推送新会话与首轮诊断结果

    opt 人工追问
        U->>W: {"action":"ask","question":"..."}
        W->>A: session websocket message
        A->>G: update_state(human_question, human_decision=continue)
        A->>G: app.invoke(None, config)
        G->>G: conversation
        G-->>A: 再次在 human_decision 前中断
        A->>S: 更新 session state
        A->>W: 广播 session_update + ask_result
        W->>U: 返回 AI 回答
    end

    opt 人工批准
        U->>W: {"action":"approve"}
        W->>A: session websocket message
        A->>G: update_state(human_decision=approve)
        A->>G: app.invoke(None, config)
        G->>G: decide
        G->>G: auto_fix
        G->>G: alert
        G-->>A: 执行完成
        A->>S: 更新 session state
        A->>W: 广播 session_update + decision_result
        W->>U: 返回最终报告与执行日志
    end

    opt 人工拒绝
        U->>W: {"action":"reject"}
        W->>A: session websocket message
        A->>G: update_state(human_decision=reject)
        A->>G: app.invoke(None, config)
        G->>G: decide
        G->>G: alert
        G-->>A: 执行完成
        A->>S: 更新 session state
        A->>W: 广播 session_update + decision_result
        W->>U: 返回最终报告
    end
```

## 2. 工作流节点流转图

```mermaid
flowchart TD
    A["Kafka 事件进入<br/>POST /api/diagnostic/start"] --> B["classify<br/>错误分类与严重度判断"]
    B --> C{"severity 是否为<br/>CRITICAL / HIGH"}
    C -- 是 --> D["gather_context<br/>查日志/知识库"]
    C -- 否 --> E["analyze<br/>根因分析"]
    D --> E
    E --> F["generate<br/>生成修复建议"]
    F --> G["human_decision<br/>人工决策前中断"]

    G --> H{"human_decision"}
    H -- continue --> I["conversation<br/>AI 回答追问"]
    I --> G
    H -- approve --> J["decide<br/>确定执行策略"]
    H -- reject --> J
    H -- 未给定 --> G

    J --> K{"should_auto_fix"}
    K -- true --> L["auto_fix<br/>执行自动修复"]
    K -- false --> M["alert<br/>发送告警"]
    L --> M
    M --> N["END"]
```

## 3. 运行逻辑说明

- Kafka 不直接连 WebSocket。
- Kafka 只把事件交给 `main.py` 转发器。
- 转发器只负责 HTTP 调用 API。
- 真正执行 Agent 节点的是 `api.py` 对 `agent.py` 中 LangGraph 的调用。
- WebSocket 的职责是把 API 中的会话状态实时推送给前端，以及接收人工追问/审批指令。

## 4. 两条核心路径

### 4.1 首轮自动诊断

```text
Kafka -> main.py -> /api/diagnostic/start -> classify -> gather_context/analyze -> generate -> 中断等待人工
```

### 4.2 人工追问或审批

```text
前端 WebSocket -> api.py update_state -> LangGraph 从 human_decision 断点恢复 -> conversation 或 decide -> 后续节点
```

## 5. 文件对应关系

- `main.py`
  Kafka 消费与 HTTP 转发
- `api.py`
  REST/WebSocket 接口、会话状态管理、恢复工作流执行
- `agent.py`
  LangGraph 节点定义与节点路由
- `frontend/index.html`
  值班台页面，左侧会话列表，右侧实时问答
