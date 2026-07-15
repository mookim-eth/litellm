# Codex 对上游限流 / 并发限制的响应要求

> 交接文档：给改 LiteLLM（或其它 Codex 上游代理）的人用。
> 主要描述 **Codex 客户端源码里的行为与约束**，并明确 LiteLLM 实现边界。
> 对照仓库：`openai/codex` 的 `codex-rs`。

---

## 1. 背景与现象

当上游对 virtual key 做并发限制（例如 max parallel = 6），如果直接返回：

```http
HTTP/1.1 429 Too Many Requests
```

Codex UI / 日志里常见：

```text
exceeded retry limit, last status: 429 Too Many Requests
```

这句话**非常误导**：

- **不是**“已经自动重试多次后失败”
- **是**“收到 HTTP 429 后，被直接映射成 `RetryLimit` 错误并立刻结束”

结论：**Codex 默认不会对 HTTP 429 自动重试。**

### 1.1 LiteLLM 实现边界

本需求只改变 **Codex 客户端最终收到的 wire response**，不改变 LiteLLM
内部对并发限制的判断和记录：

- 并发限制在 LiteLLM 内部仍是 request failed / HTTP 429。
- 日志、SpendLogs、数据库、callback、指标和 UI 仍按现有 429 失败语义记录。
- 不修改数据库 schema，不新增数据库字段，不修改 UI。
- 只在确认是目标 Codex Responses 客户端后，于下行响应输出边界把该并发限制响应适配为
  `HTTP 200 + text/event-stream + response.failed`。
- 不得把内部异常或供日志、计费、callback 使用的状态码改成 200；200 只属于发给
  Codex 客户端的传输层兼容响应。
- 非 Codex 客户端以及不符合目标并发限制条件的 429，保持现有响应行为。

---

## 2. Codex 处理路径（两层）

### 2.1 传输层（HTTP 建连 / 开 SSE）

相关代码：

- `codex-rs/model-provider-info/src/lib.rs` → `to_api_provider()`
- `codex-rs/codex-client/src/retry.rs` → `RetryOn` / `run_with_retry`
- `codex-rs/codex-api/src/endpoint/session.rs` → `run_with_request_telemetry`

默认策略：

| 错误类型 | 是否重试 |
|---|---|
| HTTP **429** | **否**（`retry_429: false`，写死） |
| HTTP 5xx | 是 |
| 网络错误 / 超时 | 是 |

默认 `request_max_retries = 4`（用户可配，上限 100）。
**`Retry-After` 响应头在 model/responses 路径不被读取。**

### 2.2 Session 层（流失败后整轮重试）

相关代码：

- `codex-rs/protocol/src/error.rs` → `CodexErr::is_retryable()`
- `codex-rs/core/src/session/turn.rs` → sampling 重试循环
- `codex-rs/core/src/responses_retry.rs` → `handle_retryable_response_stream_error`
- `codex-rs/core/src/util.rs` → `backoff()`

默认 `stream_max_retries = 5`（用户可配，上限 100）。

只有 `is_retryable() == true` 的错误才会进入 session 重试，并显示类似：

```text
Reconnecting... n/m
```

---

## 3. HTTP 429 如何被映射

相关代码：`codex-rs/codex-api/src/api_bridge.rs` → `map_api_error()`

| 上游返回 | Codex 错误 | 是否自动重试 |
|---|---|---|
| HTTP 429 + body `error.type = "usage_limit_reached"` | `UsageLimitReached` | **否**（更新 rate limit UI 后失败） |
| HTTP 429 + body `error.type = "usage_not_included"` | `UsageNotIncluded` | **否** |
| HTTP 429 + 其它 body / 空 body / 普通限流文案 | `RetryLimit`（文案即 *exceeded retry limit...*） | **否** |
| HTTP 503 + body code `server_is_overloaded` / `slow_down` | `ServerOverloaded` | **否** |
| HTTP 500 | `InternalServerError` | 会重试，但 delay 是短指数退避，**不可指定 30s** |

### 关于 `usage_limit_reached` 的 body 形状

Codex 只在 **HTTP 429** 时尝试解析：

```json
{
  "error": {
    "type": "usage_limit_reached",
    "plan_type": "pro",
    "resets_at": 1738888888
  }
}
```

注意字段名是 **`error.type`**（不是 `error.code`）。
这会走“账号用量上限”产品逻辑，**不会**按“等一会儿再试”处理。

并发限制 **不要** 伪装成 `usage_limit_reached`。

---

## 4. 唯一可靠的“等 N 秒再重试”路径

想让 Codex **等待指定时间后自动重试**，必须让错误最终变成：

```text
CodexErr::Stream(message, Some(delay))
```

来源：

```text
SSE response.failed
  → ApiError::Retryable { message, delay }
  → CodexErr::Stream(message, delay)
  → is_retryable() == true
  → sleep(delay) 后重试
```

相关代码：

- `codex-rs/codex-api/src/sse/responses.rs`（`response.failed` 解析 + `try_parse_retry_after`）
- `codex-rs/codex-api/src/api_bridge.rs`（`ApiError::Retryable` → `CodexErr::Stream`）
- `codex-rs/core/src/responses_retry.rs`（优先使用 `requested_delay`）

### 4.1 正确响应形态（关键）

1. **HTTP 状态码必须是 2xx**（通常 `200`），成功打开 SSE
2. `Content-Type: text/event-stream`
3. 推送 `response.failed` 事件
4. error **`code`** 必须是 `rate_limit_exceeded`
5. **`message` 必须匹配 Codex 的 delay 正则**（见下一节）

示例：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: response.failed
data: {"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded","message":"Concurrency limit reached. Please try again in 30s."}}}

```

最小可用 JSON：

```json
{
  "type": "response.failed",
  "response": {
    "error": {
      "code": "rate_limit_exceeded",
      "message": "Please try again in 30s."
    }
  }
}
```

### 4.2 为什么不能直接 HTTP 429 + 同样 body？

即使 body 写成 `rate_limit_exceeded`，只要 HTTP 是 **429**，就会先在 transport 层变成 `TransportError::Http { status: 429 }`，再被 `map_api_error` 打成 **不可重试** 的 `RetryLimit` / `UsageLimitReached`。
**`try_parse_retry_after` 只跑在 SSE `response.failed` 解析路径上，不会处理裸 429。**

---

## 5. message 约束（最重要）

### 5.1 解析条件

`try_parse_retry_after()` 要求：

1. `error.code == "rate_limit_exceeded"`（字符串精确匹配）
2. `error.message` 匹配正则：

```regex
(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)
```

源码：`codex-rs/codex-api/src/sse/responses.rs` 中 `rate_limit_regex()`。

### 5.2 正则含义

- 大小写不敏感
- 必须出现子串：`try again in`
- 后接空白 + 数字（可小数）+ 单位
- 单位只支持：
  - `s`
  - `ms`
  - `second` / `seconds`

### 5.3 推荐写法

| 目标等待 | 推荐 message 片段 | 结果 |
|---|---|---|
| 30 秒 | `Please try again in 30s.` | `Duration::from_secs_f64(30.0)` |
| 30 秒 | `try again in 30 seconds.` | 同上 |
| 1.5 秒 | `Please try again in 1.5s.` | 1.5s |
| 28 毫秒 | `Please try again in 28ms.` | 28ms |

完整可用示例：

```text
Concurrency limit reached for this key. Please try again in 30s.
```

```text
Rate limit exceeded. Try again in 30 seconds.
```

### 5.4 不可用写法（不会解析出 delay）

| message | 问题 |
|---|---|
| `try again in 1 minute` | 不支持 `minute` / `minutes` |
| `try again in 1 min` | 不支持 |
| `retry after 30 seconds` | 必须是 `try again in`，不是 `retry after` |
| `please retry in 30s` | 必须是 `try again in` |
| `wait 30s` | 不匹配 |
| `Retry-After: 30`（仅 header） | model 路径不读 header |
| 只有 code 没有可匹配 message | `delay = None`，退回默认短 backoff |

### 5.5 `delay = None` 时会发生什么？

即使 `code = rate_limit_exceeded`，只要 message 解析失败：

- 仍会变成可重试的 `CodexErr::Stream`
- 但等待时间用默认指数退避，而不是 30s：

```text
base ≈ 200ms * 2^(attempt-1)  （带 0.9~1.1 jitter）
```

大约：200ms → 400ms → 800ms → 1.6s → 3.2s …
**达不到稳定 30 秒。**

所以：**要固定等 30 秒，message 必须可被正则解析出 30s。**

---

## 6. 明确不要做的事

| 做法 | Codex 结果 |
|---|---|
| 直接 HTTP `429` | `RetryLimit` / 用量错误，**不重试** |
| HTTP `429` + `Retry-After: 30` | 忽略 header，仍不重试 429 |
| HTTP `429` + `usage_limit_reached` | 产品向“额度用尽”，不按并发等待重试 |
| HTTP `503` + `server_is_overloaded` / `slow_down` | `ServerOverloaded`，不重试 |
| HTTP `500` 想“偷偷触发重试” | 会重试，但 delay 不可控，且语义错误 |
| SSE 失败但 `code` 不是 `rate_limit_exceeded` | 可能仍可重试，但**没有指定 delay** |
| message 写 `1 minute` | 解析失败，退回短 backoff |

---

## 7. 重试次数与 Codex 侧配置

Session 层默认最多 `stream_max_retries = 5` 次重连。

若每次等 30s，粗算最多约 5 次 × 30s。
如果并发高峰更长，需要在 Codex provider 配置加大：

```toml
# config.toml 示例（字段名以实际 model_providers 配置为准）
stream_max_retries = 20
```

说明：

- `stream_max_retries`：流失败后的 session 重试（本方案依赖这个）
- `request_max_retries`：HTTP 传输层重试；对 429 **无效**（`retry_429=false`）

---

## 8. 验收清单

### 错误做法（应继续看到失败文案）

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{"error":{"message":"Max parallel requests reached","type":"rate_limit_error"}}
```

期望 Codex：

```text
exceeded retry limit, last status: 429 Too Many Requests
```

且上游请求次数 ≈ 1（无 429 重试）。

### 正确做法（应自动等待约 30s 后重试）

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

event: response.failed
data: {"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded","message":"Concurrency limit reached. Please try again in 30s."}}}
```

期望 Codex：

1. 不立刻以 `exceeded retry limit, last status: 429` 失败
2. 日志 / UI 出现 `Reconnecting... 1/5`（或类似）
3. 约 30 秒后再次请求上游
4. 若仍失败，继续按 `stream_max_retries` 重试

### 快速 curl 自测（代理层）

确认代理在并发打满时返回 **200 SSE**，而不是 429：

```bash
# 伪示例：观察 status 与 body 是否为 event-stream + response.failed
curl -N -i "$BASE/v1/responses" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

检查点：

- [ ] status = 200
- [ ] body 含 `response.failed`
- [ ] `error.code` = `rate_limit_exceeded`
- [ ] `error.message` 含 `try again in 30s` 或 `try again in 30 seconds`
- [ ] **没有** 直接 HTTP 429

---

## 9. 决策树（给实现者）

```text
并发超限时要让 Codex 自动等 30s 再试？
│
├─ 返回 HTTP 429 ─────────────────────────── ✗ 不会重试
│     （无论 body / Retry-After 如何）
│
├─ 返回 HTTP 500/502 ─────────────────────── △ 会重试，但 delay 不可指定 30s
│
├─ 返回 200 SSE + response.failed
│     code != rate_limit_exceeded ────────── △ 可能重试，但 delay 默认短退避
│
└─ 返回 200 SSE + response.failed
      code == rate_limit_exceeded
      message 匹配 try again in 30s ──────── ✓ 固定等约 30s 后重试
```

---

## 10. 关键源码索引

| 主题 | 路径 |
|---|---|
| 429 映射 | `codex-rs/codex-api/src/api_bridge.rs` |
| 传输重试策略（含 `retry_429: false`） | `codex-rs/model-provider-info/src/lib.rs` |
| 重试执行器 | `codex-rs/codex-client/src/retry.rs` |
| SSE `response.failed` / delay 解析 | `codex-rs/codex-api/src/sse/responses.rs` |
| 是否可重试 | `codex-rs/protocol/src/error.rs` → `CodexErr::is_retryable` |
| Session 重试循环 | `codex-rs/core/src/session/turn.rs` |
| 重试 sleep / 用户提示 | `codex-rs/core/src/responses_retry.rs` |
| 默认退避 | `codex-rs/core/src/util.rs` → `backoff` |
| 默认重试次数 | `codex-rs/model-provider-info/src/lib.rs`（`DEFAULT_STREAM_MAX_RETRIES=5`, `DEFAULT_REQUEST_MAX_RETRIES=4`） |

---

## 11. 一句话交付要求

**只对目标 Codex 客户端，不要在 wire response 中返回 HTTP 429。**
给 Codex 回 **HTTP 200 + SSE `response.failed`**，且：

```text
error.code    = "rate_limit_exceeded"
error.message 包含: "try again in 30s"   # 或 "try again in 30 seconds"
```

这样 Codex 才会按约 **30 秒** 自动重试；否则会立刻报：

```text
exceeded retry limit, last status: 429 Too Many Requests
```
