# LiteLLM Codex → Gemini 修复交接

- 交接时间：2026-09-14 17:12 CST
- 工作目录：`/root/litellm/code`
- 分支：`restore-safe-fixes-ba3b178`
- 基线 HEAD：`1cea26950822c0d9d022238074ea57ac55891155`
- 实现提交：`a0ad9431df84ad1940eb07087952065160224978`
- 状态：代码修复、单元回归、独立测试环境、真实 Gemini E2E 和生产蓝绿切换均已完成
- 生产状态：**实现已提交并推送；未执行 Docker build；已通过同镜像新容器 + Docker CP 切换到 green**

## 结论

本次修复解决了 Codex 通过 LiteLLM Responses API 调用 Gemini 时的四条本地缺陷：裸错误对象、Gemini 历史以 model turn 结束、工具返回图片的错误层级，以及 Codex 的 `max`/`xhigh` reasoning effort。`/v1/models` 现在对 Codex 增加 `models` envelope，同时保留标准 OpenAI `data`/`object` envelope；可访问的 `gemini-*` 模型不再被公共模型过滤器移除。

生产库排查快照中，对应失败族为：45 条 `max/xhigh` 映射失败、31 条“请求以 model turn 结束”、6 条 Gemini 2.5 混合工具限制。请求正文仅用于本地定位，没有复制到交接文档或测试库。

## 实现摘要

1. Responses 错误事件

   - 新增统一的 Responses 错误事件构造器，输出顶层 `type/code/message/param/sequence_number`。
   - SSE 使用 `event: error`；终止错误后不再附加 `[DONE]`。
   - Responses 路由使用服务端选择的专用 stream serializer，不信任客户端请求体标记。
   - Codex 的流式预调用错误以 HTTP 200 + 合法 SSE error event 返回，真实映射后错误信息保留。
   - 原有可重试 429/TTFT/overload 继续使用 `response.failed`，并补齐 `sequence_number`。
   - 非 Codex 的预调用错误和非 Responses 的通用流保持原兼容行为。

2. Gemini 消息与工具结果

   - 如果转换后的 Gemini contents 最后是 `role=model`，追加一个中性的空白 user continuation turn。
   - 覆盖尾部 assistant、function call，以及尾部 developer 被提取成 system instruction 后遗留 model turn 的情况。
   - 图片/PDF 等多模态工具结果从 sibling `inline_data` 改为 `function_response.parts[].inline_data`，符合 Gemini multimodal function response 结构。

3. 推理强度与模型发现

   - Gemini thinking budget 和 thinking level 两条映射都将 `max`、`xhigh` 归一到 provider 的 `high`。
   - `/v1/models` 对 Codex User-Agent 返回 `models` 别名，同时继续返回标准 `data` 与 `object`。
   - 公共模型过滤器允许已经过鉴权筛选的 `gemini-*` 模型；没有绕过 key/team 的模型授权。

参考规范：[OpenAI Responses streaming events](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses)、[Google Vertex AI function calling with multimodal responses](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)。

## 验证结果

单元/回归测试：

- 合并相关回归：`202 passed, 1 skipped, 52 deselected`。
- 最终 Responses 错误协议复跑：`34 passed`。
- 通用非 Responses stream 兼容复跑：`2 passed`。
- Gemini 转换文件：`56 passed, 1 skipped`；Gemini 参数映射文件：`95 passed`。
- 生产交付前容器化复跑：provider/转换 `213 passed, 1 skipped`；代理协议 `42 passed, 186 deselected`。
- `git diff --check`：通过。
- 新错误序列化模块 Ruff：通过。
- `docker compose -f docker-compose.test.yml config --quiet`：通过。

真实 E2E（全部走 `http://127.0.0.1:4001/v1/responses`，Codex User-Agent，使用测试数据库）：

| 场景 | 模型 | 结果 |
|---|---|---|
| text + reasoning `max` | `gemini-3.5-flash` | `response.completed` |
| text + reasoning `xhigh` | `gemini-3.5-flash` | `response.completed` |
| thinking-budget `xhigh` | `gemini-2.5-flash` | `response.completed` |
| 尾部 assistant | `gemini-3.5-flash` | `response.completed` |
| 尾部 developer | `gemini-3.5-flash` | `response.completed` |
| 尾部 function_call | `gemini-3.5-flash` | `response.completed` |
| 单工具调用 | `gemini-3.5-flash` | 返回 1 个 function_call 并 completed |
| 并行工具调用 | `gemini-3.5-flash` | 返回 2 个 function_call 并 completed |
| 图片 function_call_output | `gemini-3.5-flash` | reasoning + text 输出并 completed |
| 不存在模型（Codex） | 本地路由错误 | HTTP 200、`event: error`、仅 5 个合法顶层字段、无 `[DONE]` |
| 不存在模型（OpenAI SDK UA） | 本地路由错误 | 保持 HTTP 400 JSON |
| function + web_search 混合工具 | `gemini-2.5-flash` | HTTP 200 合法 `event: error`，真实 Vertex 400 信息可见 |

## 独立测试环境

- Compose：`docker-compose.test.yml`
- 代理：`litellm-litellm-test-1`，绑定 `127.0.0.1:4001`，healthy
- 测试镜像：`litellm:codex-gemini-test`
- 测试镜像 ID：`sha256:e18eaa014bb5b6ce703a2fdd61d02c1e1cf0d377c24b47699216449627fa9ddf`
- 数据库：`litellm-codex-gemini-test-db`，独立 named volume，healthy
- 初始数据：仅 schema、4 条 Config、6 条加密 Credentials、18 条 ProxyModel、118 条 Prisma migration；User、VerificationToken、SpendLogs 初始均为 0。
- E2E 后测试 SpendLogs：10 条预期成功（9 条 Gemini 3.5、1 条 Gemini 2.5）；4 条故意失败用于验证错误协议。

测试环境保留运行，便于继续验证。停止但保留数据库卷：

```bash
cd /root/litellm/code
docker compose -f docker-compose.test.yml stop
```

重新启动：

```bash
cd /root/litellm/code
docker compose -f docker-compose.test.yml up -d
curl --fail http://127.0.0.1:4001/health/readiness
```

## 已知边界

Gemini 2.5 的 provider 原生限制仍不允许普通 function declarations 与 web search 同次混用；本次没有静默删除任何工具。该请求现在会把 Vertex 的真实 `Multiple tools are supported only when they are all search tools` 错误作为合法 Responses `error` 事件返回。这是可观察性修复，不是对上游能力限制的规避。

## 生产部署结果

- 实现提交 `a0ad9431df84ad1940eb07087952065160224978` 已推送到 `origin/restore-safe-fixes-ba3b178`。
- 未执行 Docker build。green 使用与原 blue 完全相同的基础镜像 `mookim/litellm:9511eb7165df`，镜像 ID 为 `sha256:97ee9251f575ce5d4ccfc338cf525f6d079679b19e2560078a63e829f3c2ecd8`。
- 先确认 6 个被修改文件在镜像内与补丁父提交逐一哈希匹配，再向 inactive green 的真实 CLI import 路径 `/usr/lib/python3.13/site-packages/litellm` 复制 7 个运行时文件；复制后所有 SHA256 均与实现提交一致。
- green 在 `127.0.0.1:4003` 独立启动、重启并达到 healthy 后，完成了切流前真实 E2E：Codex 模型发现、Gemini 3.5 text/xhigh、assistant 尾部、function_call 尾部、单工具、并行工具、图片 function output、Gemini 2.5 xhigh 和合法错误事件均通过；标准 OpenAI UA 兼容行为保持不变。
- `deploy-litellm.sh switch` 于 2026-09-14 17:08 CST 将 Nginx backend 从 4002（blue）切至 4003（green）。外部与内部 readiness 均通过；旧 blue 的 1 条连接排空并保持 6 秒静默后由脚本停止。
- 切流后通过生产 HTTPS 路由复验：Codex `models` envelope、Gemini 3.5 text/xhigh、单工具、2 个并行工具及合法 `event:error` 全部通过。green 为 healthy、数据库 connected、无 OOM/意外重启，最近 5 次容器健康检查退出码均为 0。
- 日志中故意触发的非法模型请求产生预期 `BadRequestError`；部分 Gemini 部署尝试产生上游 `AuthenticationError`，Router fallback 后测试请求均成功完成，未形成终端失败。
- Compose 备份位于 `/root/litellm/backups/codex-gemini-a0ad9431df/docker-compose.yml`，Nginx 切换备份位于 `/root/litellm/backups/20260914090812/`；部署锁已释放。

重要持久性说明：green 显示的镜像标签和 ID 仍是未包含补丁的原基础镜像，补丁存在于该容器的 writable layer。普通 `docker restart` 会保留补丁，但任何 Compose recreate、容器替换或从该镜像重新启动都会丢失补丁。回滚可将 Nginx 切回 4002 并重新启动未修改的 blue；长期交付仍应将提交构建为不可变镜像。

## 完整实现 Patch

以下 patch 是相对基线 HEAD 的完整实现差异，包含所有源码、测试、测试 Compose、测试 Dockerfile 和新错误事件模块。为避免文档自引用造成无限 patch，本节唯一排除的文件是本 handover Markdown 自身。

注意：代码块中仅含单个空格的行是 unified diff 对空白 context line 的标准标记。它们为保证 patch 可应用而保留，因此把本 Markdown 作为全新文件执行 `git diff --check` 会报告这些行；实现提交自身的 `git diff --check` 已通过。

```diff
diff --git a/litellm/litellm_core_utils/prompt_templates/factory.py b/litellm/litellm_core_utils/prompt_templates/factory.py
index b37c17fafe..8b4ac937d0 100644
--- a/litellm/litellm_core_utils/prompt_templates/factory.py
+++ b/litellm/litellm_core_utils/prompt_templates/factory.py
@@ -1468,7 +1468,7 @@ def convert_to_gemini_tool_call_invoke(
 def convert_to_gemini_tool_call_result(  # noqa: PLR0915
     message: Union[ChatCompletionToolMessage, ChatCompletionFunctionMessage],
     last_message_with_tool_calls: Optional[dict],
-) -> Union[VertexPartType, List[VertexPartType]]:
+) -> VertexPartType:
     """
     OpenAI message with a tool result looks like:
     {
@@ -1640,17 +1640,16 @@ def convert_to_gemini_tool_call_result(  # noqa: PLR0915
         name=name, response=response_data  # type: ignore
     )
 
-    # Create part with function_response, and optionally inline_data for images (Computer Use)
-    _part: VertexPartType = {"function_response": _function_response}
-
-    # For Computer Use, if we have images/files, we need separate parts:
-    # - One part with function_response
-    # - One part per inline_data item
-    # Gemini's PartType is a oneof, so we can't have both in the same part
+    # Gemini requires media returned by a function to live under
+    # functionResponse.parts. Sending inline_data as sibling Content parts makes
+    # the request look like a normal user image turn and is rejected by newer
+    # Gemini generateContent endpoints.
     if inline_data_list:
-        return [_part] + [{"inline_data": d} for d in inline_data_list]
+        _function_response["parts"] = [
+            {"inline_data": inline_data} for inline_data in inline_data_list
+        ]
 
-    return _part
+    return {"function_response": _function_response}
 
 
 def _sanitize_anthropic_tool_use_id(tool_use_id: str) -> str:
diff --git a/litellm/llms/vertex_ai/gemini/transformation.py b/litellm/llms/vertex_ai/gemini/transformation.py
index 6157a384dc..b5b86b1574 100644
--- a/litellm/llms/vertex_ai/gemini/transformation.py
+++ b/litellm/llms/vertex_ai/gemini/transformation.py
@@ -611,6 +611,13 @@ def _gemini_convert_messages_with_history(  # noqa: PLR0915
         if len(tool_call_responses) > 0:
             contents.append(ContentType(role="user", parts=tool_call_responses))
 
+        if len(contents) > 0 and contents[-1].get("role") == "model":
+            # Gemini generateContent cannot continue from a history whose last
+            # content is a model turn. This occurs in Responses conversations
+            # that end in an assistant message/function call, and after a
+            # trailing developer message is lifted into system instructions.
+            contents.append(ContentType(role="user", parts=[PartType(text=" ")]))
+
         if len(contents) == 0:
             verbose_logger.warning(
                 """
diff --git a/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py b/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py
index e6e548ab98..6b9e7cbf06 100644
--- a/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py
+++ b/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py
@@ -781,7 +781,7 @@ class VertexGeminiConfig(VertexAIBaseConfig, BaseConfig):
                 "thinkingBudget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                 "includeThoughts": True,
             }
-        elif reasoning_effort == "high":
+        elif reasoning_effort in ("high", "max", "xhigh"):
             return {
                 "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
                 "includeThoughts": True,
@@ -831,7 +831,7 @@ class VertexGeminiConfig(VertexAIBaseConfig, BaseConfig):
                 return {"thinkingLevel": "medium", "includeThoughts": True}
             else:
                 return {"thinkingLevel": "high", "includeThoughts": True}
-        elif reasoning_effort == "high":
+        elif reasoning_effort in ("high", "max", "xhigh"):
             return {"thinkingLevel": "high", "includeThoughts": True}
         elif reasoning_effort == "disable":
             # Gemini 3 cannot fully disable thinking, so we use "minimal" for gemini-3-flash-preview, "low" for others
diff --git a/litellm/proxy/proxy_server.py b/litellm/proxy/proxy_server.py
index 52d77aa4bc..2ff213153a 100644
--- a/litellm/proxy/proxy_server.py
+++ b/litellm/proxy/proxy_server.py
@@ -6447,7 +6447,10 @@ async def _coalesce_plain_text_stream(response: Any):  # noqa: PLR0915
 
 
 async def async_data_generator(  # noqa: PLR0915
-    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
+    response,
+    user_api_key_dict: UserAPIKeyAuth,
+    request_data: dict,
+    is_responses_api: bool = False,
 ):
     verbose_proxy_logger.debug("inside generator")
     try:
@@ -6457,6 +6460,7 @@ async def async_data_generator(  # noqa: PLR0915
         )
         zai_responses_sequence_number = 0
         zai_responses_seen_events: set[tuple[str, str]] = set()
+        responses_error_sequence_number = 0
         model_mismatch_logged = False
         first_proxy_yield_recorded = False
         # Use a running string instead of list + join to avoid O(n^2) overhead.
@@ -6571,9 +6575,23 @@ async def async_data_generator(  # noqa: PLR0915
 
                     setattr(chunk, "sequence_number", zai_responses_sequence_number)
                     zai_responses_sequence_number += 1
+                chunk_sequence_number = getattr(chunk, "sequence_number", None)
+                if isinstance(chunk_sequence_number, int):
+                    responses_error_sequence_number = max(
+                        responses_error_sequence_number, chunk_sequence_number + 1
+                    )
                 chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
             elif isinstance(chunk, str) and chunk.startswith("data: "):
-                error_message = chunk
+                if is_responses_api:
+                    from litellm.proxy.response_api_endpoints.error_responses import (
+                        serialize_responses_error_event,
+                    )
+
+                    error_message = serialize_responses_error_event(
+                        chunk, sequence_number=responses_error_sequence_number
+                    )
+                else:
+                    error_message = chunk
                 break
 
             try:
@@ -6591,6 +6609,8 @@ async def async_data_generator(  # noqa: PLR0915
         # Streaming is done, yield the [DONE] chunk
         if error_message is not None:
             yield error_message
+            if is_responses_api:
+                return
         done_message = "[DONE]"
         yield f"data: {done_message}\n\n"
     except Exception as e:
@@ -6640,6 +6660,7 @@ async def async_data_generator(  # noqa: PLR0915
             # Responses event shape that Codex recognizes.
             retryable_error = {
                 "type": "response.failed",
+                "sequence_number": responses_error_sequence_number,
                 "response": {
                     "error": {
                         "code": "rate_limit_exceeded",
@@ -6649,7 +6670,8 @@ async def async_data_generator(  # noqa: PLR0915
                     }
                 },
             }
-            yield f"data: {json.dumps(retryable_error)}\n\n"
+            event_prefix = "event: response.failed\n" if is_responses_api else ""
+            yield f"{event_prefix}data: {json.dumps(retryable_error)}\n\n"
             return
         if getattr(e, "is_responses_stream_overload", False):
             # Codex treats server_is_overloaded/slow_down as terminal errors. Map
@@ -6657,6 +6679,7 @@ async def async_data_generator(  # noqa: PLR0915
             # shape, including the exact delay phrase parsed by the client.
             retryable_error = {
                 "type": "response.failed",
+                "sequence_number": responses_error_sequence_number,
                 "response": {
                     "error": {
                         "code": "rate_limit_exceeded",
@@ -6667,10 +6690,13 @@ async def async_data_generator(  # noqa: PLR0915
                     }
                 },
             }
-            yield f"data: {json.dumps(retryable_error)}\n\n"
+            event_prefix = "event: response.failed\n" if is_responses_api else ""
+            yield f"{event_prefix}data: {json.dumps(retryable_error)}\n\n"
             return
-        if isinstance(e, HTTPException):
+        if isinstance(e, HTTPException) and not is_responses_api:
             raise e
+        if isinstance(e, HTTPException):
+            error_msg = str(e.detail)
         elif isinstance(e, StreamingCallbackError):
             error_msg = str(e)
         else:
@@ -6686,8 +6712,18 @@ async def async_data_generator(  # noqa: PLR0915
             param=getattr(e, "param", "None"),
             code=getattr(e, "status_code", 500),
         )
-        error_returned = json.dumps({"error": proxy_exception.to_dict()})
-        yield f"data: {error_returned}\n\n"
+        if is_responses_api:
+            from litellm.proxy.response_api_endpoints.error_responses import (
+                serialize_responses_error_event,
+            )
+
+            yield serialize_responses_error_event(
+                proxy_exception,
+                sequence_number=responses_error_sequence_number,
+            )
+        else:
+            error_returned = json.dumps({"error": proxy_exception.to_dict()})
+            yield f"data: {error_returned}\n\n"
     finally:
         # Close the response stream to release the underlying HTTP connection
         # back to the connection pool. This prevents pool exhaustion when
@@ -6725,6 +6761,18 @@ def select_data_generator(
     )
 
 
+def select_responses_data_generator(
+    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
+):
+    """Select the stream serializer with the Responses SSE contract enabled."""
+    return async_data_generator(
+        response=response,
+        user_api_key_dict=user_api_key_dict,
+        request_data=request_data,
+        is_responses_api=True,
+    )
+
+
 def get_litellm_model_info(model: dict = {}):
     model_info = model.get("model_info", {})
     model_to_lookup = model.get("litellm_params", {}).get("model", None)
@@ -7632,10 +7680,22 @@ _PUBLIC_V1_MODEL_NAMES = frozenset(
 
 def _is_public_v1_model(model_name: str) -> bool:
     return model_name in _PUBLIC_V1_MODEL_NAMES or model_name.startswith(
-        ("glm-", "grok-")
+        ("gemini-", "glm-", "grok-")
     )
 
 
+def _model_list_response(model_data: List[dict], request: Request) -> Dict[str, Any]:
+    response: Dict[str, Any] = {
+        "data": model_data,
+        "object": "list",
+    }
+    if "codex" in request.headers.get("user-agent", "").lower():
+        # Codex's remote-model discovery client uses the models envelope,
+        # while OpenAI SDKs use the standard data/object list shape.
+        response["models"] = model_data
+    return response
+
+
 @router.get(
     "/v1/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
 )
@@ -7643,6 +7703,7 @@ def _is_public_v1_model(model_name: str) -> bool:
     "/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
 )  # if project requires model list
 async def model_list(
+    request: Request,
     user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
     return_wildcard_routes: Optional[bool] = False,
     team_id: Optional[str] = None,
@@ -7740,10 +7801,7 @@ async def model_list(
             model for model in model_data if _is_public_v1_model(model.get("id", ""))
         ]
 
-        return dict(
-            data=model_data,
-            object="list",
-        )
+        return _model_list_response(model_data=model_data, request=request)
 
     # Otherwise, use the normal behavior (current implementation)
     # Get available models for the user
@@ -7777,10 +7835,7 @@ async def model_list(
         model for model in model_data if _is_public_v1_model(model.get("id", ""))
     ]
 
-    return dict(
-        data=model_data,
-        object="list",
-    )
+    return _model_list_response(model_data=model_data, request=request)
 
 
 @router.get(
diff --git a/litellm/proxy/response_api_endpoints/endpoints.py b/litellm/proxy/response_api_endpoints/endpoints.py
index 9324bc737f..51e24d691c 100644
--- a/litellm/proxy/response_api_endpoints/endpoints.py
+++ b/litellm/proxy/response_api_endpoints/endpoints.py
@@ -21,6 +21,9 @@ from litellm.proxy.common_request_processing import (
     ProxyBaseLLMRequestProcessing,
     _is_expected_max_parallel_requests_limit,
 )
+from litellm.proxy.response_api_endpoints.error_responses import (
+    responses_error_streaming_response,
+)
 from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
 from litellm.types.responses.main import DeleteResponseResult
 
@@ -56,11 +59,21 @@ def _should_return_codex_concurrency_retry(
     )
 
 
+def _should_return_codex_stream_error(
+    *, request: Request, data: Dict[str, Any]
+) -> bool:
+    return (
+        "codex" in request.headers.get("user-agent", "").lower()
+        and data.get("stream") is True
+    )
+
+
 def _codex_retry_response(
     *, message: str, mapped_error: Optional[Exception] = None
 ) -> StreamingResponse:
     error_event = {
         "type": "response.failed",
+        "sequence_number": 0,
         "response": {
             "error": {
                 "code": "rate_limit_exceeded",
@@ -76,9 +89,11 @@ def _codex_retry_response(
     async def _body():
         yield body
 
-    headers = dict(getattr(mapped_error, "headers", None) or {})
-    headers.pop("content-length", None)
-    headers.pop("content-type", None)
+    headers = {
+        key: value
+        for key, value in (getattr(mapped_error, "headers", None) or {}).items()
+        if key.lower() not in {"content-length", "content-type"}
+    }
     headers["Cache-Control"] = "no-cache"
     headers["X-Accel-Buffering"] = "no"
     return StreamingResponse(
@@ -131,6 +146,11 @@ async def _handle_responses_api_exception(
             return _codex_retry_response(
                 message=retry_message, mapped_error=mapped_error
             )
+        if _should_return_codex_stream_error(request=request, data=data):
+            return responses_error_streaming_response(
+                mapped_error,
+                headers=getattr(mapped_error, "headers", None),
+            )
         raise
 
     if (
@@ -139,6 +159,11 @@ async def _handle_responses_api_exception(
         and mapped_error.code == "429"
     ):
         return _codex_retry_response(message=retry_message, mapped_error=mapped_error)
+    if _should_return_codex_stream_error(request=request, data=data):
+        return responses_error_streaming_response(
+            mapped_error,
+            headers=getattr(mapped_error, "headers", None),
+        )
     raise mapped_error
 
 
@@ -201,7 +226,7 @@ async def responses_api(
         proxy_config,
         proxy_logging_obj,
         redis_usage_cache,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -300,7 +325,7 @@ async def responses_api(
                 llm_router=llm_router,
                 proxy_config=proxy_config,
                 proxy_logging_obj=proxy_logging_obj,
-                select_data_generator=select_data_generator,
+                select_data_generator=select_responses_data_generator,
                 user_model=user_model,
                 user_temperature=user_temperature,
                 user_request_timeout=user_request_timeout,
@@ -326,7 +351,7 @@ async def responses_api(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
@@ -617,7 +642,7 @@ async def get_response(
         proxy_config,
         proxy_logging_obj,
         redis_usage_cache,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -665,7 +690,7 @@ async def get_response(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
@@ -725,7 +750,7 @@ async def delete_response(
         proxy_config,
         proxy_logging_obj,
         redis_usage_cache,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -775,7 +800,7 @@ async def delete_response(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
@@ -821,7 +846,7 @@ async def get_response_input_items(
         llm_router,
         proxy_config,
         proxy_logging_obj,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -843,7 +868,7 @@ async def get_response_input_items(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
@@ -904,7 +929,7 @@ async def compact_response(
         llm_router,
         proxy_config,
         proxy_logging_obj,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -925,7 +950,7 @@ async def compact_response(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
@@ -990,7 +1015,7 @@ async def cancel_response(
         proxy_config,
         proxy_logging_obj,
         redis_usage_cache,
-        select_data_generator,
+        select_responses_data_generator,
         user_api_base,
         user_max_tokens,
         user_model,
@@ -1044,7 +1069,7 @@ async def cancel_response(
             llm_router=llm_router,
             general_settings=general_settings,
             proxy_config=proxy_config,
-            select_data_generator=select_data_generator,
+            select_data_generator=select_responses_data_generator,
             model=None,
             user_model=user_model,
             user_temperature=user_temperature,
diff --git a/litellm/types/llms/vertex_ai.py b/litellm/types/llms/vertex_ai.py
index 86d7b92621..b2bc5b5adf 100644
--- a/litellm/types/llms/vertex_ai.py
+++ b/litellm/types/llms/vertex_ai.py
@@ -7,9 +7,15 @@ from typing_extensions import (
 )
 
 
-class FunctionResponse(TypedDict):
-    name: str
-    response: Optional[dict]
+class FunctionResponsePart(TypedDict, total=False):
+    inline_data: "BlobType"
+    file_data: "FileDataType"
+
+
+class FunctionResponse(TypedDict, total=False):
+    name: Required[str]
+    response: Required[Optional[dict]]
+    parts: List[FunctionResponsePart]
 
 
 class FunctionCall(TypedDict):
diff --git a/tests/proxy_unit_tests/test_codex_responses_lite.py b/tests/proxy_unit_tests/test_codex_responses_lite.py
index 7f66ec07af..07de0c7dee 100644
--- a/tests/proxy_unit_tests/test_codex_responses_lite.py
+++ b/tests/proxy_unit_tests/test_codex_responses_lite.py
@@ -190,6 +190,7 @@ async def test_responses_endpoint_returns_retryable_sse_after_recording_429():
     payload = json.loads(event_lines[1].removeprefix("data: "))
     assert payload == {
         "type": "response.failed",
+        "sequence_number": 0,
         "response": {
             "error": {
                 "code": "rate_limit_exceeded",
diff --git a/tests/proxy_unit_tests/test_codex_upstream_rate_limit.py b/tests/proxy_unit_tests/test_codex_upstream_rate_limit.py
index e526f7a6ee..bc194fa6cb 100644
--- a/tests/proxy_unit_tests/test_codex_upstream_rate_limit.py
+++ b/tests/proxy_unit_tests/test_codex_upstream_rate_limit.py
@@ -50,6 +50,7 @@ async def _assert_retry_event(response):
     event = json.loads(body.split("data: ", 1)[1])
     assert event == {
         "type": "response.failed",
+        "sequence_number": 0,
         "response": {
             "error": {
                 "code": "rate_limit_exceeded",
@@ -59,6 +60,22 @@ async def _assert_retry_event(response):
     }
 
 
+async def _assert_error_event(response, mapped_error):
+    assert response.status_code == 200
+    assert response.media_type == "text/event-stream"
+    body = b"".join([chunk async for chunk in response.body_iterator]).decode()
+    assert body.startswith("event: error\n")
+    assert "[DONE]" not in body
+    event = json.loads(body.split("data: ", 1)[1])
+    assert event == {
+        "type": "error",
+        "code": str(mapped_error.code),
+        "message": mapped_error.message,
+        "param": None,
+        "sequence_number": 0,
+    }
+
+
 @pytest.mark.asyncio
 @pytest.mark.parametrize("fallback_state", ["absent", "exhausted", "success"])
 async def test_should_map_only_final_router_429_to_codex_retry(fallback_state):
@@ -181,15 +198,6 @@ async def test_should_preserve_mapped_headers_and_record_failure(
         ("openai-python/2.30.0", True, _rate_limit()),
         ("", True, _rate_limit()),
         ("codex_cli_rs/0.144.4", False, _rate_limit()),
-        ("codex_cli_rs/0.144.4", True, _rate_limit("openai")),
-        ("codex_cli_rs/0.144.4", True, HTTPException(429, "TPM limit reached")),
-        (
-            "codex_cli_rs/0.144.4",
-            True,
-            litellm.AuthenticationError(
-                message="token_revoked", llm_provider="chatgpt", model="gpt-5.6-sol"
-            ),
-        ),
     ],
 )
 async def test_should_preserve_unrelated_error_responses(user_agent, stream, error):
@@ -214,10 +222,44 @@ async def test_should_preserve_unrelated_error_responses(user_agent, stream, err
     assert exc.value is mapped_error
 
 
+@pytest.mark.asyncio
+@pytest.mark.parametrize(
+    "error",
+    [
+        _rate_limit("openai"),
+        HTTPException(429, "TPM limit reached"),
+        litellm.AuthenticationError(
+            message="token_revoked", llm_provider="chatgpt", model="gpt-5.6-sol"
+        ),
+    ],
+)
+async def test_should_return_other_codex_stream_failures_as_error_events(error):
+    processor = AsyncMock()
+    mapped_error = ProxyException(
+        message=str(error),
+        type="error",
+        param=None,
+        code=getattr(error, "status_code", 500),
+    )
+    processor._handle_llm_api_exception.side_effect = mapped_error
+
+    response = await _handle_responses_api_exception(
+        error=error,
+        request=_request(),
+        data={"stream": True},
+        processor=processor,
+        user_api_key_dict=UserAPIKeyAuth(),
+        proxy_logging_obj=None,
+        version=None,
+    )
+
+    await _assert_error_event(response, mapped_error)
+
+
 @pytest.mark.asyncio
 @pytest.mark.parametrize("handler_raises", [False, True])
 @pytest.mark.parametrize("code", [403, 500])
-async def test_should_not_hide_failure_hook_errors(code, handler_raises):
+async def test_should_expose_failure_hook_errors_in_codex_stream(code, handler_raises):
     processor = AsyncMock()
     mapped_error = ProxyException(
         message="hook error", type="error", param=None, code=code
@@ -226,17 +268,16 @@ async def test_should_not_hide_failure_hook_errors(code, handler_raises):
         processor._handle_llm_api_exception.side_effect = mapped_error
     else:
         processor._handle_llm_api_exception.return_value = mapped_error
-    with pytest.raises(ProxyException) as exc:
-        await _handle_responses_api_exception(
-            error=_rate_limit(),
-            request=_request(),
-            data={"stream": True},
-            processor=processor,
-            user_api_key_dict=UserAPIKeyAuth(),
-            proxy_logging_obj=None,
-            version=None,
-        )
-    assert exc.value is mapped_error
+    response = await _handle_responses_api_exception(
+        error=_rate_limit(),
+        request=_request(),
+        data={"stream": True},
+        processor=processor,
+        user_api_key_dict=UserAPIKeyAuth(),
+        proxy_logging_obj=None,
+        version=None,
+    )
+    await _assert_error_event(response, mapped_error)
 
 
 def _responses_ttft_timeout():
@@ -307,6 +348,7 @@ async def test_should_return_codex_ttft_408_as_retryable_http_200_after_logging(
     event = json.loads(body.removeprefix("data: "))
     assert event == {
         "type": "response.failed",
+        "sequence_number": 0,
         "response": {
             "error": {
                 "code": "rate_limit_exceeded",
diff --git a/tests/proxy_unit_tests/test_proxy_server.py b/tests/proxy_unit_tests/test_proxy_server.py
index 958d13e172..a498a1dea7 100644
--- a/tests/proxy_unit_tests/test_proxy_server.py
+++ b/tests/proxy_unit_tests/test_proxy_server.py
@@ -2063,6 +2063,7 @@ async def test_gemini_pass_through_endpoint():
     ("model_name", "expected"),
     [
         ("gpt-5.6-sol", True),
+        ("gemini-3.8-flash", True),
         ("glm-5.3", True),
         ("grok-4.5", True),
         ("gpt-5.6-sol-1", False),
@@ -2075,6 +2076,36 @@ def test_public_v1_model_allowlist(model_name, expected):
     assert _is_public_v1_model(model_name) is expected
 
 
+def test_model_list_response_adds_codex_models_envelope():
+    from fastapi import Request
+
+    from litellm.proxy.proxy_server import _model_list_response
+
+    models = [{"id": "gemini-3.8-flash", "object": "model"}]
+    codex_request = Request(
+        {
+            "type": "http",
+            "method": "GET",
+            "path": "/v1/models",
+            "headers": [(b"user-agent", b"codex_cli_rs/0.154.0")],
+        }
+    )
+    standard_request = Request(
+        {
+            "type": "http",
+            "method": "GET",
+            "path": "/v1/models",
+            "headers": [(b"user-agent", b"openai-python/2.30.0")],
+        }
+    )
+
+    codex_response = _model_list_response(models, codex_request)
+    standard_response = _model_list_response(models, standard_request)
+
+    assert codex_response == {"data": models, "object": "list", "models": models}
+    assert standard_response == {"data": models, "object": "list"}
+
+
 @pytest.mark.parametrize("hidden", [True, False])
 @pytest.mark.asyncio
 @pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@@ -2117,6 +2148,7 @@ async def test_proxy_model_group_alias_checks(prisma_client, hidden):
     request._url = URL(url="/v1/models")
 
     resp = await model_list(
+        request=request,
         user_api_key_dict=UserAPIKeyAuth(models=[]),
     )
 
@@ -2197,6 +2229,7 @@ async def test_proxy_model_group_info_rerank(prisma_client):
     request._url = URL(url="/v1/models")
 
     resp = await model_list(
+        request=request,
         user_api_key_dict=UserAPIKeyAuth(models=[]),
     )
 
diff --git a/tests/test_litellm/litellm_core_utils/prompt_templates/test_litellm_core_utils_prompt_templates_factory.py b/tests/test_litellm/litellm_core_utils/prompt_templates/test_litellm_core_utils_prompt_templates_factory.py
index 30b47a853e..d1d0689e8b 100644
--- a/tests/test_litellm/litellm_core_utils/prompt_templates/test_litellm_core_utils_prompt_templates_factory.py
+++ b/tests/test_litellm/litellm_core_utils/prompt_templates/test_litellm_core_utils_prompt_templates_factory.py
@@ -536,8 +536,11 @@ def test_convert_gemini_tool_call_result_with_image_url():
         message=message_str_format,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    # Should have inline_data for the image
-    assert isinstance(result, list) and any("inline_data" in p for p in result)
+    # Gemini requires tool-result media inside functionResponse.parts.
+    assert (
+        result["function_response"]["parts"][0]["inline_data"]["mime_type"]
+        == "image/jpeg"
+    )
 
     # Test with dict image_url format (OpenAI standard)
     message_dict_format = ChatCompletionToolMessage(
@@ -551,7 +554,10 @@ def test_convert_gemini_tool_call_result_with_image_url():
         message=message_dict_format,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    assert isinstance(result2, list) and any("inline_data" in p for p in result2)
+    assert (
+        result2["function_response"]["parts"][0]["inline_data"]["mime_type"]
+        == "image/jpeg"
+    )
 
 
 def test_convert_gemini_tool_call_result_with_anthropic_image_block():
@@ -593,8 +599,7 @@ def test_convert_gemini_tool_call_result_with_anthropic_image_block():
         message=message,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    assert isinstance(result, list), "expected a list of parts"
-    inline_parts = [p for p in result if "inline_data" in p]
+    inline_parts = result["function_response"]["parts"]
     assert len(inline_parts) == 1, "expected exactly one inline_data part"
     assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
     assert inline_parts[0]["inline_data"]["data"] == tiny_png_b64
@@ -642,8 +647,7 @@ def test_convert_gemini_tool_call_result_with_multiple_anthropic_image_blocks():
         message=message,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    assert isinstance(result, list), "expected a list of parts"
-    inline_parts = [p for p in result if "inline_data" in p]
+    inline_parts = result["function_response"]["parts"]
     assert len(inline_parts) == 2, f"expected 2 inline_data parts, got {len(inline_parts)}"
     mime_types = {p["inline_data"]["mime_type"] for p in inline_parts}
     assert mime_types == {"image/png", "image/jpeg"}
@@ -679,8 +683,7 @@ def test_convert_gemini_tool_call_result_with_data_url_string():
         message=message,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    assert isinstance(result, list), "expected a list of parts"
-    inline_parts = [p for p in result if "inline_data" in p]
+    inline_parts = result["function_response"]["parts"]
     assert len(inline_parts) == 1, "data-URL image string was not converted to inline_data"
     assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
     assert inline_parts[0]["inline_data"]["data"] == tiny_png_b64
@@ -715,8 +718,7 @@ def test_convert_gemini_tool_call_result_with_data_url_extra_params():
         message=message,
         last_message_with_tool_calls=last_message_with_tool_calls,
     )
-    assert isinstance(result, list), "expected a list of parts"
-    inline_parts = [p for p in result if "inline_data" in p]
+    inline_parts = result["function_response"]["parts"]
     assert len(inline_parts) == 1
     assert inline_parts[0]["inline_data"]["mime_type"] == "image/png", (
         f"expected clean 'image/png', got '{inline_parts[0]['inline_data']['mime_type']}'"
diff --git a/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_ai_gemini_transformation.py b/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_ai_gemini_transformation.py
index 98cdf83030..f15edcfea0 100644
--- a/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_ai_gemini_transformation.py
+++ b/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_ai_gemini_transformation.py
@@ -86,6 +86,80 @@ def test_check_if_part_exists_in_parts_camel_case_snake_case():
     assert check_if_part_exists_in_parts(parts_mixed, part_mixed_casing)
 
 
+def test_gemini_history_ending_in_assistant_adds_continuation_turn():
+    contents = _gemini_convert_messages_with_history(
+        messages=[
+            {"role": "user", "content": "Start"},
+            {"role": "assistant", "content": "Intermediate answer"},
+        ],
+        model="gemini-3.8-flash",
+    )
+
+    assert [content["role"] for content in contents] == ["user", "model", "user"]
+    assert contents[-1]["parts"] == [{"text": " "}]
+
+
+def test_gemini_history_ending_in_function_call_adds_continuation_turn():
+    contents = _gemini_convert_messages_with_history(
+        messages=[
+            {"role": "user", "content": "Inspect the repository"},
+            {
+                "role": "assistant",
+                "content": None,
+                "tool_calls": [
+                    {
+                        "id": "call_123",
+                        "type": "function",
+                        "function": {"name": "inspect", "arguments": "{}"},
+                    }
+                ],
+            },
+        ],
+        model="gemini-3.8-flash",
+    )
+
+    assert [content["role"] for content in contents] == ["user", "model", "user"]
+    assert "function_call" in contents[-2]["parts"][0]
+    assert contents[-1]["parts"] == [{"text": " "}]
+
+
+def test_gemini_history_ending_in_user_is_unchanged():
+    contents = _gemini_convert_messages_with_history(
+        messages=[{"role": "user", "content": "Continue"}],
+        model="gemini-3.8-flash",
+    )
+
+    assert contents == [{"role": "user", "parts": [{"text": "Continue"}]}]
+
+
+def test_trailing_developer_message_does_not_leave_gemini_model_turn_last():
+    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
+        VertexGeminiConfig,
+    )
+
+    messages = VertexGeminiConfig().translate_developer_role_to_system_role(
+        messages=[
+            {"role": "user", "content": "Start"},
+            {"role": "assistant", "content": "Intermediate answer"},
+            {"role": "developer", "content": "Continue with the same constraints"},
+        ]
+    )
+
+    result = _transform_request_body(
+        messages=messages,
+        model="gemini-3.8-flash",
+        optional_params={},
+        custom_llm_provider="vertex_ai",
+        litellm_params={},
+        cached_content=None,
+    )
+
+    assert result["system_instruction"]["parts"] == [
+        {"text": "Continue with the same constraints"}
+    ]
+    assert result["contents"][-1] == {"role": "user", "parts": [{"text": " "}]}
+
+
 # Tests for issue #14556: Labels field provider-aware filtering
 def test_google_genai_excludes_labels():
     """Test that Google GenAI/AI Studio endpoints exclude labels when custom_llm_provider='gemini'"""
@@ -965,36 +1039,23 @@ def test_convert_tool_response_with_base64_image():
         ]
     }
 
-    # Convert tool response (returns list when image is present)
+    # Convert tool response
     result = convert_to_gemini_tool_call_result(
         tool_message, last_message_with_tool_calls
     )
 
-    # Verify results - should be a list with 2 parts (function_response + inline_data)
-    assert isinstance(result, list), f"Expected list when image present, got {type(result)}"
-    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"
-
-    # Find function_response part and inline_data part
-    function_response_part = None
-    inline_data_part = None
-    for part in result:
-        if "function_response" in part:
-            function_response_part = part
-        elif "inline_data" in part:
-            inline_data_part = part
-
-    # Check function_response exists
-    assert function_response_part is not None, "Missing function_response part"
-    function_response = function_response_part["function_response"]
+    # Media must be nested in functionResponse.parts, not emitted as a sibling.
+    assert "function_response" in result
+    assert "inline_data" not in result
+    function_response = result["function_response"]
     assert function_response["name"] == "click_at"
     assert "response" in function_response
     # Verify JSON response is parsed correctly
     assert "url" in function_response["response"]
     assert function_response["response"]["url"] == "https://example.com"
 
-    # Check inline_data exists
-    assert inline_data_part is not None, "Missing inline_data part"
-    inline_data: BlobType = inline_data_part["inline_data"]
+    assert len(function_response["parts"]) == 1
+    inline_data: BlobType = function_response["parts"][0]["inline_data"]
     assert "data" in inline_data
     assert "mime_type" in inline_data
     assert inline_data["mime_type"] == "image/png"
@@ -1040,22 +1101,14 @@ def test_convert_tool_response_with_url_image():
             tool_message, last_message_with_tool_calls
         )
 
-        # Should be a list with 2 parts when image is present
-        assert isinstance(result, list), f"Expected list when image present, got {type(result)}"
-        assert len(result) == 2, f"Expected 2 parts, got {len(result)}"
-
-        # Find parts
-        function_response_part = next(p for p in result if "function_response" in p)
-        inline_data_part = next(p for p in result if "inline_data" in p)
-
-        # Check function_response exists
-        assert function_response_part is not None, "Missing function_response part"
-        function_response = function_response_part["function_response"]
+        assert "function_response" in result
+        assert "inline_data" not in result
+        function_response = result["function_response"]
         assert function_response["name"] == "type_text_at"
 
-        # Check inline_data exists (URL should be downloaded and converted)
-        assert inline_data_part is not None, "Missing inline_data part"
-        inline_data: BlobType = inline_data_part["inline_data"]
+        # URL should be downloaded and nested in functionResponse.parts.
+        assert len(function_response["parts"]) == 1
+        inline_data: BlobType = function_response["parts"][0]["inline_data"]
         assert "data" in inline_data
         assert "mime_type" in inline_data
     except Exception as e:
@@ -1105,6 +1158,7 @@ def test_convert_tool_response_text_only():
 
     # Check inline_data does NOT exist (no image provided)
     assert "inline_data" not in result
+    assert "parts" not in function_response
 
 
 def test_file_data_field_order():
@@ -1341,36 +1395,22 @@ def test_convert_tool_response_with_pdf_file():
         ]
     }
 
-    # Convert tool response (returns list when file is present)
+    # Convert tool response
     result = convert_to_gemini_tool_call_result(
         tool_message, last_message_with_tool_calls
     )
 
-    # Verify results - should be a list with 2 parts (function_response + inline_data)
-    assert isinstance(result, list), f"Expected list when file present, got {type(result)}"
-    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"
-
-    # Find function_response part and inline_data part
-    function_response_part = None
-    inline_data_part = None
-    for part in result:
-        if "function_response" in part:
-            function_response_part = part
-        elif "inline_data" in part:
-            inline_data_part = part
-
-    # Check function_response exists
-    assert function_response_part is not None, "Missing function_response part"
-    function_response = function_response_part["function_response"]
+    assert "function_response" in result
+    assert "inline_data" not in result
+    function_response = result["function_response"]
     assert function_response["name"] == "analyze_document"
     assert "response" in function_response
     # Verify JSON response is parsed correctly
     assert "status" in function_response["response"]
     assert function_response["response"]["status"] == "success"
 
-    # Check inline_data exists
-    assert inline_data_part is not None, "Missing inline_data part"
-    inline_data: BlobType = inline_data_part["inline_data"]
+    assert len(function_response["parts"]) == 1
+    inline_data: BlobType = function_response["parts"][0]["inline_data"]
     assert "data" in inline_data
     assert "mime_type" in inline_data
     assert inline_data["mime_type"] == "application/pdf"
@@ -1413,19 +1453,11 @@ def test_convert_tool_response_with_input_file_type():
         tool_message, last_message_with_tool_calls
     )
 
-    # Verify results
-    assert isinstance(result, list), f"Expected list when file present, got {type(result)}"
-    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"
-
-    # Find inline_data part
-    inline_data_part = None
-    for part in result:
-        if "inline_data" in part:
-            inline_data_part = part
-
-    # Check inline_data exists
-    assert inline_data_part is not None, "Missing inline_data part"
-    assert inline_data_part["inline_data"]["mime_type"] == "application/pdf"
+    function_response = result["function_response"]
+    assert (
+        function_response["parts"][0]["inline_data"]["mime_type"]
+        == "application/pdf"
+    )
 
 
 def test_convert_tool_response_with_nested_file_object():
@@ -1466,19 +1498,9 @@ def test_convert_tool_response_with_nested_file_object():
         tool_message, last_message_with_tool_calls
     )
 
-    # Verify results - should be a list with 2 parts
-    assert isinstance(result, list), f"Expected list when file present, got {type(result)}"
-    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"
-
-    # Find inline_data part
-    inline_data_part = None
-    for part in result:
-        if "inline_data" in part:
-            inline_data_part = part
-
-    # Check inline_data exists
-    assert inline_data_part is not None, "Missing inline_data part"
-    inline_data: BlobType = inline_data_part["inline_data"]
+    function_response = result["function_response"]
+    assert len(function_response["parts"]) == 1
+    inline_data: BlobType = function_response["parts"][0]["inline_data"]
     assert "data" in inline_data
     assert "mime_type" in inline_data
     assert inline_data["mime_type"] == "application/pdf"
@@ -1522,7 +1544,7 @@ def test_assistant_message_with_images_field():
     contents = _gemini_convert_messages_with_history(messages=messages)
     
     # Verify structure
-    assert len(contents) == 2, f"Expected 2 content blocks, got {len(contents)}"
+    assert len(contents) == 3, f"Expected 3 content blocks, got {len(contents)}"
     
     # Verify user message
     assert contents[0]["role"] == "user"
@@ -1553,6 +1575,7 @@ def test_assistant_message_with_images_field():
     assert "mime_type" in inline_data
     assert inline_data["mime_type"] == "image/png"
     assert inline_data["data"] == test_image_base64
+    assert contents[2] == {"role": "user", "parts": [{"text": " "}]}
 
 
 def test_assistant_message_with_multiple_images():
@@ -1757,6 +1780,55 @@ def test_function_response_has_user_role():
     assert "function_response" in contents[2]["parts"][0]
 
 
+def test_image_tool_response_is_nested_under_function_response_parts():
+    image_base64 = (
+        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
+        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
+    )
+    messages = [
+        {"role": "user", "content": "Inspect this image"},
+        {
+            "role": "assistant",
+            "content": None,
+            "tool_calls": [
+                {
+                    "id": "call_image",
+                    "type": "function",
+                    "function": {"name": "inspect_image", "arguments": "{}"},
+                }
+            ],
+        },
+        {
+            "role": "tool",
+            "tool_call_id": "call_image",
+            "content": [
+                {
+                    "type": "input_image",
+                    "image_url": f"data:image/png;base64,{image_base64}",
+                }
+            ],
+        },
+    ]
+
+    contents = _gemini_convert_messages_with_history(
+        messages=messages, model="gemini-3.8-flash"
+    )
+
+    assert [content["role"] for content in contents] == ["user", "model", "user"]
+    assert len(contents[2]["parts"]) == 1
+    function_response = contents[2]["parts"][0]["function_response"]
+    assert function_response["name"] == "inspect_image"
+    assert function_response["response"] == {"content": ""}
+    assert function_response["parts"] == [
+        {
+            "inline_data": {
+                "mime_type": "image/png",
+                "data": image_base64,
+            }
+        }
+    ]
+
+
 def test_multi_turn_function_calling_roles():
     """
     Test a full multi-turn function calling conversation produces correct roles.
diff --git a/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_and_google_ai_studio_gemini.py b/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_and_google_ai_studio_gemini.py
index ddc404cb8c..bd655cdc1b 100644
--- a/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_and_google_ai_studio_gemini.py
+++ b/tests/test_litellm/llms/vertex_ai/gemini/test_vertex_and_google_ai_studio_gemini.py
@@ -2112,6 +2112,20 @@ def test_reasoning_effort_maps_to_thinking_level_gemini_3():
     assert result["thinkingConfig"]["includeThoughts"] is False
 
 
+@pytest.mark.parametrize("reasoning_effort", ["max", "xhigh"])
+def test_codex_max_reasoning_effort_maps_to_gemini_high(reasoning_effort):
+    assert VertexGeminiConfig._map_reasoning_effort_to_thinking_level(
+        reasoning_effort, model="gemini-3.8-flash"
+    ) == VertexGeminiConfig._map_reasoning_effort_to_thinking_level(
+        "high", model="gemini-3.8-flash"
+    )
+    assert VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(
+        reasoning_effort, model="gemini-2.5-pro"
+    ) == VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(
+        "high", model="gemini-2.5-pro"
+    )
+
+
 def test_reasoning_effort_dict_format_gemini_3():
     """
     Test that reasoning_effort works when passed as dict format from OpenAI Agents SDK.
diff --git a/tests/test_litellm/proxy/test_proxy_server.py b/tests/test_litellm/proxy/test_proxy_server.py
index e211a31c6b..d494eae3c0 100644
--- a/tests/test_litellm/proxy/test_proxy_server.py
+++ b/tests/test_litellm/proxy/test_proxy_server.py
@@ -2225,6 +2225,56 @@ async def test_async_data_generator_midstream_error():
     mock_proxy_logging_obj.post_call_failure_hook.assert_not_called()
 
 
+@pytest.mark.asyncio
+async def test_responses_data_generator_normalizes_bare_midstream_error_event():
+    from litellm.proxy._types import UserAPIKeyAuth
+    from litellm.proxy.proxy_server import async_data_generator
+
+    bare_error = (
+        'data: {"error":{"code":"invalid_request_error",'
+        '"message":"Gemini rejected the final model turn","param":null}}'
+    )
+
+    async def mock_streaming_iterator(*args, **kwargs):
+        yield bare_error
+
+    mock_proxy_logging_obj = MagicMock()
+    mock_proxy_logging_obj.async_post_call_streaming_iterator_hook = (
+        mock_streaming_iterator
+    )
+    mock_proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
+        side_effect=lambda **kwargs: kwargs["response"]
+    )
+    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()
+    mock_response = MagicMock()
+    mock_response.aclose = AsyncMock()
+
+    with patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj):
+        chunks = [
+            chunk
+            async for chunk in async_data_generator(
+                mock_response,
+                UserAPIKeyAuth(),
+                {"model": "gemini-3.8-flash", "stream": True},
+                is_responses_api=True,
+            )
+        ]
+
+    assert len(chunks) == 1
+    assert chunks[0].startswith("event: error\n")
+    assert "[DONE]" not in chunks[0]
+    event = json.loads(chunks[0].split("data: ", 1)[1])
+    assert event == {
+        "type": "error",
+        "code": "invalid_request_error",
+        "message": "Gemini rejected the final model turn",
+        "param": None,
+        "sequence_number": 0,
+    }
+    assert "error" not in event
+    mock_response.aclose.assert_awaited_once()
+
+
 def _has_nested_none_values(obj, path="root"):
     """
     Recursively check if an object contains nested None values.
@@ -4817,13 +4867,16 @@ async def test_async_data_generator_maps_responses_overload_to_retryable_event()
                 mock_response,
                 MagicMock(spec=UserAPIKeyAuth),
                 {"model": "test-model", "stream": True},
+                is_responses_api=True,
             )
         ]
 
     assert len(chunks) == 1
-    event = json.loads(chunks[0].removeprefix("data: "))
+    assert chunks[0].startswith("event: response.failed\n")
+    event = json.loads(chunks[0].split("data: ", 1)[1])
     assert event == {
         "type": "response.failed",
+        "sequence_number": 0,
         "response": {
             "error": {
                 "code": "rate_limit_exceeded",
diff --git a/tests/test_litellm/responses/litellm_completion_transformation/test_function_call_output_normalization.py b/tests/test_litellm/responses/litellm_completion_transformation/test_function_call_output_normalization.py
index 19aeba7f9c..3633b8187e 100644
--- a/tests/test_litellm/responses/litellm_completion_transformation/test_function_call_output_normalization.py
+++ b/tests/test_litellm/responses/litellm_completion_transformation/test_function_call_output_normalization.py
@@ -38,3 +38,54 @@ def test_function_call_output_string_passthrough():
     assert len(out) == 1
     assert out[0]["content"] == '{"ok":true}'
 
+
+def test_image_function_call_output_uses_gemini_function_response_parts():
+    from litellm.llms.vertex_ai.gemini.transformation import (
+        _gemini_convert_messages_with_history,
+    )
+
+    image_base64 = (
+        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
+        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
+    )
+    transform_input = (
+        LiteLLMCompletionResponsesConfig
+        ._transform_response_input_param_to_chat_completion_message
+    )
+    messages = transform_input(
+        input=[
+            {
+                "type": "message",
+                "role": "user",
+                "content": [{"type": "input_text", "text": "Inspect the image"}],
+            },
+            {
+                "type": "function_call",
+                "name": "inspect_image",
+                "call_id": "call_image",
+                "arguments": "{}",
+            },
+            {
+                "type": "function_call_output",
+                "call_id": "call_image",
+                "output": [
+                    {
+                        "type": "input_image",
+                        "image_url": f"data:image/png;base64,{image_base64}",
+                    }
+                ],
+            },
+        ]
+    )
+
+    contents = _gemini_convert_messages_with_history(
+        messages=messages, model="gemini-3.8-flash"
+    )
+    function_response = contents[-1]["parts"][0]["function_response"]
+
+    assert contents[-1]["role"] == "user"
+    assert function_response["name"] == "inspect_image"
+    assert function_response["parts"][0]["inline_data"] == {
+        "mime_type": "image/png",
+        "data": image_base64,
+    }
diff --git a/docker-compose.test.yml b/docker-compose.test.yml
new file mode 100644
index 0000000000..8258064653
--- /dev/null
+++ b/docker-compose.test.yml
@@ -0,0 +1,53 @@
+name: litellm-codex-gemini-test
+
+services:
+  test-db:
+    image: postgres:16
+    container_name: litellm-codex-gemini-test-db
+    environment:
+      POSTGRES_DB: litellm_test
+      POSTGRES_USER: llmproxy_test
+      POSTGRES_PASSWORD: litellm_test_only
+    volumes:
+      - litellm_codex_gemini_test_data:/var/lib/postgresql/data
+    healthcheck:
+      test: ["CMD-SHELL", "pg_isready -d litellm_test -U llmproxy_test"]
+      interval: 1s
+      timeout: 5s
+      retries: 20
+
+  litellm-test:
+    image: ${LITELLM_TEST_IMAGE:-litellm:codex-gemini-test}
+    container_name: litellm-litellm-test-1
+    command:
+      - "--config=/app/config.yaml"
+    ports:
+      - "127.0.0.1:4001:4000"
+    environment:
+      DATABASE_URL: postgresql://llmproxy_test:litellm_test_only@test-db:5432/litellm_test
+      DISABLE_SCHEMA_UPDATE: "false"
+      STORE_MODEL_IN_DB: "True"
+      CHATGPT_TOKEN_DIR: "/root/litellm/auth"
+      CHATGPT_DISABLE_INTERACTIVE_LOGIN: "true"
+      LITELLM_LOCAL_MODEL_COST_MAP: "true"
+    env_file:
+      - /root/litellm/.env
+    volumes:
+      - /root/litellm/config.yaml:/app/config.yaml:ro
+      - /root/litellm/auth:/root/litellm/auth:ro
+    depends_on:
+      test-db:
+        condition: service_healthy
+    restart: unless-stopped
+    healthcheck:
+      test:
+        - CMD-SHELL
+        - python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/readiness')"
+      interval: 5s
+      timeout: 3s
+      retries: 24
+      start_period: 15s
+
+volumes:
+  litellm_codex_gemini_test_data:
+    name: litellm_codex_gemini_test_data
diff --git a/docker/Dockerfile.codex-gemini-test b/docker/Dockerfile.codex-gemini-test
new file mode 100644
index 0000000000..eceef4fd4b
--- /dev/null
+++ b/docker/Dockerfile.codex-gemini-test
@@ -0,0 +1,21 @@
+ARG LITELLM_TEST_BASE_IMAGE=mookim/litellm:9511eb7165df
+FROM ${LITELLM_TEST_BASE_IMAGE}
+
+USER root
+RUN python -m pip install --no-cache-dir \
+    black==24.10.0 \
+    fakeredis==2.34.1 \
+    parameterized==0.9.0 \
+    psycopg-binary==3.3.5 \
+    pytest==8.3.5 \
+    pytest-asyncio==1.2.0 \
+    pytest-mock==3.15.1 \
+    pytest-postgresql==7.0.2 \
+    requests-mock==1.12.1 \
+    responses==0.26.0 \
+    respx==0.22.0 \
+    ruff==0.15.3
+
+COPY litellm /app/litellm
+ENV PYTHONPATH=/app
+WORKDIR /app
diff --git a/litellm/proxy/response_api_endpoints/error_responses.py b/litellm/proxy/response_api_endpoints/error_responses.py
new file mode 100644
index 0000000000..4474f007f3
--- /dev/null
+++ b/litellm/proxy/response_api_endpoints/error_responses.py
@@ -0,0 +1,112 @@
+import json
+from enum import Enum
+from typing import Any, Dict, Mapping, Optional
+
+from starlette.responses import StreamingResponse
+
+
+def _as_error_mapping(error: Any) -> Mapping[str, Any]:
+    """Extract an OpenAI-style error object without exposing exception internals."""
+    value: Any = error
+    if isinstance(value, str):
+        candidate = value.strip()
+        if candidate.startswith("event:"):
+            candidate = "\n".join(
+                line.removeprefix("data: ")
+                for line in candidate.splitlines()
+                if line.startswith("data: ")
+            )
+        elif candidate.startswith("data: "):
+            candidate = candidate.removeprefix("data: ").strip()
+        try:
+            value = json.loads(candidate)
+        except (json.JSONDecodeError, TypeError):
+            return {"message": error}
+
+    if isinstance(value, Mapping):
+        nested_error = value.get("error")
+        if isinstance(nested_error, Mapping):
+            return nested_error
+        return value
+
+    detail = getattr(value, "detail", None)
+    if isinstance(detail, Mapping):
+        nested_error = detail.get("error")
+        if isinstance(nested_error, Mapping):
+            return nested_error
+        return detail
+
+    return {
+        "code": getattr(value, "code", None) or getattr(value, "status_code", None),
+        "message": getattr(value, "message", None) or detail or str(value),
+        "param": getattr(value, "param", None),
+    }
+
+
+def _string_value(value: Any, *, default: str) -> str:
+    if value is None or value == "None":
+        return default
+    if isinstance(value, Enum):
+        value = value.value
+    if isinstance(value, (dict, list)):
+        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
+    return str(value)
+
+
+def build_responses_error_event(
+    error: Any, *, sequence_number: int = 0
+) -> Dict[str, Any]:
+    """Build the top-level ``error`` event defined by the Responses API."""
+    error_mapping = _as_error_mapping(error)
+    param = error_mapping.get("param")
+    if param == "None":
+        param = None
+    elif param is not None:
+        param = _string_value(param, default="")
+    return {
+        "type": "error",
+        "code": _string_value(error_mapping.get("code"), default="server_error"),
+        "message": _string_value(
+            error_mapping.get("message") or error_mapping.get("error"),
+            default="An unknown error occurred.",
+        ),
+        "param": param,
+        "sequence_number": sequence_number,
+    }
+
+
+def serialize_responses_error_event(error: Any, *, sequence_number: int = 0) -> str:
+    event = build_responses_error_event(error, sequence_number=sequence_number)
+    return (
+        "event: error\n"
+        f"data: {json.dumps(event, separators=(',', ':'), ensure_ascii=False)}\n\n"
+    )
+
+
+def responses_error_streaming_response(
+    error: Any,
+    *,
+    headers: Optional[Mapping[str, str]] = None,
+    sequence_number: int = 0,
+) -> StreamingResponse:
+    """Return a terminal, spec-compliant Responses error SSE stream."""
+    body = serialize_responses_error_event(
+        error, sequence_number=sequence_number
+    ).encode("utf-8")
+
+    async def _body():
+        yield body
+
+    response_headers = {
+        key: value
+        for key, value in (headers or {}).items()
+        if key.lower() not in {"content-length", "content-type"}
+    }
+    response_headers["Cache-Control"] = "no-cache"
+    response_headers["X-Accel-Buffering"] = "no"
+    return StreamingResponse(
+        _body(),
+        status_code=200,
+        media_type="text/event-stream",
+        headers=response_headers,
+    )
```
