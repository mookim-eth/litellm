import asyncio
import contextlib
import json
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Union, cast
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.websockets import WebSocket

from litellm._logging import verbose_proxy_logger
from litellm.constants import STREAM_CLOSE_TIMEOUT_SECONDS
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import (
    UserAPIKeyAuth,
    user_api_key_auth,
    user_api_key_auth_websocket,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.responses.main import DeleteResponseResult

router = APIRouter()

CODEX_RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"
RESPONSES_SSE_KEEPALIVE = b": keepalive\n\n"
DEFAULT_RESPONSES_KEEPALIVE_INTERVAL_SECONDS = 45.0
DEFAULT_RESPONSES_PROVIDER_START_TIMEOUT_SECONDS = 300.0
RESPONSES_KEEPALIVE_ENABLED_SETTING = "enable_responses_stream_keepalive"
RESPONSES_KEEPALIVE_INTERVAL_SETTING = (
    "responses_stream_keepalive_interval_seconds"
)
RESPONSES_PROVIDER_START_TIMEOUT_SETTING = (
    "responses_provider_start_timeout_seconds"
)


def _apply_codex_responses_lite_request_overrides(
    data: Dict[str, Any], request: Request
) -> None:
    header_value = request.headers.get(CODEX_RESPONSES_LITE_HEADER)
    if header_value is None:
        return

    extra_headers = data.get("extra_headers")
    if not isinstance(extra_headers, dict):
        extra_headers = {}
        data["extra_headers"] = extra_headers
    extra_headers[CODEX_RESPONSES_LITE_HEADER] = header_value
    data["parallel_tool_calls"] = False


def _is_codex_responses_lite_request(data: Dict[str, Any]) -> bool:
    extra_headers = data.get("extra_headers")
    return isinstance(extra_headers, dict) and bool(
        extra_headers.get(CODEX_RESPONSES_LITE_HEADER)
    )


def _get_responses_provider_start_timeout(general_settings: dict) -> float:
    value = general_settings.get(
        RESPONSES_PROVIDER_START_TIMEOUT_SETTING,
        DEFAULT_RESPONSES_PROVIDER_START_TIMEOUT_SECONDS,
    )
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = DEFAULT_RESPONSES_PROVIDER_START_TIMEOUT_SECONDS
    if timeout <= 0:
        timeout = DEFAULT_RESPONSES_PROVIDER_START_TIMEOUT_SECONDS
    return timeout


def _get_responses_keepalive_interval(general_settings: dict) -> float:
    value = general_settings.get(
        RESPONSES_KEEPALIVE_INTERVAL_SETTING,
        DEFAULT_RESPONSES_KEEPALIVE_INTERVAL_SECONDS,
    )
    try:
        interval = float(value)
    except (TypeError, ValueError):
        interval = DEFAULT_RESPONSES_KEEPALIVE_INTERVAL_SECONDS
    if interval <= 0:
        interval = DEFAULT_RESPONSES_KEEPALIVE_INTERVAL_SECONDS
    return interval


def _should_enable_responses_keepalive(
    data: Dict[str, Any], general_settings: dict
) -> bool:
    return _is_codex_responses_lite_request(data) or (
        general_settings.get(RESPONSES_KEEPALIVE_ENABLED_SETTING) is True
    )


async def _wait_for_request_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.1)


async def _await_with_request_disconnect(
    request: Request,
    operation: Awaitable[Any],
    provider_start_timeout_seconds: float,
) -> Any:
    """Cancel an inline provider operation when its downstream disappears."""
    request_task = asyncio.current_task()
    if request_task is None:
        return await operation

    disconnected = asyncio.Event()

    async def cancel_request_on_disconnect() -> None:
        await _wait_for_request_disconnect(request)
        disconnected.set()
        request_task.cancel()

    watcher = asyncio.create_task(cancel_request_on_disconnect())
    try:
        with anyio.fail_after(provider_start_timeout_seconds):
            return await operation
    except asyncio.CancelledError:
        if disconnected.is_set():
            raise ClientDisconnect() from None
        raise
    finally:
        watcher.cancel()
        with contextlib.suppress(BaseException):
            await watcher


def _response_failed_sse(
    error: Any, *, response_id: Optional[str] = None
) -> bytes:
    """Serialize an error after the streaming HTTP response has started."""
    message = getattr(error, "message", None) or str(error)
    code = (
        getattr(error, "code", None)
        or getattr(error, "status_code", None)
        or "proxy_error"
    )
    event = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": response_id or f"resp_{uuid4()}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "failed",
            "error": {"code": str(code), "message": message},
        },
    }
    return (
        "event: response.failed\n"
        f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
    ).encode()


def _response_error_from_started_response(response: Response) -> Any:
    """Extract an OpenAI-style error from a non-stream response."""
    body = getattr(response, "body", b"")
    if isinstance(body, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            body = body.decode()
    if isinstance(body, str):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                error = parsed.get("error", parsed)
                if isinstance(error, dict):
                    return SimpleNamespace(
                        message=error.get("message", body),
                        code=error.get("code", response.status_code),
                    )
    return RuntimeError(
        f"Expected a streaming Responses response, got HTTP {response.status_code}"
    )


async def _deferred_responses_stream(
    process_request: Callable[[], Awaitable[Any]],
    provider_start_timeout_seconds: float = (
        DEFAULT_RESPONSES_PROVIDER_START_TIMEOUT_SECONDS
    ),
    keepalive_interval_seconds: float = (
        DEFAULT_RESPONSES_KEEPALIVE_INTERVAL_SECONDS
    ),
    send_initial_keepalive: bool = True,
    error_handler: Optional[Callable[[Exception], Awaitable[Exception]]] = None,
) -> AsyncIterator[Union[bytes, str]]:
    """Forward one provider stream while emitting protocol-neutral SSE comments."""
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    stream_finished = object()

    async def _produce_provider_stream() -> None:
        response: Optional[Response] = None
        body_iterator: Optional[Any] = None
        producer_cancelled = False
        try:
            with anyio.fail_after(provider_start_timeout_seconds):
                response = await process_request()
            if not isinstance(response, StreamingResponse):
                await queue.put(
                    _response_failed_sse(
                        _response_error_from_started_response(response)
                    )
                )
                return

            body_iterator = response.body_iterator
            async for chunk in body_iterator:
                await queue.put(chunk)
        except asyncio.CancelledError:
            producer_cancelled = True
            raise
        except TimeoutError:
            error: Exception = HTTPException(
                status_code=504,
                detail=(
                    "Provider did not start a Responses stream within "
                    f"{provider_start_timeout_seconds:g} seconds"
                ),
            )
            if error_handler is not None:
                error = await error_handler(error)
            await queue.put(_response_failed_sse(error))
        except Exception as error:
            if error_handler is not None:
                error = await error_handler(error)
            await queue.put(_response_failed_sse(error))
        finally:
            close = getattr(body_iterator, "aclose", None)
            if callable(close):
                with anyio.move_on_after(
                    STREAM_CLOSE_TIMEOUT_SECONDS, shield=True
                ) as cancel_scope:
                    with contextlib.suppress(BaseException):
                        await close()
                if cancel_scope.cancelled_caught:
                    verbose_proxy_logger.warning(
                        "Timed out after %.1fs closing deferred Responses stream",
                        STREAM_CLOSE_TIMEOUT_SECONDS,
                    )
            background = getattr(response, "background", None)
            if background is not None:
                with anyio.move_on_after(
                    STREAM_CLOSE_TIMEOUT_SECONDS, shield=True
                ) as background_scope:
                    with contextlib.suppress(Exception):
                        await background()
                if background_scope.cancelled_caught:
                    verbose_proxy_logger.warning(
                        "Timed out after %.1fs running deferred Responses cleanup",
                        STREAM_CLOSE_TIMEOUT_SECONDS,
                    )
            if not producer_cancelled:
                await queue.put(stream_finished)

    if send_initial_keepalive:
        yield RESPONSES_SSE_KEEPALIVE
    producer_task = asyncio.create_task(_produce_provider_stream())
    queue_read_task: Optional[asyncio.Task] = None
    try:
        while True:
            if queue_read_task is None:
                queue_read_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {queue_read_task}, timeout=keepalive_interval_seconds
            )
            if not done:
                if producer_task.done():
                    await producer_task
                    break
                yield RESPONSES_SSE_KEEPALIVE
                continue
            item = queue_read_task.result()
            queue_read_task = None
            if item is stream_finished:
                break
            yield item

            if producer_task.done() and queue.empty():
                await producer_task
                break
    finally:
        if not producer_task.done():
            producer_task.cancel()
            with anyio.move_on_after(
                STREAM_CLOSE_TIMEOUT_SECONDS, shield=True
            ) as cancel_scope:
                with contextlib.suppress(BaseException):
                    await producer_task
            if cancel_scope.cancelled_caught:
                verbose_proxy_logger.warning(
                    "Timed out after %.1fs cancelling deferred Responses stream",
                    STREAM_CLOSE_TIMEOUT_SECONDS,
                )
        if queue_read_task is not None and not queue_read_task.done():
            queue_read_task.cancel()
            with contextlib.suppress(BaseException):
                await queue_read_task


@router.post(
    "/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def responses_api(  # noqa: PLR0915
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses

    Supports background mode with polling_via_cache for partial response retrieval.
    When background=true and polling_via_cache is enabled, returns a polling_id immediately
    and streams the response in the background, updating Redis cache.

    ```bash
    # Normal request
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI"
    }'

    # Background request with polling
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI",
        "background": true
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        apply_pro_header_model_override,
        general_settings,
        llm_router,
        native_background_mode,
        polling_cache_ttl,
        polling_via_cache_enabled,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    apply_pro_header_model_override(data=data, request=request)
    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    # Check if polling via cache should be used for this request
    from litellm.proxy.response_polling.polling_handler import (
        should_use_polling_for_request,
    )

    should_use_polling = should_use_polling_for_request(
        background_mode=data.get("background", False),
        polling_via_cache_enabled=polling_via_cache_enabled,
        redis_cache=redis_usage_cache,
        model=data.get("model", ""),
        llm_router=llm_router,
        native_background_mode=native_background_mode,
    )

    # If polling is enabled, use polling mode
    if should_use_polling:
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )
        from litellm.proxy.response_polling.polling_handler import (
            ResponsePollingHandler,
        )

        verbose_proxy_logger.info(
            f"Starting background response with polling for model={data.get('model')}"
        )

        # Run pre-call checks (rate limits, guardrails, budget) BEFORE creating
        # polling ID. This ensures rate-limited requests get a synchronous 429
        # instead of a polling ID that immediately fails in the background task.
        processor = ProxyBaseLLMRequestProcessing(data=data)
        try:
            data, _logging_obj = await processor.common_processing_pre_call_logic(
                request=request,
                general_settings=general_settings,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                version=version,
                proxy_config=proxy_config,
                user_model=user_model,
                user_temperature=user_temperature,
                user_request_timeout=user_request_timeout,
                user_max_tokens=user_max_tokens,
                user_api_base=user_api_base,
                model=None,
                route_type="aresponses",
                llm_router=llm_router,
            )
        except Exception as e:
            raise await processor._handle_llm_api_exception(
                e=e,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
                version=version,
            )

        # Initialize polling handler with configured TTL (from global config)
        polling_handler = ResponsePollingHandler(
            redis_cache=redis_usage_cache,
            ttl=polling_cache_ttl,  # Global var set at startup
        )

        # Generate polling ID
        polling_id = ResponsePollingHandler.generate_polling_id()

        # Create initial state in Redis
        initial_state = await polling_handler.create_initial_state(
            polling_id=polling_id,
            request_data=data,
        )

        # Start background task to stream and update cache.
        # Pass pre-processed data so the background task skips pre-call logic
        # (rate limits, guardrails already checked above).
        asyncio.create_task(
            background_streaming_task(
                polling_id=polling_id,
                data=data.copy(),
                polling_handler=polling_handler,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                general_settings=general_settings,
                llm_router=llm_router,
                proxy_config=proxy_config,
                proxy_logging_obj=proxy_logging_obj,
                select_data_generator=select_data_generator,
                user_model=user_model,
                user_temperature=user_temperature,
                user_request_timeout=user_request_timeout,
                user_max_tokens=user_max_tokens,
                user_api_base=user_api_base,
                version=version,
            )
        )

        # Return OpenAI Response object format (initial state)
        # https://platform.openai.com/docs/api-reference/responses/object
        return initial_state

    # Normal response flow
    processor = ProxyBaseLLMRequestProcessing(data=data)
    provider_start_timeout_seconds = _get_responses_provider_start_timeout(
        general_settings
    )
    keepalive_interval_seconds = _get_responses_keepalive_interval(
        general_settings
    )

    async def _process_request(*, skip_pre_call_logic: bool = False) -> Any:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
            skip_pre_call_logic=skip_pre_call_logic,
        )

    try:
        if data.get("stream") is True and _should_enable_responses_keepalive(
            data, general_settings
        ):
            processor.data, logging_obj = (
                await processor.common_processing_pre_call_logic(
                    request=request,
                    general_settings=general_settings,
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_dict=user_api_key_dict,
                    version=version,
                    proxy_config=proxy_config,
                    user_model=user_model,
                    user_temperature=user_temperature,
                    user_request_timeout=user_request_timeout,
                    user_max_tokens=user_max_tokens,
                    user_api_base=user_api_base,
                    model=None,
                    route_type="aresponses",
                    llm_router=llm_router,
                )
            )

            async def _process_deferred_request() -> Any:
                return await _process_request(skip_pre_call_logic=True)

            async def _map_deferred_error(error: Exception) -> Exception:
                try:
                    await processor._handle_llm_api_exception(
                        e=error,
                        user_api_key_dict=user_api_key_dict,
                        proxy_logging_obj=proxy_logging_obj,
                        version=version,
                    )
                except Exception as mapped_error:
                    return mapped_error
                return error

            initial_headers = dict(fastapi_response.headers)
            initial_headers.pop("content-length", None)
            initial_headers.pop("content-type", None)
            initial_headers.setdefault("Cache-Control", "no-cache")
            initial_headers.setdefault("X-Accel-Buffering", "no")
            initial_headers.update(
                ProxyBaseLLMRequestProcessing.get_custom_headers(
                    user_api_key_dict=user_api_key_dict,
                    call_id=logging_obj.litellm_call_id,
                    model_id=processor.maybe_get_model_id(logging_obj),
                    version=version,
                    model_region=getattr(
                        user_api_key_dict, "allowed_model_region", ""
                    ),
                    request_data=processor.data,
                    litellm_logging_obj=logging_obj,
                )
            )
            return StreamingResponse(
                _deferred_responses_stream(
                    _process_deferred_request,
                    provider_start_timeout_seconds=provider_start_timeout_seconds,
                    keepalive_interval_seconds=keepalive_interval_seconds,
                    send_initial_keepalive=_is_codex_responses_lite_request(
                        data
                    ),
                    error_handler=_map_deferred_error,
                ),
                media_type="text/event-stream",
                headers=initial_headers,
            )

        if data.get("stream") is True:
            try:
                response = await _await_with_request_disconnect(
                    request,
                    _process_request(),
                    provider_start_timeout_seconds,
                )
            except ClientDisconnect:
                return Response(status_code=499)
        else:
            response = await _process_request()

        # Store in managed objects table if background mode is enabled
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                from litellm_enterprise.proxy.hooks.managed_files import (  # type: ignore
                    _PROXY_LiteLLMManagedFiles,
                )

                managed_files_obj = cast(
                    Optional[_PROXY_LiteLLMManagedFiles],
                    proxy_logging_obj.get_proxy_hook("managed_files"),
                )

                if managed_files_obj and llm_router:
                    try:
                        # Get the actual deployment model_id from hidden params
                        hidden_params = getattr(response, "_hidden_params", {}) or {}
                        model_id = hidden_params.get("model_id", None)

                        if not model_id:
                            verbose_proxy_logger.warning(
                                f"No model_id found in response hidden params for response {response.id}, skipping managed object storage"
                            )
                            raise Exception(
                                "No model_id found in response hidden params"
                            )
                        # Store in managed objects table
                        await managed_files_obj.store_unified_object_id(
                            unified_object_id=response.id,
                            file_object=response,
                            litellm_parent_otel_span=None,
                            model_object_id=response.id,
                            file_purpose="response",
                            user_api_key_dict=user_api_key_dict,
                        )

                        verbose_proxy_logger.info(
                            f"Stored background response {response.id} in managed objects table with unified_id={response.id}"
                        )
                    except Exception as e:
                        verbose_proxy_logger.error(
                            f"Failed to store background response in managed objects table: {str(e)}"
                        )

        return response
    except ModifyResponseException as e:
        # Guardrail passthrough: return violation message in Responses API format (200)
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )

        violation_text = e.message
        response_obj = ResponsesAPIResponse(
            id=f"resp_{uuid4()}",
            object="response",
            created_at=int(time.time()),
            model=e.model or data.get("model"),
            output=cast(Any, [{"content": [{"type": "text", "text": violation_text}]}]),
            status="completed",
            usage=ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        return response_obj
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/cursor/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def cursor_chat_completions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cursor-specific endpoint that accepts Responses API input format but returns chat completions format.
    
    This endpoint handles requests from Cursor IDE which sends Responses API format (`input` field)
    but expects chat completions format response (`choices`, `messages`, etc.).
    
    ```bash
    curl -X POST http://localhost:4000/cursor/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}]
    }'
    Responds back in chat completions format.
    ```
    """
    from litellm.completion_extras.litellm_responses_transformation.handler import (
        responses_api_bridge,
    )
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.proxy_server import (
        _read_request_body,
        apply_pro_header_model_override,
        async_data_generator,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.utils import ModelResponse

    data = await _read_request_body(request=request)
    apply_pro_header_model_override(data=data, request=request)

    # Convert 'messages' to 'input' for Responses API compatibility
    # Cursor sends 'messages' but Responses API expects 'input'
    if "messages" in data and "input" not in data:
        data["input"] = data.pop("messages")

    processor = ProxyBaseLLMRequestProcessing(data=data)

    def cursor_data_generator(response, user_api_key_dict, request_data):
        """
        Custom generator that transforms Responses API streaming chunks to chat completion chunks.

        This generator is used for the cursor endpoint to convert Responses API format responses
        to chat completion format that Cursor IDE expects.

        Args:
            response: The streaming response (BaseResponsesAPIStreamingIterator or other)
            user_api_key_dict: User API key authentication dict
            request_data: Request data containing model, logging_obj, etc.

        Returns:
            Async generator that yields SSE-formatted chat completion chunks
        """
        # If response is a BaseResponsesAPIStreamingIterator, transform it first
        if isinstance(response, BaseResponsesAPIStreamingIterator):
            # Transform Responses API iterator to chat completion iterator
            # Cast to AsyncIterator[str] since BaseResponsesAPIStreamingIterator implements __aiter__/__anext__
            completion_stream = (
                responses_api_bridge.transformation_handler.get_model_response_iterator(
                    streaming_response=cast(AsyncIterator[str], response),
                    sync_stream=False,
                    json_mode=False,
                )
            )
            # Wrap in CustomStreamWrapper to get the async generator
            logging_obj = request_data.get("litellm_logging_obj")
            streamwrapper = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=request_data.get("model", ""),
                custom_llm_provider=None,
                logging_obj=logging_obj,
            )
            # Use async_data_generator to format as SSE
            return async_data_generator(
                response=streamwrapper,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
            )
        # Otherwise, use the default generator
        return async_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=cursor_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        # Transform non-streaming Responses API response to chat completions format
        if isinstance(response, ResponsesAPIResponse):
            logging_obj = processor.data.get("litellm_logging_obj")
            transformed_response = (
                responses_api_bridge.transformation_handler.transform_response(
                    model=processor.data.get("model", ""),
                    raw_response=response,
                    model_response=ModelResponse(),
                    logging_obj=cast(Any, logging_obj),
                    request_data=processor.data,
                    messages=processor.data.get("input", []),
                    optional_params={},
                    litellm_params={},
                    encoding=None,
                    api_key=None,
                    json_mode=None,
                )
            )
            return transformed_response

        # Streaming responses are already transformed by cursor_select_data_generator
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def get_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/get
    
    ```bash
    # Get polling response
    curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123 \
    -H "Authorization: Bearer sk-1234"
    
    # Get provider response
    curl -X GET http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response
        if not redis_usage_cache:
            raise HTTPException(
                status_code=500,
                detail="Redis cache not configured. Polling requires Redis.",
            )

        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get current state from cache
        state = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Polling response {response_id} not found or expired",
            )

        # Return the whole state directly (OpenAI Response object format)
        # https://platform.openai.com/docs/api-reference/responses/object
        return state

    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.delete(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.delete(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def delete_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Deletes from Redis cache
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/delete
    
    ```bash
    curl -X DELETE http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response deletion
        if not redis_usage_cache:
            raise HTTPException(status_code=500, detail="Redis cache not configured.")

        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get state to verify access
        state = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(
                status_code=404, detail=f"Polling response {response_id} not found"
            )

        # Delete from cache
        success = await polling_handler.delete_polling(response_id)

        if success:
            return DeleteResponseResult(id=response_id, object="response", deleted=True)
        else:
            raise HTTPException(
                status_code=500, detail="Failed to delete polling response"
            )

    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/openai/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def get_response_input_items(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List input items for a response."""
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_input_items",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def compact_response(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Compact a response by running a compaction pass over a conversation.
    
    Returns encrypted, opaque items that can be used to reduce context size.
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/compact
    
    ```bash
    curl -X POST http://localhost:4000/v1/responses/compact \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}]
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acompact_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def cancel_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/cancel
    
    ```bash
    # Cancel polling response
    curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel \
    -H "Authorization: Bearer sk-1234"
    
    # Cancel provider response
    curl -X POST http://localhost:4000/v1/responses/resp_abc123/cancel \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response cancellation
        if not redis_usage_cache:
            raise HTTPException(status_code=500, detail="Redis cache not configured.")

        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get current state to verify it exists
        state = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(
                status_code=404, detail=f"Polling response {response_id} not found"
            )

        # Cancel the polling response (sets status to "cancelled")
        success = await polling_handler.cancel_polling(response_id)

        if success:
            # Fetch the updated state with cancelled status
            updated_state = await polling_handler.get_state(response_id)

            # Return the whole state directly (now with status="cancelled")
            return updated_state
        else:
            raise HTTPException(
                status_code=500, detail="Failed to cancel polling response"
            )

    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.websocket("/v1/responses")
@router.websocket("/responses")
async def responses_websocket_endpoint(
    websocket: WebSocket,
):
    """
    Responses API WebSocket mode endpoint.

    Keeps a persistent WebSocket connection for response.create events,
    enabling lower-latency agentic workflows with many tool-call round trips.

    See: https://developers.openai.com/api/docs/guides/websocket-mode/
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.route_llm_request import route_request

    # Accept the WebSocket handshake
    requested_protocols = [
        p.strip()
        for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",")
        if p.strip()
    ]
    accept_kwargs: dict = {}
    if requested_protocols:
        accept_kwargs["subprotocol"] = requested_protocols[0]
    await websocket.accept(**accept_kwargs)

    try:
        user_api_key_dict = await user_api_key_auth_websocket(websocket)
    except Exception as e:
        verbose_proxy_logger.exception("Responses WebSocket authentication error")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": str(e),
                        },
                    }
                )
            )
        except Exception:
            pass
        await websocket.close(code=1008, reason="Authentication failed")
        return

    initial_client_message: Optional[str] = None
    resolved_model = cast(Optional[str], websocket.query_params.get("model"))
    if resolved_model is None:
        try:
            initial_client_message = await websocket.receive_text()
            initial_payload = json.loads(initial_client_message)
            response_payload = initial_payload.get("response")
            if isinstance(response_payload, dict) and response_payload:
                resolved_model = cast(Optional[str], response_payload.get("model"))
            else:
                resolved_model = cast(Optional[str], initial_payload.get("model"))
        except json.JSONDecodeError:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Invalid JSON in first websocket message",
                        },
                    }
                )
            )
            await websocket.close(code=1003, reason="Invalid JSON")
            return
        except Exception:
            verbose_proxy_logger.exception(
                "Responses WebSocket failed to read initial client message"
            )
            await websocket.close(code=1011, reason="Failed to read initial message")
            return

        if resolved_model is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Model must be provided either in the websocket query string or the first response.create event",
                        },
                    }
                )
            )
            await websocket.close(code=1008, reason="Model required")
            return

    data: Dict[str, Any] = {"model": resolved_model, "websocket": websocket}

    # Construct a synthetic Request for pre-call processing
    headers_list = list(websocket.scope.get("headers") or [])
    scope: Dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": headers_list,
    }
    request = Request(scope=scope)
    request._url = websocket.url

    async def return_body():
        return json.dumps({"model": resolved_model}).encode()

    request.body = return_body  # type: ignore

    # Phase 1: pre-call processing (auth, guardrails, rate limits)
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            model=resolved_model,
            route_type="_aresponses_websocket",
        )
    except Exception as e:
        verbose_proxy_logger.exception("Responses WebSocket pre-call error")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "pre_call_error",
                            "message": str(e),
                        },
                    }
                )
            )
        except Exception:
            pass
        await websocket.close(code=1011, reason="Pre-call error")
        return

    # Phase 2: route to upstream provider
    try:
        data["user_api_key_dict"] = user_api_key_dict
        if initial_client_message is not None:
            data["initial_client_message"] = initial_client_message
        llm_call = await route_request(
            data=data,
            route_type="_aresponses_websocket",
            llm_router=llm_router,
            user_model=user_model,
        )
        await llm_call
    except Exception:
        verbose_proxy_logger.exception("Responses WebSocket error")
        await websocket.close(code=1011, reason="Internal server error")
