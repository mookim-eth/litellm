import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect, Request

from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.response_api_endpoints.endpoints import (
    CODEX_RESPONSES_LITE_HEADER,
    RESPONSES_SSE_KEEPALIVE,
    _await_with_request_disconnect,
    _deferred_responses_stream,
    _get_responses_keepalive_interval,
    _get_responses_provider_start_timeout,
    _is_codex_responses_lite_request,
    _should_enable_responses_keepalive,
    responses_api,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth


def test_only_codex_responses_lite_enables_early_keepalive():
    assert _is_codex_responses_lite_request(
        {
            "extra_headers": {
                "x-openai-internal-codex-responses-lite": "true"
            }
        }
    )
    assert not _is_codex_responses_lite_request({})


def test_provider_start_timeout_setting_is_validated():
    assert _get_responses_provider_start_timeout({}) == 300
    assert _get_responses_provider_start_timeout(
        {"responses_provider_start_timeout_seconds": "12.5"}
    ) == 12.5
    assert _get_responses_provider_start_timeout(
        {"responses_provider_start_timeout_seconds": 0}
    ) == 300

    settings = ConfigGeneralSettings(
        responses_provider_start_timeout_seconds=12.5
    )
    assert settings.responses_provider_start_timeout_seconds == 12.5


def test_regular_responses_keepalive_requires_explicit_opt_in():
    data = {"stream": True}

    assert not _should_enable_responses_keepalive(data, {})
    assert _should_enable_responses_keepalive(
        data, {"enable_responses_stream_keepalive": True}
    )
    assert _get_responses_keepalive_interval({}) == 45
    assert _get_responses_keepalive_interval(
        {"responses_stream_keepalive_interval_seconds": "12.5"}
    ) == 12.5

    settings = ConfigGeneralSettings(
        enable_responses_stream_keepalive=True,
        responses_stream_keepalive_interval_seconds=12.5,
    )
    assert settings.enable_responses_stream_keepalive is True
    assert settings.responses_stream_keepalive_interval_seconds == 12.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("codex_lite", "general_keepalive"),
    [(True, False), (False, True)],
)
async def test_keepalive_endpoint_runs_pre_call_before_deferred_provider(
    codex_lite, general_keepalive
):
    request = MagicMock(spec=Request)
    request.headers = (
        {CODEX_RESPONSES_LITE_HEADER: "true"} if codex_lite else {}
    )
    fastapi_response = Response()
    user_api_key_dict = UserAPIKeyAuth()
    request_data = {
        "model": "gpt-test",
        "input": "hello",
        "stream": True,
    }
    if codex_lite:
        request_data["extra_headers"] = {
            CODEX_RESPONSES_LITE_HEADER: "true"
        }
    logging_obj = MagicMock(litellm_call_id="call-test")
    provider_started = asyncio.Event()

    async def block_provider(**_kwargs):
        provider_started.set()
        await asyncio.Event().wait()

    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new=AsyncMock(return_value=request_data),
        ),
        patch(
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch(
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as processor_class,
        patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {
                "enable_responses_stream_keepalive": general_keepalive,
                "responses_stream_keepalive_interval_seconds": 0.01,
            },
        ),
    ):
        processor = processor_class.return_value
        processor.data = request_data
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=(request_data, logging_obj)
        )
        processor.base_process_llm_request = AsyncMock(side_effect=block_provider)
        processor.maybe_get_model_id.return_value = None
        processor_class.get_custom_headers.return_value = {
            "x-litellm-call-id": "call-test"
        }

        response = await responses_api(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )

        assert isinstance(response, StreamingResponse)
        assert processor.common_processing_pre_call_logic.await_count == 1
        assert response.headers["x-litellm-call-id"] == "call-test"

        if codex_lite:
            assert await response.body_iterator.__anext__() == RESPONSES_SSE_KEEPALIVE
            assert processor.base_process_llm_request.await_count == 0

        pending = asyncio.create_task(response.body_iterator.__anext__())
        await asyncio.wait_for(provider_started.wait(), 1)
        assert processor.base_process_llm_request.call_args.kwargs[
            "skip_pre_call_logic"
        ] is True
        if codex_lite:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert await asyncio.wait_for(pending, 1) == RESPONSES_SSE_KEEPALIVE
            await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_keepalive_is_sent_before_provider_call_starts():
    provider_started = False

    async def process_request():
        nonlocal provider_started
        provider_started = True
        await asyncio.Event().wait()

    stream = _deferred_responses_stream(process_request)
    started = time.monotonic()

    first = await asyncio.wait_for(stream.__anext__(), 0.1)

    assert first == RESPONSES_SSE_KEEPALIVE
    assert time.monotonic() - started < 0.05
    assert provider_started is False
    await stream.aclose()


@pytest.mark.asyncio
async def test_delayed_keepalive_forwards_fast_event_without_heartbeat():
    event = b'data: {"type":"response.completed","response":{}}\n\n'

    async def body():
        yield event

    async def process_request():
        return StreamingResponse(body(), media_type="text/event-stream")

    chunks = [
        chunk
        async for chunk in _deferred_responses_stream(
            process_request,
            keepalive_interval_seconds=0.01,
            send_initial_keepalive=False,
        )
    ]

    assert chunks == [event]


@pytest.mark.asyncio
async def test_delayed_keepalive_waits_for_first_silence_window():
    provider_started = asyncio.Event()

    async def process_request():
        provider_started.set()
        await asyncio.Event().wait()

    stream = _deferred_responses_stream(
        process_request,
        keepalive_interval_seconds=0.02,
        send_initial_keepalive=False,
    )
    started = time.monotonic()

    first = await asyncio.wait_for(stream.__anext__(), 0.1)

    assert provider_started.is_set()
    assert first == RESPONSES_SSE_KEEPALIVE
    assert time.monotonic() - started >= 0.02
    await stream.aclose()


@pytest.mark.asyncio
async def test_forwards_responses_events_without_changing_sequence_or_text():
    events = [
        b'data: {"type":"response.created","sequence_number":7,"response":{}}\n\n',
        b'data: {"type":"response.output_text.delta","sequence_number":8,"delta":"hello"}\n\n',
        b'data: {"type":"response.completed","sequence_number":9,"response":{"id":"resp-test","usage":null}}\n\n',
    ]

    async def body():
        for event in events:
            yield event

    async def process_request():
        return StreamingResponse(body(), media_type="text/event-stream")

    chunks = [chunk async for chunk in _deferred_responses_stream(process_request)]

    assert chunks == [RESPONSES_SSE_KEEPALIVE, *events]


@pytest.mark.asyncio
async def test_emits_keepalives_during_provider_start_and_stream_gaps():
    event = b'data: {"type":"response.completed","response":{}}\n\n'

    async def body():
        await asyncio.sleep(0.05)
        yield event

    async def process_request():
        await asyncio.sleep(0.05)
        return StreamingResponse(body(), media_type="text/event-stream")

    chunks = [
        chunk
        async for chunk in _deferred_responses_stream(
            process_request, keepalive_interval_seconds=0.01
        )
    ]

    assert chunks[-1] == event
    assert len(chunks[:-1]) >= 8
    assert all(chunk == RESPONSES_SSE_KEEPALIVE for chunk in chunks[:-1])


@pytest.mark.asyncio
async def test_provider_start_and_iteration_stay_in_one_producer_task():
    provider_task = None
    iterator_task = None

    async def body():
        nonlocal iterator_task
        iterator_task = asyncio.current_task()
        yield b'data: {"type":"response.completed","response":{}}\n\n'

    async def process_request():
        nonlocal provider_task
        provider_task = asyncio.current_task()
        return StreamingResponse(body(), media_type="text/event-stream")

    chunks = [chunk async for chunk in _deferred_responses_stream(process_request)]

    assert len(chunks) == 2
    assert provider_task is iterator_task


@pytest.mark.asyncio
async def test_provider_exception_after_keepalive_becomes_response_failed():
    async def process_request():
        raise RuntimeError("provider unavailable")

    chunks = [chunk async for chunk in _deferred_responses_stream(process_request)]
    payload = json.loads(chunks[1].split(b"data: ", 1)[1])

    assert chunks[0] == RESPONSES_SSE_KEEPALIVE
    assert payload["type"] == "response.failed"
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["error"]["message"] == "provider unavailable"


@pytest.mark.asyncio
async def test_provider_failure_runs_error_handler_before_response_failed():
    handled = []

    async def process_request():
        raise RuntimeError("provider unavailable")

    async def error_handler(error):
        handled.append(error)
        return SimpleNamespace(message="mapped failure", code=503)

    chunks = [
        chunk
        async for chunk in _deferred_responses_stream(
            process_request, error_handler=error_handler
        )
    ]
    payload = json.loads(chunks[1].split(b"data: ", 1)[1])

    assert len(handled) == 1
    assert payload["response"]["error"] == {
        "code": "503",
        "message": "mapped failure",
    }


@pytest.mark.asyncio
async def test_provider_start_timeout_cancels_call_and_emits_response_failed():
    provider_cancelled = asyncio.Event()

    async def process_request():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    chunks = [
        chunk
        async for chunk in _deferred_responses_stream(
            process_request, provider_start_timeout_seconds=0.01
        )
    ]
    payload = json.loads(chunks[1].split(b"data: ", 1)[1])

    assert provider_cancelled.is_set()
    assert payload["type"] == "response.failed"
    assert payload["response"]["error"]["code"] == "504"
    assert "within 0.01 seconds" in payload["response"]["error"]["message"]


@pytest.mark.asyncio
async def test_prefetched_json_error_after_keepalive_becomes_response_failed():
    async def process_request():
        return JSONResponse(
            {"error": {"message": "bad request", "code": "invalid_request"}},
            status_code=400,
        )

    chunks = [chunk async for chunk in _deferred_responses_stream(process_request)]
    payload = json.loads(chunks[1].split(b"data: ", 1)[1])

    assert payload["type"] == "response.failed"
    assert payload["response"]["error"] == {
        "code": "invalid_request",
        "message": "bad request",
    }


@pytest.mark.asyncio
async def test_disconnect_while_waiting_for_provider_headers_cancels_request():
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    disconnect = asyncio.Event()
    receive_calls = 0

    async def process_request():
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    async def endpoint(_scope, receive, send):
        response = StreamingResponse(
            _deferred_responses_stream(process_request),
            media_type="text/event-stream",
        )
        await response(_scope, receive, send)

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [],
        "asgi": {"version": "3.0", "spec_version": "2.3"},
    }
    InFlightRequestsMiddleware._in_flight = 0
    task = asyncio.create_task(
        InFlightRequestsMiddleware(endpoint)(scope, receive, send)
    )
    await asyncio.wait_for(provider_started.wait(), 1)
    disconnect.set()
    await asyncio.wait_for(task, 1)

    assert messages[0]["status"] == 200
    assert messages[1]["body"] == RESPONSES_SSE_KEEPALIVE
    assert provider_cancelled.is_set()
    assert InFlightRequestsMiddleware.get_count() == 0
    InFlightRequestsMiddleware._in_flight = 0


@pytest.mark.asyncio
async def test_inline_disconnect_watcher_preserves_provider_task_and_cancels_it():
    provider_task = None
    provider_cancelled = asyncio.Event()
    disconnect = asyncio.Event()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        },
        receive,
    )

    async def provider_call():
        nonlocal provider_task
        provider_task = asyncio.current_task()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    current_task = asyncio.current_task()
    operation = asyncio.create_task(
        _await_with_request_disconnect(request, provider_call(), 300)
    )
    await asyncio.sleep(0)
    disconnect.set()

    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(operation, 1)

    assert provider_task is operation
    assert provider_task is not current_task
    assert provider_cancelled.is_set()


@pytest.mark.asyncio
async def test_connected_long_stream_is_not_cancelled():
    provider_cancelled = False

    async def body():
        nonlocal provider_cancelled
        try:
            yield b'data: {"type":"response.created","response":{}}\n\n'
            await asyncio.sleep(0.15)
            yield b'data: {"type":"response.completed","response":{"id":"resp-long","usage":null}}\n\n'
        except asyncio.CancelledError:
            provider_cancelled = True
            raise

    async def process_request():
        await asyncio.sleep(0.05)
        return StreamingResponse(body(), media_type="text/event-stream")

    started = time.monotonic()
    chunks = [chunk async for chunk in _deferred_responses_stream(process_request)]

    assert chunks[0] == RESPONSES_SSE_KEEPALIVE
    assert b"response.completed" in chunks[-1]
    assert time.monotonic() - started >= 0.2
    assert provider_cancelled is False
