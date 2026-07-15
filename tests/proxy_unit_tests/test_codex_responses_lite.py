import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.response_api_endpoints.endpoints import (
    CODEX_CONCURRENCY_RETRY_DELAY_SECONDS,
    CODEX_RESPONSES_LITE_HEADER,
    _apply_codex_responses_lite_request_overrides,
    _should_return_codex_concurrency_retry,
    responses_api,
)


def test_codex_responses_lite_overrides_add_header_and_disables_parallel_tools():
    data = {"model": "gpt-5.6-sol", "parallel_tool_calls": True}
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert data["extra_headers"] == {CODEX_RESPONSES_LITE_HEADER: "true"}
    assert data["parallel_tool_calls"] is False


def test_forward_codex_responses_lite_header_preserves_existing_extra_headers():
    data = {
        "model": "gpt-5.6-sol",
        "extra_headers": {"x-existing": "value"},
    }
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert data["extra_headers"] == {
        "x-existing": "value",
        CODEX_RESPONSES_LITE_HEADER: "true",
    }
    assert data["parallel_tool_calls"] is False


def test_forward_codex_responses_lite_header_ignores_missing_header():
    data = {"model": "gpt-5.5"}
    request = SimpleNamespace(headers=Headers({}))

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert "extra_headers" not in data
    assert "parallel_tool_calls" not in data


def _request_with_codex_header(value: str = "true") -> Request:
    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(CODEX_RESPONSES_LITE_HEADER.encode(), value.encode())],
        }
    )


def _max_parallel_error() -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            "Rate limit exceeded for api_key. "
            "Limit type: max_parallel_requests."
        ),
        headers={"rate_limit_type": "max_parallel_requests"},
    )


def test_codex_streaming_max_parallel_error_is_selected_for_retry_response():
    assert (
        _should_return_codex_concurrency_retry(
            request=_request_with_codex_header(),
            data={"stream": True},
            error=_max_parallel_error(),
        )
        is True
    )


@pytest.mark.parametrize(
    ("header_value", "stream", "rate_limit_type"),
    [
        ("false", True, "max_parallel_requests"),
        ("true", False, "max_parallel_requests"),
        ("true", True, "requests_per_minute"),
    ],
)
def test_codex_retry_response_does_not_change_other_requests(
    header_value, stream, rate_limit_type
):
    error = HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. Limit type: {rate_limit_type}.",
        headers={"rate_limit_type": rate_limit_type},
    )

    assert (
        _should_return_codex_concurrency_retry(
            request=_request_with_codex_header(header_value),
            data={"stream": stream},
            error=error,
        )
        is False
    )


@pytest.mark.asyncio
async def test_responses_endpoint_returns_retryable_sse_after_recording_429():
    original_error = _max_parallel_error()
    mapped_error = ProxyException(
        message=str(original_error.detail),
        type="None",
        param="None",
        code=429,
        headers={"x-litellm-call-id": "call-1"},
    )
    request_data = {"model": "test-model", "stream": True}

    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new_callable=AsyncMock,
            return_value=request_data,
        ),
        patch(
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            new_callable=AsyncMock,
            side_effect=original_error,
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "_handle_llm_api_exception",
            new_callable=AsyncMock,
            side_effect=mapped_error,
        ) as handle_error,
    ):
        response = await responses_api(
            request=_request_with_codex_header(),
            fastapi_response=Response(),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert response.headers["x-litellm-call-id"] == "call-1"
    body = b"".join([chunk async for chunk in response.body_iterator])
    event_lines = body.decode().splitlines()
    assert event_lines[0] == "event: response.failed"
    payload = json.loads(event_lines[1].removeprefix("data: "))
    assert payload == {
        "type": "response.failed",
        "response": {
            "error": {
                "code": "rate_limit_exceeded",
                "message": (
                    "Concurrency limit reached. Please try again in "
                    f"{CODEX_CONCURRENCY_RETRY_DELAY_SECONDS}s."
                ),
            }
        },
    }
    handle_error.assert_awaited_once()
    assert handle_error.await_args.kwargs["e"] is original_error


@pytest.mark.asyncio
async def test_responses_endpoint_keeps_non_codex_max_parallel_error_as_429():
    original_error = _max_parallel_error()
    mapped_error = ProxyException(
        message=str(original_error.detail), type="None", param="None", code=429
    )
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
        }
    )

    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new_callable=AsyncMock,
            return_value={"model": "test-model", "stream": True},
        ),
        patch(
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            new_callable=AsyncMock,
            side_effect=original_error,
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "_handle_llm_api_exception",
            new_callable=AsyncMock,
            side_effect=mapped_error,
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await responses_api(
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            )

    assert exc_info.value.code == "429"
