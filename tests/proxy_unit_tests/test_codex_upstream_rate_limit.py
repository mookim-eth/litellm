import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_response,
)
from litellm.proxy.response_api_endpoints.endpoints import (
    _handle_responses_api_exception,
    responses_api,
)


def _request(user_agent="codex_cli_rs/0.144.4"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"user-agent", user_agent.encode())],
        }
    )


def _rate_limit(provider="chatgpt", account_limit=False):
    error = litellm.RateLimitError(
        message='{"detail":"Rate limit exceeded"}',
        llm_provider=provider,
        model="gpt-5.6-sol",
        response=httpx.Response(429, headers={"retry-after": "30"}),
    )
    if account_limit:
        error.is_provider_account_concurrency_limit = True
    return error


async def _assert_retry_event(response):
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert response.headers["x-litellm-call-id"] == "upstream-429-test"
    body = b"".join([chunk async for chunk in response.body_iterator]).decode()
    assert body.startswith("event: response.failed\n")
    event = json.loads(body.split("data: ", 1)[1])
    assert event == {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please try again in 10 seconds.",
            }
        },
    }


async def _assert_error_event(response, mapped_error):
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    body = b"".join([chunk async for chunk in response.body_iterator]).decode()
    assert body.startswith("event: error\n")
    assert "[DONE]" not in body
    event = json.loads(body.split("data: ", 1)[1])
    assert event == {
        "type": "error",
        "code": str(mapped_error.code),
        "message": mapped_error.message,
        "param": None,
        "sequence_number": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_state", ["absent", "exhausted", "success"])
async def test_should_map_only_final_router_429_to_codex_retry(fallback_state):
    router = litellm.Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "chatgpt/gpt-5.6-sol"}}
            for name in ("primary", "secondary")
        ],
        fallbacks=None if fallback_state == "absent" else [{"primary": ["secondary"]}],
        num_retries=2,
    )
    calls = []
    success = Response(content="successful fallback")
    original_error = _rate_limit()
    mapped_error = ProxyException(
        message=str(original_error),
        type="rate_limit_error",
        param=None,
        code=429,
        headers={"x-litellm-call-id": "upstream-429-test"},
    )

    async def provider_call(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "secondary" and fallback_state == "success":
            return success
        raise original_error if kwargs["model"] == "primary" else _rate_limit()

    async def process(**kwargs):
        return await router.async_function_with_fallbacks(
            model="primary",
            original_function=provider_call,
            num_retries=2,
            stream=True,
            metadata={},
        )

    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            AsyncMock(return_value={"model": "primary", "stream": True}),
        ),
        patch(
            "litellm.proxy.response_polling.polling_handler.should_use_polling_for_request",
            return_value=False,
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            AsyncMock(side_effect=process),
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "_handle_llm_api_exception",
            AsyncMock(side_effect=mapped_error),
        ) as handle_error,
        patch("litellm.router.asyncio.sleep", AsyncMock()) as sleep,
    ):
        response = await responses_api(
            request=_request(),
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(),
        )

    assert calls == (
        ["primary"] if fallback_state == "absent" else ["primary", "secondary"]
    )
    sleep.assert_not_awaited()
    if fallback_state == "success":
        assert response is success
        handle_error.assert_not_awaited()
    else:
        await _assert_retry_event(response)
        handle_error.assert_awaited_once()
        assert handle_error.await_args.kwargs["e"] is original_error


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_raises", [False, True])
@pytest.mark.parametrize("account_limit", [False, True])
async def test_should_preserve_mapped_headers_and_record_failure(
    handler_raises, account_limit
):
    original_error = _rate_limit(account_limit=account_limit)
    mapped_error = ProxyException(
        message=str(original_error),
        type="rate_limit_error",
        param=None,
        code=429,
        headers={
            "x-litellm-call-id": "upstream-429-test",
            "content-length": "123",
            "content-type": "application/json",
        },
    )
    processor = AsyncMock()
    if handler_raises:
        processor._handle_llm_api_exception.side_effect = mapped_error
    else:
        processor._handle_llm_api_exception.return_value = mapped_error
    response = await _handle_responses_api_exception(
        error=original_error,
        request=_request("Codex Desktop/1.0"),
        data={"stream": True},
        processor=processor,
        user_api_key_dict=UserAPIKeyAuth(),
        proxy_logging_obj=None,
        version=None,
    )
    await _assert_retry_event(response)
    assert "content-length" not in response.headers
    assert response.headers["content-type"].startswith("text/event-stream")
    processor._handle_llm_api_exception.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_agent,stream,error",
    [
        ("openai-python/2.30.0", True, _rate_limit()),
        ("", True, _rate_limit()),
        ("codex_cli_rs/0.144.4", False, _rate_limit()),
    ],
)
async def test_should_preserve_unrelated_error_responses(user_agent, stream, error):
    processor = AsyncMock()
    mapped_error = ProxyException(
        message=str(error),
        type="error",
        param=None,
        code=getattr(error, "status_code", 500),
    )
    processor._handle_llm_api_exception.side_effect = mapped_error
    with pytest.raises(ProxyException) as exc:
        await _handle_responses_api_exception(
            error=error,
            request=_request(user_agent),
            data={"stream": stream},
            processor=processor,
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_logging_obj=None,
            version=None,
        )
    assert exc.value is mapped_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _rate_limit("openai"),
        HTTPException(429, "TPM limit reached"),
        litellm.AuthenticationError(
            message="token_revoked", llm_provider="chatgpt", model="gpt-5.6-sol"
        ),
    ],
)
async def test_should_return_other_codex_stream_failures_as_error_events(error):
    processor = AsyncMock()
    mapped_error = ProxyException(
        message=str(error),
        type="error",
        param=None,
        code=getattr(error, "status_code", 500),
    )
    processor._handle_llm_api_exception.side_effect = mapped_error

    response = await _handle_responses_api_exception(
        error=error,
        request=_request(),
        data={"stream": True},
        processor=processor,
        user_api_key_dict=UserAPIKeyAuth(),
        proxy_logging_obj=None,
        version=None,
    )

    await _assert_error_event(response, mapped_error)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_raises", [False, True])
@pytest.mark.parametrize("code", [403, 500])
async def test_should_expose_failure_hook_errors_in_codex_stream(code, handler_raises):
    processor = AsyncMock()
    mapped_error = ProxyException(
        message="hook error", type="error", param=None, code=code
    )
    if handler_raises:
        processor._handle_llm_api_exception.side_effect = mapped_error
    else:
        processor._handle_llm_api_exception.return_value = mapped_error
    response = await _handle_responses_api_exception(
        error=_rate_limit(),
        request=_request(),
        data={"stream": True},
        processor=processor,
        user_api_key_dict=UserAPIKeyAuth(),
        proxy_logging_obj=None,
        version=None,
    )
    await _assert_error_event(response, mapped_error)


def _responses_ttft_timeout():
    timeout = litellm.Timeout(
        message="Timed out waiting for the first effective Responses output",
        model="test-deployment",
        llm_provider="chatgpt",
    )
    error = MidStreamFallbackError(
        message=str(timeout),
        model=timeout.model,
        llm_provider=timeout.llm_provider,
        original_exception=timeout,
        is_pre_first_chunk=True,
    )
    error.is_responses_ttft_timeout = True
    return error


async def _ttft_http_response(
    *, user_agent: str, stream: bool, marked: bool = True, status_code: int = 408
):
    from litellm.proxy.proxy_server import async_data_generator

    error = _responses_ttft_timeout()
    error.status_code = status_code
    if not marked:
        del error.is_responses_ttft_timeout

    async def raise_ttft(*args, **kwargs):
        raise error
        yield  # pragma: no cover

    proxy_logging = AsyncMock()
    proxy_logging.async_post_call_streaming_iterator_hook = raise_ttft
    provider_response = AsyncMock()
    request_data = {
        "model": "gpt-5.6-sol",
        "stream": stream,
        "proxy_server_request": {"headers": {"user-agent": user_agent}},
    }
    with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging):
        response = await create_response(
            async_data_generator(
                provider_response,
                UserAPIKeyAuth(),
                request_data,
            ),
            "text/event-stream",
            {},
        )
    return response, error, proxy_logging


@pytest.mark.asyncio
async def test_should_return_codex_ttft_408_as_retryable_http_200_after_logging():
    response, original_error, proxy_logging = await _ttft_http_response(
        user_agent="codex_cli_rs/0.144.4",
        stream=True,
    )

    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(
        chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks
    )
    event = json.loads(body.removeprefix("data: "))
    assert event == {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Request timed out. Please try again in 10 seconds.",
            }
        },
    }
    assert original_error.status_code == 408
    proxy_logging.post_call_failure_hook.assert_awaited_once()
    assert (
        proxy_logging.post_call_failure_hook.await_args.kwargs["original_exception"]
        is original_error
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_agent,stream,marked,status_code",
    [
        ("openai-python/2.30.0", True, True, 408),
        ("codex_cli_rs/0.144.4", False, True, 408),
        ("codex_cli_rs/0.144.4", True, False, 408),
        ("codex_cli_rs/0.144.4", True, True, 503),
    ],
)
async def test_should_preserve_non_matching_ttft_error_responses(
    user_agent, stream, marked, status_code
):
    response, original_error, proxy_logging = await _ttft_http_response(
        user_agent=user_agent,
        stream=stream,
        marked=marked,
        status_code=status_code,
    )

    assert response.status_code == status_code
    assert json.loads(response.body)["error"]["code"] == str(status_code)
    assert original_error.status_code == status_code
    proxy_logging.post_call_failure_hook.assert_awaited_once()
    assert (
        proxy_logging.post_call_failure_hook.await_args.kwargs["original_exception"]
        is original_error
    )
