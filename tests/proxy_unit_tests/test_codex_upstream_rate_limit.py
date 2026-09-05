import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
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
        "response": {
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please try again in 10 seconds.",
            }
        },
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
async def test_should_preserve_mapped_headers_and_record_failure(handler_raises):
    original_error = _rate_limit()
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
        ("codex_cli_rs/0.144.4", True, _rate_limit("openai")),
        ("codex_cli_rs/0.144.4", True, _rate_limit(account_limit=True)),
        ("codex_cli_rs/0.144.4", True, HTTPException(429, "TPM limit reached")),
        (
            "codex_cli_rs/0.144.4",
            True,
            litellm.AuthenticationError(
                message="token_revoked", llm_provider="chatgpt", model="gpt-5.6-sol"
            ),
        ),
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
@pytest.mark.parametrize("handler_raises", [False, True])
@pytest.mark.parametrize("code", [403, 500])
async def test_should_not_hide_failure_hook_errors(code, handler_raises):
    processor = AsyncMock()
    mapped_error = ProxyException(
        message="hook error", type="error", param=None, code=code
    )
    if handler_raises:
        processor._handle_llm_api_exception.side_effect = mapped_error
    else:
        processor._handle_llm_api_exception.return_value = mapped_error
    with pytest.raises(ProxyException) as exc:
        await _handle_responses_api_exception(
            error=_rate_limit(),
            request=_request(),
            data={"stream": True},
            processor=processor,
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_logging_obj=None,
            version=None,
        )
    assert exc.value is mapped_error
