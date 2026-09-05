from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.types.router import RetryPolicy


def _router(fallbacks=None, retry_policy=None):
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "chatgpt/gpt-5.6-luna"},
            }
            for name in ("primary", "secondary", "third")
        ],
        fallbacks=fallbacks,
        num_retries=2,
        retry_policy=retry_policy,
    )


def _rate_limit(provider="chatgpt"):
    return litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider=provider,
        model="gpt-5.6-luna",
        response=httpx.Response(429, headers={"retry-after": "30"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_scope", ["none", "router", "request"])
@pytest.mark.parametrize("stream", [False, True])
async def test_should_fallback_without_retry_after_or_same_account_retry(
    policy_scope, stream, caplog
):
    router = _router(
        fallbacks=[{"primary": ["secondary"]}],
        retry_policy=RetryPolicy(RateLimitErrorRetries=4)
        if policy_scope == "router" else None,
    )
    calls = []
    result = litellm.ModelResponse(model="secondary")

    async def provider_call(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise _rate_limit()
        return result

    request = {
        "model": "primary",
        "original_function": provider_call,
        "num_retries": 2,
        "stream": stream,
        "litellm_call_id": "chatgpt-fallback-test",
        "metadata": {},
    }
    if policy_scope == "request":
        request["model_group_retry_policy"] = {
            "primary": RetryPolicy(RateLimitErrorRetries=4)
        }
    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock) as sleep:
        response = await router.async_function_with_fallbacks(**request)

    assert response is result
    assert calls == ["primary", "secondary"]
    sleep.assert_not_awaited()
    assert response._hidden_params["additional_headers"][
        "x-litellm-attempted-fallbacks"
    ] == 1
    assert "litellm_rate_limit_fallback" in caplog.text
    assert "request_id=chatgpt-fallback-test" in caplog.text
    assert "action=skip_same_group_retry" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("all_fail", [False, True])
async def test_should_try_each_fallback_group_once_without_sleep(all_fail):
    router = _router(fallbacks=[{"primary": ["secondary", "third"]}])
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs["model"])
        if all_fail or kwargs["model"] != "third":
            error = _rate_limit()
            raise error
        return litellm.ModelResponse(model="third")

    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock) as sleep:
        call = router.async_function_with_fallbacks(
            model="primary", original_function=provider_call,
            num_retries=2, metadata={},
        )
        if all_fail:
            with pytest.raises(litellm.RateLimitError):
                await call
        else:
            assert (await call).model == "third"
    assert calls == ["primary", "secondary", "third"]
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallbacks", [None, [{"unrelated": ["secondary"]}]])
async def test_should_fail_without_waiting_when_no_fallback_matches(fallbacks):
    router = _router(fallbacks=fallbacks)
    error = _rate_limit()
    call = AsyncMock(side_effect=error)
    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(litellm.RateLimitError) as exc:
            await router.async_function_with_fallbacks(
                model="primary", original_function=call,
                num_retries=2, metadata={},
            )
    assert exc.value is error
    call.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_policy", [False, True])
async def test_should_stop_retry_loop_when_later_attempt_hits_chatgpt_429(with_policy):
    router = _router(
        fallbacks=[{"primary": ["secondary"]}],
        retry_policy=RetryPolicy(InternalServerErrorRetries=3, RateLimitErrorRetries=4)
        if with_policy else None,
    )
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise litellm.InternalServerError(
                message="temporary error", llm_provider="chatgpt", model="test"
            )
        if kwargs["model"] == "primary":
            raise _rate_limit()
        return litellm.ModelResponse(model="secondary")

    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock) as sleep:
        response = await router.async_function_with_fallbacks(
            model="primary", original_function=provider_call,
            num_retries=3, metadata={},
        )
    assert response.model == "secondary"
    assert calls == ["primary", "primary", "secondary"]
    # Only the initial 500 used ordinary retry backoff; the 429 did not.
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_preserve_other_providers_retry_after():
    router = _router(fallbacks=[{"primary": ["secondary"]}])
    result = litellm.ModelResponse(model="primary")
    call = AsyncMock(side_effect=[_rate_limit(provider="openai"), result])
    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock) as sleep:
        response = await router.async_function_with_fallbacks(
            model="primary", original_function=call,
            num_retries=2, metadata={},
        )
    assert response is result
    assert [c.kwargs["model"] for c in call.await_args_list] == ["primary", "primary"]
    sleep.assert_awaited_once()
    assert 30 <= sleep.await_args.args[0] <= 31
