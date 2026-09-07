"""SpendLogs counts retries across model groups, including Responses streams."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError
from litellm.proxy.spend_tracking.spend_tracking_utils import _get_spend_logs_metadata
from litellm.types.router import RetryPolicy


def make_router(fallbacks):
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "chatgpt/gpt-5.6-sol"},
            }
            for name in ("primary", "secondary", "third")
        ],
        fallbacks=fallbacks,
        num_retries=1,
        retry_policy=RetryPolicy(InternalServerErrorRetries=1),
    )


def rate_limit():
    return litellm.RateLimitError(
        message="Rate limit exceeded", llm_provider="chatgpt", model="test"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("all_fail", [False, True])
async def test_cumulative_retries_across_groups(metadata_key, nested, all_fail):
    fallbacks = (
        [{"primary": ["secondary"]}, {"secondary": ["third"]}]
        if nested
        else [{"primary": ["secondary", "third"]}]
    )
    router = make_router(fallbacks)
    metadata = {}
    calls = []

    async def provider_call(**kwargs):
        model = kwargs["model"]
        calls.append((model, kwargs[metadata_key]["attempted_retries"]))
        if model == "third":
            if all_fail:
                raise rate_limit()
            return litellm.ModelResponse(model=model)
        if sum(name == model for name, _ in calls) == 1:
            raise litellm.InternalServerError(
                message="temporary failure", llm_provider="chatgpt", model=model
            )
        raise rate_limit()

    with patch("litellm.router.asyncio.sleep", new_callable=AsyncMock):
        call = router.async_function_with_fallbacks(
            model="primary", original_function=provider_call,
            num_retries=1, **{metadata_key: metadata},
        )
        if all_fail:
            with pytest.raises(litellm.RateLimitError):
                await call
        else:
            assert (await call).model == "third"

    assert calls == [
        ("primary", 0), ("primary", 1), ("secondary", 2),
        ("secondary", 3), ("third", 4),
    ]
    assert metadata["attempted_retries"] == 4
    assert _get_spend_logs_metadata(metadata)["attempted_retries"] == 4
    # The retry policy is still per group; accounting must not change it.
    assert metadata["max_retries"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fallbacks", [[], [{"primary": ["primary"]}]])
async def test_no_fallback_attempt_does_not_increment(fallbacks):
    router = make_router(fallbacks)
    metadata = {}
    call = AsyncMock(side_effect=rate_limit())
    with pytest.raises(litellm.RateLimitError):
        await router.async_function_with_fallbacks(
            model="primary", original_function=call, num_retries=1,
            metadata=metadata,
        )
    call.assert_awaited_once()
    assert metadata["attempted_retries"] == 0


@pytest.mark.asyncio
async def test_fallback_depth_limit_does_not_increment():
    router = make_router([{"primary": ["secondary"]}, {"secondary": ["third"]}])
    metadata = {}
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs["model"])
        raise rate_limit()

    with pytest.raises(litellm.RateLimitError):
        await router.async_function_with_fallbacks(
            model="primary", original_function=provider_call, num_retries=1,
            metadata=metadata, max_fallbacks=1,
        )
    assert calls == ["primary", "secondary"]
    assert metadata["attempted_retries"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
async def test_fresh_request_resets_retry_count(metadata_key):
    router = make_router([])
    metadata = {"attempted_retries": 99}

    async def provider_call(**kwargs):
        assert kwargs[metadata_key]["attempted_retries"] == 0
        return litellm.ModelResponse(model="primary")

    await router.async_function_with_fallbacks(
        model="primary", original_function=provider_call, num_retries=1,
        **{metadata_key: metadata},
    )
    assert metadata["attempted_retries"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["overloaded", "timeout"])
@pytest.mark.parametrize("all_fail", [False, True])
async def test_responses_stream_fallbacks_accumulate(error_type, all_fail):
    router = make_router([{"primary": ["secondary"]}, {"secondary": ["third"]}])
    metadata = {}
    calls = []
    closed = []

    async def provider_call(**kwargs):
        model = kwargs["model"]
        calls.append((model, kwargs["litellm_metadata"]["attempted_retries"]))

        async def events():
            try:
                yield SimpleNamespace(type="response.in_progress")
                if model != "third" or all_fail:
                    if error_type == "overloaded":
                        error = rate_limit()
                        error.is_responses_stream_overload = True
                    else:
                        error = MidStreamFallbackError(
                            message="no effective output", model=model,
                            llm_provider="chatgpt",
                            original_exception=litellm.Timeout(
                                message="TTFT deadline", model=model,
                                llm_provider="chatgpt",
                            ),
                            is_pre_first_chunk=True,
                        )
                    raise error
                yield SimpleNamespace(type="response.output_text.delta", delta="ok")
            finally:
                closed.append(model)

        return router._aresponses_streaming_iterator(
            events(), {**kwargs, "original_function": provider_call}
        )

    response = await router.async_function_with_fallbacks(
        model="primary", original_function=provider_call, num_retries=1,
        litellm_metadata=metadata, stream=True,
    )
    assert metadata["attempted_retries"] == 0  # No stream iteration yet.
    if all_fail:
        with pytest.raises((litellm.RateLimitError, MidStreamFallbackError)):
            _ = [event async for event in response]
    else:
        events = [event async for event in response]
        assert events[-1].delta == "ok"
    assert calls == [("primary", 0), ("secondary", 1), ("third", 2)]
    assert metadata["attempted_retries"] == 2
    assert _get_spend_logs_metadata(metadata)["attempted_retries"] == 2
    assert closed == ["primary", "secondary", "third"]
