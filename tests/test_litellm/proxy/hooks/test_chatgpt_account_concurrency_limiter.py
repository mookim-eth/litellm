import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.hooks.chatgpt_account_concurrency_limiter import (
    ChatGPTAccountConcurrencyLimiter,
)
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)


class _FakeInternalUsageCache:
    def __init__(self) -> None:
        self.dual_cache = SimpleNamespace(redis_cache=None)


def _write_auth_file(path: Path, account_id: str, plan_type: str) -> None:
    path.write_text(
        json.dumps({"account_id": account_id, "plan_type": plan_type}),
        encoding="utf-8",
    )


def _logging_obj() -> Logging:
    return Logging(
        model="chatgpt/gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="acompletion",
        start_time=None,
        litellm_call_id="test-call",
        function_id="test-function",
    )


@pytest.fixture
def configured_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(
        proxy_server.general_settings,
        "chatgpt_plan_max_parallel_requests",
        {"plus": 3, "k12": 3, "team": 5, "pro": 10, "prolite": 7},
    )


@pytest.mark.asyncio
async def test_should_share_limit_across_models_for_one_account(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, "account-a", "plus")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_objects = []

    for model in ("chatgpt/gpt-5.4", "chatgpt/gpt-5.3-codex", "chatgpt/gpt-5.5"):
        logging_obj = _logging_obj()
        logging_objects.append(logging_obj)
        await limiter.async_pre_call_deployment_hook(
            {
                "model": model,
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": logging_obj,
            },
            None,
        )

    snapshot = await limiter.get_concurrency_snapshot()
    assert snapshot["storage"] == "local"
    assert snapshot["total_active"] == 3
    assert snapshot["accounts"] == [
        {
            "account_hash_prefix": limiter._account_key("account-a").split(":")[-1][
                :12
            ],
            "plan_type": "plus",
            "active": 3,
            "limit": 3,
            "remaining": 0,
        }
    ]

    with pytest.raises(litellm.RateLimitError) as exc_info:
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4-mini",
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": _logging_obj(),
            },
            None,
        )

    assert exc_info.value.num_retries == 0
    assert getattr(exc_info.value, "skip_deployment_cooldown") is True
    assert "Please retry in 10 seconds." in str(exc_info.value)
    assert exc_info.value.response.headers["retry-after"] == "10"

    await logging_objects[0].async_cleanup_deployment_resources()
    replacement_logging_obj = _logging_obj()
    await limiter.async_pre_call_deployment_hook(
        {
            "model": "chatgpt/gpt-5.4-mini",
            "chatgpt_auth_file_path": str(auth_file),
            "litellm_logging_obj": replacement_logging_obj,
        },
        None,
    )

    for logging_obj in logging_objects[1:]:
        await logging_obj.async_cleanup_deployment_resources()
    await replacement_logging_obj.async_cleanup_deployment_resources()
    assert (await limiter.get_concurrency_snapshot())["accounts"] == []


@pytest.mark.asyncio
async def test_should_keep_separate_limits_for_different_accounts(
    tmp_path: Path, configured_limits: None
) -> None:
    first_auth = tmp_path / "first.json"
    second_auth = tmp_path / "second.json"
    _write_auth_file(first_auth, "account-a", "plus")
    _write_auth_file(second_auth, "account-b", "plus")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_objects = []

    for auth_file in (first_auth, second_auth):
        for _ in range(3):
            logging_obj = _logging_obj()
            logging_objects.append(logging_obj)
            await limiter.async_pre_call_deployment_hook(
                {
                    "model": "chatgpt/gpt-5.4",
                    "chatgpt_auth_file_path": str(auth_file),
                    "litellm_logging_obj": logging_obj,
                },
                None,
            )

    for logging_obj in logging_objects:
        await logging_obj.async_cleanup_deployment_resources()


@pytest.mark.asyncio
async def test_should_acquire_only_once_for_nested_wrappers(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, "account-a", "plus")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    first_logging_obj = _logging_obj()
    kwargs = {
        "model": "chatgpt/gpt-5.4",
        "chatgpt_auth_file_path": str(auth_file),
        "litellm_logging_obj": first_logging_obj,
    }

    for _ in range(3):
        await limiter.async_pre_call_deployment_hook(kwargs, None)

    additional_logging_objects = []
    for _ in range(2):
        logging_obj = _logging_obj()
        additional_logging_objects.append(logging_obj)
        await limiter.async_pre_call_deployment_hook(
            {**kwargs, "litellm_logging_obj": logging_obj}, None
        )

    with pytest.raises(litellm.RateLimitError):
        await limiter.async_pre_call_deployment_hook(
            {**kwargs, "litellm_logging_obj": _logging_obj()}, None
        )

    await first_logging_obj.async_cleanup_deployment_resources()
    for logging_obj in additional_logging_objects:
        await logging_obj.async_cleanup_deployment_resources()


@pytest.mark.asyncio
async def test_should_allow_ten_concurrent_requests_for_pro_plan(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "pro.json"
    _write_auth_file(auth_file, "pro-account", "pro")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_objects = []

    for _ in range(10):
        logging_obj = _logging_obj()
        logging_objects.append(logging_obj)
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": logging_obj,
            },
            None,
        )

    with pytest.raises(litellm.RateLimitError):
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": _logging_obj(),
            },
            None,
        )

    await asyncio.gather(
        *(obj.async_cleanup_deployment_resources() for obj in logging_objects)
    )


@pytest.mark.asyncio
async def test_should_allow_seven_concurrent_requests_for_prolite_plan(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "prolite.json"
    _write_auth_file(auth_file, "prolite-account", "prolite")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_objects = []

    for _ in range(7):
        logging_obj = _logging_obj()
        logging_objects.append(logging_obj)
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": logging_obj,
            },
            None,
        )

    with pytest.raises(litellm.RateLimitError):
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-5.4",
                "chatgpt_auth_file_path": str(auth_file),
                "litellm_logging_obj": _logging_obj(),
            },
            None,
        )

    await asyncio.gather(
        *(obj.async_cleanup_deployment_resources() for obj in logging_objects)
    )


@pytest.mark.asyncio
async def test_should_run_deployment_cleanup_only_once() -> None:
    logging_obj = _logging_obj()
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1

    logging_obj.add_async_deployment_cleanup_callback(cleanup)
    await asyncio.gather(
        logging_obj.async_cleanup_deployment_resources(),
        logging_obj.async_cleanup_deployment_resources(),
    )

    assert calls == 1


@pytest.mark.asyncio
async def test_should_scope_cleanup_to_the_streaming_attempt() -> None:
    logging_obj = _logging_obj()
    released = []

    async def release_first() -> None:
        released.append("first")

    async def release_second() -> None:
        released.append("second")

    logging_obj.add_async_deployment_cleanup_callback(release_first)
    logging_obj.add_async_deployment_cleanup_callback(release_second)

    await logging_obj.async_failure_handler(
        RuntimeError("attempt failed"),
        "traceback",
        deployment_cleanup_callbacks=[release_first],
    )

    assert released == ["first"]
    # Closing/logging the same attempt again must not invoke its callback twice.
    await asyncio.gather(
        logging_obj.async_cleanup_deployment_resources(callbacks=[release_first]),
        logging_obj.async_cleanup_deployment_resources(callbacks=[release_first]),
    )
    assert released == ["first"]
    await logging_obj.async_cleanup_deployment_resources()
    assert released == ["first", "second"]


def _attempt_stream(logging_obj: Logging, stream_kind: str):
    if stream_kind.startswith("responses"):
        iterator = (
            SyncResponsesAPIStreamingIterator
            if stream_kind == "responses_sync"
            else ResponsesAPIStreamingIterator
        )
        return iterator(
            response=httpx.Response(200),
            model="chatgpt/gpt-5.4",
            responses_api_provider_config=MagicMock(),
            logging_obj=logging_obj,
        )
    return CustomStreamWrapper(
        completion_stream=iter([]),
        model="chatgpt/gpt-5.4",
        custom_llm_provider="chatgpt",
        logging_obj=logging_obj,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["responses", "responses_sync", "chat"])
@pytest.mark.parametrize("next_account", ["account-a", "account-b"])
@pytest.mark.parametrize("outcome", ["failure", "success", "timeout"])
async def test_should_keep_next_attempt_lease_during_delayed_stream_logging(  # noqa: PLR0915
    tmp_path: Path,
    configured_limits: None,
    stream_kind: str,
    next_account: str,
    outcome: str,
) -> None:
    """Replay old logging after fallback/retry has acquired its local lease."""
    auth_file = tmp_path / "first.json"
    next_auth_file = tmp_path / "next.json"
    _write_auth_file(auth_file, "account-a", "pro")
    _write_auth_file(next_auth_file, next_account, "pro")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_obj = _logging_obj()
    # Cleanup must still execute when logging's other callbacks are deduplicated.
    logging_obj.has_run_logging(event_type="async_success")
    logging_obj.has_run_logging(event_type="async_failure")
    logging_obj.handle_sync_failure_callbacks_for_async_calls = MagicMock()
    logging_obj.handle_sync_success_callbacks_for_async_calls = MagicMock()
    logging_obj.failure_handler = MagicMock()
    logging_obj.success_handler = MagicMock()
    kwargs = {
        "model": "chatgpt/gpt-5.4",
        "chatgpt_auth_file_path": str(auth_file),
        "litellm_logging_obj": logging_obj,
    }
    await limiter.async_pre_call_deployment_hook(kwargs, None)
    first_stream = _attempt_stream(logging_obj, stream_kind)
    queued = []
    error = (
        httpx.ReadTimeout("no first event")
        if outcome == "timeout"
        else litellm.RateLimitError(
            message="server_is_overloaded", llm_provider="chatgpt", model="gpt-5.4"
        )
    )

    if stream_kind.startswith("responses"):
        with patch(
            "litellm.responses.streaming_iterator.GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue",
            side_effect=lambda async_coroutine: queued.append(async_coroutine),
        ), patch(
            "litellm.responses.streaming_iterator.run_async_function",
            side_effect=lambda async_function, **kw: queued.append(async_function(**kw)),
        ):
            if outcome == "success":
                first_stream._handle_logging_completed_response()
            else:
                first_stream._handle_failure(error)
    else:
        async def failed_stream():
            raise error
            yield  # pragma: no cover

        if outcome == "success":
            first_stream.sent_last_chunk = True
        else:
            first_stream.completion_stream = failed_stream()
        with patch(
            "litellm.litellm_core_utils.streaming_handler.asyncio.create_task",
            side_effect=queued.append,
        ):
            expected = (
                StopAsyncIteration if outcome == "success"
                else httpx.ReadTimeout if outcome == "timeout"
                else MidStreamFallbackError
            )
            with pytest.raises(expected):
                await first_stream.__anext__()

    assert len(queued) == 1
    await first_stream.aclose()
    assert (await limiter.get_concurrency_snapshot())["total_active"] == 0

    # Router uses the same Logging instance for the new account/attempt.
    kwargs["chatgpt_auth_file_path"] = str(next_auth_file)
    await limiter.async_pre_call_deployment_hook(kwargs, None)
    next_stream = _attempt_stream(logging_obj, stream_kind)
    before = await limiter.get_concurrency_snapshot()
    assert before["storage"] == "local"
    assert before["total_active"] == 1
    try:
        await queued[0]
        await first_stream.aclose()
        assert await limiter.get_concurrency_snapshot() == before
        # Old cleanup must not clear same-account dedup state, either.
        await limiter.async_pre_call_deployment_hook(kwargs, None)
        assert await limiter.get_concurrency_snapshot() == before
    finally:
        await next_stream.aclose()
    assert (await limiter.get_concurrency_snapshot())["total_active"] == 0
    assert not logging_obj._async_deployment_cleanup_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["responses", "chat"])
async def test_should_not_clean_future_lease_from_stream_without_lease(
    stream_kind: str,
) -> None:
    logging_obj = _logging_obj()
    first_stream = _attempt_stream(logging_obj, stream_kind)
    released = MagicMock()
    logging_obj.add_async_deployment_cleanup_callback(released)
    next_stream = _attempt_stream(logging_obj, stream_kind)
    await first_stream.aclose()
    released.assert_not_called()
    await next_stream.aclose()
    released.assert_called_once_with()


@pytest.mark.asyncio
async def test_should_release_resources_before_success_callback_deduplication() -> None:
    logging_obj = _logging_obj()
    cleanup_called = asyncio.Event()

    async def cleanup() -> None:
        cleanup_called.set()

    logging_obj.add_async_deployment_cleanup_callback(cleanup)
    logging_obj.has_run_logging(event_type="async_success")
    await logging_obj.async_success_handler(result=None)

    assert cleanup_called.is_set()


@pytest.mark.asyncio
async def test_should_release_resources_when_provider_call_is_cancelled() -> None:
    logging_obj = _logging_obj()
    cleanup_called = asyncio.Event()

    async def cleanup() -> None:
        cleanup_called.set()

    logging_obj.add_async_deployment_cleanup_callback(cleanup)
    call = asyncio.create_task(
        litellm.acompletion(
            model="openai/mock",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="slow",
            mock_delay=30,
            litellm_logging_obj=logging_obj,
        )
    )
    await asyncio.sleep(0.05)
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call

    assert cleanup_called.is_set()


@pytest.mark.asyncio
async def test_should_release_lease_when_messages_stream_is_cancelled(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, "account-a", "plus")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_obj = _logging_obj()
    await limiter.async_pre_call_deployment_hook(
        {
            "model": "chatgpt/gpt-5.4",
            "chatgpt_auth_file_path": str(auth_file),
            "litellm_logging_obj": logging_obj,
        },
        None,
    )

    class _MessagesStream:
        def __init__(self) -> None:
            self.closed = False
            self.sent_chunk = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent_chunk:
                self.sent_chunk = True
                return {"content": "hello"}
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            self.closed = True
            await logging_obj.async_cleanup_deployment_resources()

    class _ProxyLogging:
        async def async_post_call_streaming_iterator_hook(self, response, **kwargs):
            async for chunk in response:
                yield chunk

        async def async_post_call_streaming_hook(self, response, **kwargs):
            return response

    stream = _MessagesStream()
    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=stream,
        user_api_key_dict=UserAPIKeyAuth(),
        request_data={"model": "chatgpt/gpt-5.4"},
        proxy_logging_obj=_ProxyLogging(),  # type: ignore[arg-type]
        serialize_chunk=lambda chunk: str(chunk),
        serialize_error=lambda error: str(error),
    )

    await generator.__anext__()
    pending_chunk = asyncio.create_task(generator.__anext__())
    await asyncio.sleep(0)
    pending_chunk.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_chunk

    assert stream.closed is True
    assert limiter._local_leases == {}


@pytest.mark.asyncio
async def test_should_fallback_without_retrying_a_saturated_account(
    tmp_path: Path,
    configured_limits: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plus_auth = tmp_path / "plus.json"
    pro_auth = tmp_path / "pro.json"
    _write_auth_file(plus_auth, "plus-account", "plus")
    _write_auth_file(pro_auth, "pro-account", "pro")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    held_logging_objects = []

    for _ in range(3):
        logging_obj = _logging_obj()
        held_logging_objects.append(logging_obj)
        await limiter.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/mock-primary",
                "chatgpt_auth_file_path": str(plus_auth),
                "litellm_logging_obj": logging_obj,
            },
            None,
        )

    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "chatgpt/mock-primary",
                    "chatgpt_auth_file_path": str(plus_auth),
                    "mock_response": "primary",
                },
            },
            {
                "model_name": "secondary",
                "litellm_params": {
                    "model": "chatgpt/mock-secondary",
                    "chatgpt_auth_file_path": str(pro_auth),
                    "mock_response": "secondary",
                },
            },
        ],
        fallbacks=[{"primary": ["secondary"]}],
        num_retries=2,
    )

    request_logging_obj = _logging_obj()
    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hello"}],
        litellm_logging_obj=request_logging_obj,
    )

    assert response.choices[0].message.content == "secondary"
    assert response._hidden_params["additional_headers"][
        "x-litellm-attempted-retries"
    ] == 0
    await request_logging_obj.async_cleanup_deployment_resources()
    for logging_obj in held_logging_objects:
        await logging_obj.async_cleanup_deployment_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_should_release_http_429_account_before_immediate_fallback(
    tmp_path: Path, configured_limits: None, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    first_auth = tmp_path / "first.json"
    second_auth = tmp_path / "second.json"
    _write_auth_file(first_auth, "first-account", "pro")
    _write_auth_file(second_auth, "second-account", "pro")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    acquired_second = asyncio.Event()
    finish_second = asyncio.Event()
    original_hook = limiter.async_pre_call_deployment_hook
    attempts = []

    async def observe_hook(kwargs, call_type):
        await original_hook(kwargs, call_type)
        path = kwargs.get("chatgpt_auth_file_path")
        attempts.append(path)
        if path == str(second_auth):
            acquired_second.set()
            await finish_second.wait()

    monkeypatch.setattr(limiter, "async_pre_call_deployment_hook", observe_hook)
    router = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "chatgpt/mock",
                    "chatgpt_auth_file_path": str(auth),
                    "mock_response": mock_response,
                },
            }
            for name, auth, mock_response in [
                ("primary", first_auth, "litellm.RateLimitError"),
                ("secondary", second_auth, "secondary response"),
            ]
        ],
        fallbacks=[{"primary": ["secondary"]}],
        num_retries=2,
    )
    logging_obj = _logging_obj()
    with patch.object(
        router, "_time_to_sleep_before_retry",
        side_effect=AssertionError("ChatGPT 429 must not calculate backoff"),
    ):
        call = asyncio.create_task(router.acompletion(
            model="primary", messages=[{"role": "user", "content": "hello"}],
            stream=stream, litellm_logging_obj=logging_obj,
        ))
        try:
            await asyncio.wait_for(acquired_second.wait(), timeout=5)
            snapshot = await limiter.get_concurrency_snapshot()
            assert snapshot["storage"] == "local"
            assert snapshot["total_active"] == 1
            assert snapshot["accounts"][0]["account_hash_prefix"] == (
                limiter._account_key("second-account").split(":")[-1][:12]
            )
            assert attempts == [str(first_auth), str(second_auth)]
        finally:
            finish_second.set()
            response = await call
        if stream:
            chunks = [chunk async for chunk in response]
            assert chunks
            await response.aclose()
        else:
            assert response.choices[0].message.content == "secondary response"
        await logging_obj.async_cleanup_deployment_resources()
    assert (await limiter.get_concurrency_snapshot())["total_active"] == 0
