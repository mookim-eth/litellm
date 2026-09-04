import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import litellm
from litellm import Router
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.hooks.chatgpt_account_concurrency_limiter import (
    ChatGPTAccountConcurrencyLimiter,
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
        {"plus": 3, "k12": 3, "team": 5, "pro": 20},
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
async def test_should_allow_twenty_concurrent_requests_for_pro_plan(
    tmp_path: Path, configured_limits: None
) -> None:
    auth_file = tmp_path / "pro.json"
    _write_auth_file(auth_file, "pro-account", "pro")
    limiter = ChatGPTAccountConcurrencyLimiter(_FakeInternalUsageCache())
    logging_objects = []

    for _ in range(20):
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
