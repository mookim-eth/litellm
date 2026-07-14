import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator


class _OneEventThenStallResponse:
    def __init__(self) -> None:
        self.headers = {}
        self.closed = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def aiter_bytes(self):
        yield b'data: {"type":"response.created"}\n\n'
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    async def aclose(self) -> None:
        self.closed.set()


def _make_iterator(response, timeout_seconds: float) -> ResponsesAPIStreamingIterator:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.litellm_call_id = "sse-idle-timeout-test"
    iterator = ResponsesAPIStreamingIterator(
        response=response,
        model="test-model",
        responses_api_provider_config=MagicMock(),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        provider_sse_event_timeout_seconds=timeout_seconds,
    )
    processed_chunk = SimpleNamespace()
    iterator._process_chunk = MagicMock(return_value=processed_chunk)
    iterator._call_post_streaming_deployment_hook = AsyncMock(
        return_value=processed_chunk
    )
    iterator._handle_failure = MagicMock()
    return iterator


@pytest.mark.asyncio
async def test_sse_event_timeout_resets_after_event_and_closes_response():
    response = _OneEventThenStallResponse()
    iterator = _make_iterator(response=response, timeout_seconds=0.02)

    first_event = await iterator.__anext__()
    assert first_event is not None

    with pytest.raises(litellm.Timeout, match="next provider Responses SSE event"):
        await iterator.__anext__()

    assert response.closed.is_set()
    assert response.cancelled.is_set()
    assert iterator.finished is True
    iterator._handle_failure.assert_called_once()


@pytest.mark.asyncio
async def test_sse_event_timeout_covers_pending_coalescing_read():
    response = _OneEventThenStallResponse()
    iterator = _make_iterator(response=response, timeout_seconds=0.02)
    pending_read = asyncio.create_task(iterator.stream_iterator.__anext__())
    first_sse = await pending_read
    assert first_sse.data == '{"type":"response.created"}'

    iterator._pending_stream_event_task = asyncio.create_task(
        iterator.stream_iterator.__anext__()
    )
    iterator._pending_stream_event_task_started_at = (
        asyncio.get_running_loop().time()
    )

    with pytest.raises(litellm.Timeout, match="next provider Responses SSE event"):
        await iterator.__anext__()

    assert response.closed.is_set()
    assert response.cancelled.is_set()
    assert iterator._pending_stream_event_task is None
