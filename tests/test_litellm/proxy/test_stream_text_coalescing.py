import asyncio
import time
from unittest.mock import patch

import pytest

from litellm.proxy.proxy_server import _coalesce_plain_text_stream, async_data_generator
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def _chunk(
    content: str = "",
    *,
    finish_reason=None,
    tool_calls=None,
    usage=None,
) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test",
        model="test-model",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


@pytest.mark.asyncio
async def test_coalesces_adjacent_plain_text_chunks():
    async def stream():
        yield _chunk("Hello")
        yield _chunk(" ")
        yield _chunk("world")

    chunks = [chunk async for chunk in _coalesce_plain_text_stream(stream())]

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "Hello world"


@pytest.mark.asyncio
async def test_flushes_text_before_terminal_chunk():
    async def stream():
        yield _chunk("Hello")
        yield _chunk(" world")
        yield _chunk("", finish_reason="stop")

    chunks = [chunk async for chunk in _coalesce_plain_text_stream(stream())]

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello world"
    assert chunks[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_does_not_wait_past_coalescing_window():
    async def stream():
        yield _chunk("first")
        await asyncio.sleep(0.35)
        yield _chunk("second")

    chunks = [chunk async for chunk in _coalesce_plain_text_stream(stream())]

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["first", "second"]


@pytest.mark.asyncio
async def test_preserves_text_order_across_reasoning_and_finish_boundaries():
    reasoning = _chunk("")
    reasoning.choices[0].delta.reasoning_content = "think"

    async def stream():
        yield _chunk("alpha")
        yield _chunk(" beta")
        yield reasoning
        yield _chunk(" gamma")
        yield _chunk(" delta")
        yield _chunk("", finish_reason="stop")

    chunks = [chunk async for chunk in _coalesce_plain_text_stream(stream())]

    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert text == "alpha beta gamma delta"
    assert chunks[1].choices[0].delta.reasoning_content == "think"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_first_chunk_delay_is_bounded_by_configured_window():
    async def stream():
        yield _chunk("first")
        await asyncio.Event().wait()

    with patch("litellm.proxy.proxy_server.STREAM_TEXT_COALESCE_SECONDS", 0.05):
        iterator = _coalesce_plain_text_stream(stream())
        started = time.monotonic()
        first = await asyncio.wait_for(iterator.__anext__(), timeout=0.2)
        elapsed = time.monotonic() - started
        await iterator.aclose()

    assert first.choices[0].delta.content == "first"
    assert 0.03 <= elapsed < 0.15


@pytest.mark.asyncio
async def test_closes_concrete_iterator_when_async_iterable_has_no_aclose():
    inner_closed = asyncio.Event()

    async def inner():
        try:
            yield _chunk("first")
            await asyncio.Event().wait()
        finally:
            inner_closed.set()

    class IterableOnly:
        def __aiter__(self):
            return inner()

    with patch("litellm.proxy.proxy_server.STREAM_TEXT_COALESCE_SECONDS", 0):
        coalesced = _coalesce_plain_text_stream(IterableOnly())
        await coalesced.__anext__()
        await coalesced.aclose()

    assert inner_closed.is_set()


@pytest.mark.asyncio
async def test_callbacks_keep_original_chunk_boundaries_before_outbound_coalescing():
    iterator_hook_chunks = []
    per_chunk_hook_chunks = []

    class RecordingProxyLogging:
        async def async_post_call_streaming_iterator_hook(self, response, **_kwargs):
            async for chunk in response:
                iterator_hook_chunks.append(chunk.choices[0].delta.content)
                yield chunk

        async def async_post_call_streaming_hook(self, response, **_kwargs):
            per_chunk_hook_chunks.append(response.choices[0].delta.content)
            return response

        async def post_call_failure_hook(self, **_kwargs):
            raise AssertionError("failure hook should not run")

    async def response():
        yield _chunk("one")
        yield _chunk(" two")
        yield _chunk(" three")

    with (
        patch("litellm.proxy.proxy_server.proxy_logging_obj", RecordingProxyLogging()),
        patch("litellm.proxy.proxy_server.STREAM_TEXT_COALESCE_SECONDS", 0.05),
    ):
        outbound = [
            event
            async for event in async_data_generator(
                response(), object(), {"model": "test-model"}
            )
        ]

    assert iterator_hook_chunks == ["one", " two", " three"]
    assert per_chunk_hook_chunks == ["one", " two", " three"]
    assert len([event for event in outbound if "[DONE]" not in event]) == 1
