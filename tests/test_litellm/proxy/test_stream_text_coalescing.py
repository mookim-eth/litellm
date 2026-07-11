import asyncio

import pytest

from litellm.proxy.proxy_server import _coalesce_plain_text_stream
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
