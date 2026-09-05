"""Native Anthropic timing must include upstream wait, not empty SSE events."""

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


def sse(payload, newline=b"\n"):
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + newline * 2


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"type": "message_start"}, False),
        ({"type": "ping"}, False),
        ({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, False),
        (
            {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            },
            False,
        ),
        (
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "test", "input": {}},
            },
            False,
        ),
        ({"type": "content_block_delta", "delta": {"text": ""}}, False),
        ({"type": "content_block_delta", "delta": {"signature": "signature"}}, False),
        ({"type": "content_block_delta", "delta": {"text": "hi"}}, True),
        ({"type": "content_block_delta", "delta": {"thinking": "think"}}, True),
        ({"type": "content_block_delta", "delta": {"partial_json": "{"}}, True),
        ({"type": "content_block_start", "content_block": {"text": "hi"}}, True),
        ({"type": "content_block_start", "content_block": {"thinking": "think"}}, True),
        (
            {
                "type": "content_block_start",
                "content_block": {"input": {"city": "Paris"}},
            },
            True,
        ),
        (
            {
                "type": "content_block_start",
                "content_block": {"type": "redacted_thinking", "data": "opaque"},
            },
            True,
        ),
        ([], False),
        (None, False),
    ],
)
def test_should_recognize_only_generated_content(payload, expected):
    assert (
        PassThroughStreamingHandler._anthropic_event_has_content(sse(payload))
        is expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("content_field", ["text", "thinking", "partial_json"])
async def test_should_preserve_request_start_and_wait_for_split_content(
    newline, content_field
):
    start = datetime(2026, 9, 5, 8, 0, 0)
    clock = Mock(wraps=datetime)
    clock.now.return_value = start + timedelta(seconds=5)  # HTTP headers arrive.
    logging_obj = SimpleNamespace(start_time=start, completion_start_time=None)
    logging_obj._update_completion_start_time = Mock(
        side_effect=lambda **kw: setattr(
            logging_obj, "completion_start_time", kw["completion_start_time"]
        )
    )
    content = sse(
        {"type": "content_block_delta", "delta": {content_field: "你好"}}, newline
    )
    # One byte per network chunk exercises split UTF-8, JSON and delimiters.
    chunks = [
        b": keepalive\n\n",
        sse({"type": "message_start"}, newline),
        sse({"type": "ping"}, newline),
        sse({"type": "content_block_start", "content_block": {"text": ""}}, newline),
        b"data: invalid json" + newline * 2,
        *[content[i : i + 1] for i in range(len(content))],
        sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, newline),
    ]
    content_index = len(chunks) - 2

    async def aiter_bytes():
        for index, chunk in enumerate(chunks):
            clock.now.return_value = start + timedelta(seconds=6, milliseconds=index)
            yield chunk

    sink = AsyncMock()
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator.datetime",
            clock,
        ),
        patch("litellm.proxy.pass_through_endpoints.streaming_handler.datetime", clock),
        patch.object(
            PassThroughStreamingHandler, "_route_streaming_logging_to_handler", sink
        ),
    ):
        handler = BaseAnthropicMessagesStreamingIterator(
            logging_obj, {"model": "glm-5.3"}
        )
        iterator = handler.get_async_streaming_response_iterator(
            SimpleNamespace(aiter_bytes=aiter_bytes), {"model": "glm-5.3"}, logging_obj
        )
        index = 0
        async for chunk in iterator:
            assert chunk == chunks[index]  # Timing must not change or delay delivery.
            if index < content_index:
                assert logging_obj.completion_start_time is None
            else:
                assert logging_obj.completion_start_time == start + timedelta(
                    seconds=6, milliseconds=content_index
                )
            index += 1
        await asyncio.sleep(0)

    assert index == len(chunks)
    assert sink.call_args.kwargs["start_time"] == start
    assert sink.call_args.kwargs["raw_bytes"] == chunks
    logging_obj._update_completion_start_time.assert_called_once()


@pytest.mark.asyncio
async def test_should_leave_empty_stream_ttft_unset_for_end_time_fallback():
    start = datetime.now()
    logging_obj = SimpleNamespace(start_time=start, completion_start_time=None)
    logging_obj._update_completion_start_time = Mock()
    chunks = [b": ping\n\n", sse({"type": "message_stop"})]

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk

    with patch.object(
        PassThroughStreamingHandler, "_route_streaming_logging_to_handler", AsyncMock()
    ) as sink:
        handler = BaseAnthropicMessagesStreamingIterator(logging_obj, {})
        result = [
            chunk
            async for chunk in handler.get_async_streaming_response_iterator(
                SimpleNamespace(aiter_bytes=aiter_bytes), {}, logging_obj
            )
        ]
        await asyncio.sleep(0)

    assert result == chunks
    logging_obj._update_completion_start_time.assert_not_called()
    assert sink.call_args.kwargs["start_time"] == start


def test_should_parse_multiline_data_and_mixed_event_line_endings():
    event = b'event: content_block_delta\r\ndata: {"type":"content_block_delta",\r\ndata: "delta":{"text":"hi"}}\r\n\r\n'
    assert PassThroughStreamingHandler._anthropic_event_has_content(event)
    assert not PassThroughStreamingHandler._anthropic_event_has_content(
        b": heartbeat\n\n"
    )


@pytest.mark.asyncio
async def test_should_handle_multiple_events_in_one_chunk_and_stop_parsing_after_content():
    logging_obj = SimpleNamespace(start_time=datetime.now(), completion_start_time=None)
    logging_obj._update_completion_start_time = Mock(
        side_effect=lambda **kw: setattr(
            logging_obj, "completion_start_time", kw["completion_start_time"]
        )
    )
    chunks = [
        sse({"type": "ping"})
        + sse({"type": "content_block_delta", "delta": {"text": "hello"}}, b"\r\n"),
        sse({"type": "content_block_delta", "delta": {"text": "world"}}),
    ]

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk

    with (
        patch.object(
            PassThroughStreamingHandler,
            "_route_streaming_logging_to_handler",
            AsyncMock(),
        ),
        patch.object(
            PassThroughStreamingHandler,
            "_anthropic_event_has_content",
            wraps=PassThroughStreamingHandler._anthropic_event_has_content,
        ) as parser,
    ):
        handler = BaseAnthropicMessagesStreamingIterator(logging_obj, {})
        result = [
            chunk
            async for chunk in handler.get_async_streaming_response_iterator(
                SimpleNamespace(aiter_bytes=aiter_bytes), {}, logging_obj
            )
        ]
        await asyncio.sleep(0)

    assert result == chunks
    assert parser.call_count == 2  # No parsing of the second network chunk.
    logging_obj._update_completion_start_time.assert_called_once()
