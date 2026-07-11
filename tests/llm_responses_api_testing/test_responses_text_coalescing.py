import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    FunctionCallArgumentsDeltaEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
)


class FakeResponse:
    def __init__(self, events, *, sse_separators=False):
        self.headers = {}
        self._events = events
        self._sse_separators = sse_separators
        self.aclose = AsyncMock()

    async def _lines(self):
        for event in self._events:
            if self._sse_separators:
                yield f"event: {event['type']}"
            yield f"data: {json.dumps(event)}"
            if self._sse_separators:
                yield ""

    def aiter_lines(self):
        return self._lines()


def _text(delta, sequence, **extra):
    return {
        "type": "response.output_text.delta",
        "item_id": "msg-1",
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
        "sequence_number": sequence,
        **extra,
    }


def _iterator(events, *, sse_separators=False, request_data=None):
    config = Mock()

    def transform_streaming_response(*, parsed_chunk, **_kwargs):
        event_type = parsed_chunk["type"]
        if event_type == "response.output_text.delta":
            return OutputTextDeltaEvent(**parsed_chunk)
        if event_type == "response.output_text.done":
            return OutputTextDoneEvent(**parsed_chunk)
        if event_type == "response.function_call_arguments.delta":
            return FunctionCallArgumentsDeltaEvent(**parsed_chunk)
        raise AssertionError(f"unexpected event: {parsed_chunk}")

    config.transform_streaming_response.side_effect = transform_streaming_response
    logging = Mock()
    logging.model_call_details = {"litellm_params": {}}
    return ResponsesAPIStreamingIterator(
        response=FakeResponse(events, sse_separators=sse_separators),
        model="test-model",
        responses_api_provider_config=config,
        logging_obj=logging,
        request_data=request_data,
    )


@pytest.mark.asyncio
async def test_coalesces_text_and_preserves_sequence_start():
    events = [
        _text("alpha", 10),
        _text(" beta", 11),
        {
            "type": "response.output_text.done",
            "item_id": "msg-1",
            "output_index": 0,
            "content_index": 0,
            "text": "alpha beta",
            "sequence_number": 12,
        },
    ]

    chunks = [chunk async for chunk in _iterator(events)]

    assert [chunk.sequence_number for chunk in chunks] == [10, 11]
    assert chunks[0].delta == "alpha beta"
    assert chunks[1].text == "alpha beta"


@pytest.mark.asyncio
async def test_coalesces_across_real_sse_empty_separator_lines():
    events = [
        _text("alpha", 10),
        _text(" beta", 11),
        {
            "type": "response.output_text.done",
            "item_id": "msg-1",
            "output_index": 0,
            "content_index": 0,
            "text": "alpha beta",
            "sequence_number": 12,
        },
    ]

    chunks = [chunk async for chunk in _iterator(events, sse_separators=True)]

    assert [chunk.sequence_number for chunk in chunks] == [10, 11]
    assert chunks[0].delta == "alpha beta"
    assert chunks[1].text == "alpha beta"


@pytest.mark.asyncio
async def test_does_not_merge_across_tool_boundary_and_preserves_text():
    events = [
        _text("before", 0),
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "call-1",
            "output_index": 1,
            "delta": '{"city":"Tokyo"}',
            "sequence_number": 1,
        },
        _text(" after", 2),
    ]

    chunks = [chunk async for chunk in _iterator(events)]

    assert [chunk.sequence_number for chunk in chunks] == [0, 1, 2]
    assert chunks[0].delta + chunks[2].delta == "before after"
    assert chunks[1].delta == '{"city":"Tokyo"}'


@pytest.mark.asyncio
async def test_safety_buffering_delta_is_not_merged():
    events = [
        _text("plain", 0),
        _text(
            " guarded",
            1,
            safety_buffering={"use_cases": ["cyber"], "reasons": ["user_risk"]},
        ),
        _text(" tail", 2),
    ]

    chunks = [chunk async for chunk in _iterator(events)]

    assert [chunk.sequence_number for chunk in chunks] == [0, 1, 2]
    assert "".join(chunk.delta for chunk in chunks) == "plain guarded tail"
    assert chunks[1].safety_buffering == {
        "use_cases": ["cyber"],
        "reasons": ["user_risk"],
    }


@pytest.mark.asyncio
async def test_nonempty_logprobs_delta_is_not_merged():
    events = [
        _text("plain", 0),
        _text(" scored", 1, logprobs=[{"token": " scored", "logprob": -0.1}]),
        _text(" tail", 2),
    ]

    chunks = [chunk async for chunk in _iterator(events)]

    assert [chunk.sequence_number for chunk in chunks] == [0, 1, 2]
    assert "".join(chunk.delta for chunk in chunks) == "plain scored tail"


@pytest.mark.asyncio
async def test_flushes_after_coalescing_deadline():
    async def delayed_lines():
        yield f"data: {json.dumps(_text('first', 0))}"
        import asyncio

        await asyncio.sleep(0.03)
        yield f"data: {json.dumps(_text('second', 1))}"

    response = FakeResponse([])
    response.aiter_lines = delayed_lines
    iterator = _iterator([])
    iterator.response = response
    iterator.stream_iterator = response.aiter_lines()

    with patch(
        "litellm.responses.streaming_iterator.STREAM_TEXT_COALESCE_SECONDS",
        0.01,
    ):
        chunks = [chunk async for chunk in iterator]

    assert [chunk.delta for chunk in chunks] == ["first", "second"]


@pytest.mark.asyncio
async def test_sequence_numbers_continue_across_fallback_iterators():
    request_data = {}
    first_attempt = _iterator(
        [
            _text("first", 20),
            {
                "type": "response.output_text.done",
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "text": "first",
                "sequence_number": 21,
            },
        ],
        request_data=request_data,
    )
    fallback_attempt = _iterator(
        [
            _text(" fallback", 0),
            {
                "type": "response.output_text.done",
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "text": " fallback",
                "sequence_number": 1,
            },
        ],
        request_data=request_data,
    )

    first_chunks = [chunk async for chunk in first_attempt]
    fallback_chunks = [chunk async for chunk in fallback_attempt]

    assert [chunk.sequence_number for chunk in first_chunks] == [20, 21]
    assert [chunk.sequence_number for chunk in fallback_chunks] == [22, 23]
