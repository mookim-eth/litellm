from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator


def _iterator(request_input="describe this image"):
    iterator = object.__new__(BaseResponsesAPIStreamingIterator)
    iterator.model = "gpt-5.4-mini"
    iterator.logging_obj = SimpleNamespace(litellm_call_id="request-filtered")
    iterator.request_data = {
        "input": [{"role": "user", "content": request_input}]
    }
    return iterator


def test_should_log_streaming_responses_content_filter(caplog):
    iterator = _iterator()
    chunk = SimpleNamespace(
        response=SimpleNamespace(
            incomplete_details=SimpleNamespace(reason="content_filter")
        )
    )

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        iterator._log_incomplete_response(chunk)

    assert any(
        "upstream content filter blocked streaming request" in record.message
        and "request_id=request-filtered" in record.message
        and "reason=content_filter" in record.message
        and "describe this image" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("reason", [None, "max_output_tokens", "unknown"])
def test_should_log_other_incomplete_streaming_responses(caplog, reason):
    iterator = _iterator()
    chunk = MagicMock()
    chunk.response.incomplete_details.reason = reason

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        iterator._log_incomplete_response(chunk)

    expected_reason = reason or "unknown"
    assert any(
        "upstream returned an incomplete streaming response" in record.message
        and f"reason={expected_reason}" in record.message
        and "event=" in record.message
        for record in caplog.records
    )
    assert not any("describe this image" in record.message for record in caplog.records)
