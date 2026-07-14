"""
Unit tests for BaseResponsesAPIStreamingIterator

Tests core functionality including:
1. Processing chunks and handling ResponseCompletedEvent 
2. Ensuring _update_responses_api_response_id_with_model_id is called for final chunk
3. Verifying ID update is NOT called for non-final chunks (delta events)
4. Edge case handling for invalid JSON, empty chunks, and [DONE] markers

These tests ensure the streaming iterator correctly processes response chunks 
and applies model ID updates only to completed responses, as required for proper
response tracking and logging.
"""

import json
import os
import sys
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    OutputTextDeltaEvent,
)


class TestBaseResponsesAPIStreamingIterator:
    """Test cases for BaseResponsesAPIStreamingIterator"""

    def test_process_chunk_stamps_completion_start_time_once(self):
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None

        def update_completion_start_time(*, completion_start_time):
            mock_logging_obj.completion_start_time = completion_start_time
            mock_logging_obj.model_call_details[
                "completion_start_time"
            ] = completion_start_time

        mock_logging_obj._update_completion_start_time.side_effect = (
            update_completion_start_time
        )
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_config.transform_streaming_response.side_effect = [Mock(), Mock()]

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
        )

        iterator._process_chunk(json.dumps({"type": "response.created"}))
        iterator._process_chunk(
            json.dumps({"type": "response.output_text.delta", "delta": "hello"})
        )

        mock_logging_obj._update_completion_start_time.assert_called_once()
        stamped = mock_logging_obj._update_completion_start_time.call_args.kwargs[
            "completion_start_time"
        ]
        assert isinstance(stamped, datetime)
        assert mock_logging_obj.model_call_details["completion_start_time"] == stamped

    def test_done_marker_does_not_stamp_completion_start_time(self):
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
        )

        assert iterator._process_chunk(STREAM_SSE_DONE_STRING) is None
        mock_logging_obj._update_completion_start_time.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_iterator_preserves_u2028_in_sse_json(self):
        """A real U+2028 byte sequence must not split or drop the completed event."""
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        payload = json.dumps(
            {
                "type": "response.completed",
                "response": {"instructions": "eligible\u2028promo"},
            },
            ensure_ascii=False,
        )
        assert "\u2028" in payload

        async def mock_aiter_bytes():
            yield f"data: {payload}\n\n".encode("utf-8")

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock(side_effect=mock_aiter_bytes)
        mock_response.aiter_lines = Mock(
            side_effect=AssertionError("Responses streaming must decode bytes")
        )
        mock_response.aclose = AsyncMock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        completed_response = Mock(spec=ResponsesAPIResponse)
        completed_event = Mock(spec=ResponseCompletedEvent)
        completed_event.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        completed_event.response = completed_response
        mock_config.transform_streaming_response.return_value = completed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
        )
        iterator._handle_logging_completed_response = Mock()

        chunks = [chunk async for chunk in iterator]

        assert chunks == [completed_event]
        assert iterator.completed_response is completed_event
        parsed_chunk = mock_config.transform_streaming_response.call_args.kwargs[
            "parsed_chunk"
        ]
        assert parsed_chunk["response"]["instructions"] == "eligible\u2028promo"
        mock_response.aiter_bytes.assert_called_once_with()
        mock_response.aiter_lines.assert_not_called()

    @pytest.mark.asyncio
    async def test_slow_ttft_trace_records_streaming_stages(self, monkeypatch):
        from litellm.responses.streaming_iterator import (
            ResponsesAPIStreamingIterator,
            initialize_stream_ttft_trace,
            mark_stream_ttft_trace,
        )

        monkeypatch.setenv("LITELLM_STREAM_TTFT_TRACE_ENABLED", "true")
        monkeypatch.setenv("LITELLM_STREAM_TTFT_TRACE_MIN_DURATION_MS", "0")

        async def mock_aiter_bytes():
            yield b'data: {"type": "response.created"}\n\n'

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.aclose = AsyncMock()
        mock_logging_obj = Mock()
        mock_logging_obj.start_time = datetime.now()
        mock_logging_obj.litellm_call_id = "trace-call-id"
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.completion_start_time = None

        def update_completion_start_time(*, completion_start_time):
            mock_logging_obj.completion_start_time = completion_start_time

        mock_logging_obj._update_completion_start_time.side_effect = (
            update_completion_start_time
        )
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_config.transform_streaming_response.return_value = Mock(
            type="response.created"
        )
        request_data = {}
        initialize_stream_ttft_trace(
            request_data,
            logging_obj=mock_logging_obj,
            model="gpt-test",
            custom_llm_provider="openai",
        )
        mark_stream_ttft_trace(request_data, "provider_request_start")
        mark_stream_ttft_trace(request_data, "provider_headers_received")

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-test",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
            request_data=request_data,
        )

        with patch(
            "litellm.responses.streaming_iterator.verbose_proxy_logger"
        ) as logger:
            await iterator.__anext__()
            iterator.record_proxy_first_yield()
            await iterator.aclose()

        trace = request_data["_litellm_stream_ttft_trace"]
        assert trace["call_id"] == "trace-call-id"
        assert "provider_first_raw_byte" in trace
        assert "provider_first_sse_event" in trace
        assert "iterator_first_yield" in trace
        assert "proxy_first_yield" in trace
        logger.warning.assert_called_once()
        assert "headers_to_first_byte_ms" in logger.warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_async_iterator_still_coalesces_output_text_deltas(self):
        """SSE byte decoding must retain the local output-text coalescing behavior."""
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        events = [
            {
                "type": "response.output_text.delta",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello ",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "world",
            },
            {"type": "response.completed", "response": {"id": "resp_1"}},
        ]

        async def mock_aiter_bytes():
            for event in events:
                yield f"data: {json.dumps(event)}\n\n".encode()

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.aclose = AsyncMock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        def transform_streaming_response(*, parsed_chunk, **kwargs):
            transformed = Mock()
            transformed.type = parsed_chunk["type"]
            transformed.delta = parsed_chunk.get("delta")
            if "response" in parsed_chunk:
                transformed.response = Mock(spec=ResponsesAPIResponse)
            return transformed

        mock_config.transform_streaming_response.side_effect = transform_streaming_response
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            custom_llm_provider="openai",
        )
        iterator._handle_logging_completed_response = Mock()

        chunks = [chunk async for chunk in iterator]

        assert [chunk.type for chunk in chunks] == [
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        ]
        assert chunks[0].delta == "Hello world"

    def test_process_chunk_with_response_completed_event(self):
        """
        Test that _process_chunk correctly processes a ResponseCompletedEvent 
        and calls _update_responses_api_response_id_with_model_id for the final chunk.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create a mock ResponsesAPIResponse for the completed event
        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "original_response_id"
        
        # Create a mock ResponseCompletedEvent
        mock_completed_event = Mock(spec=ResponseCompletedEvent)
        mock_completed_event.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
        mock_completed_event.response = mock_responses_api_response
        
        # Set up the mock transform method to return our completed event
        mock_config.transform_streaming_response.return_value = mock_completed_event
        
        # Mock the _update_responses_api_response_id_with_model_id method
        updated_response = Mock(spec=ResponsesAPIResponse)
        updated_response.id = "updated_response_id"
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Prepare test chunk data
        test_chunk_data = {
            "type": "response.completed",
            "response": {
                "id": "original_response_id",
                "output": [{"type": "message", "content": [{"text": "Hello World"}]}]
            }
        }
        
        with patch.object(
            ResponsesAPIRequestUtils, 
            '_update_responses_api_response_id_with_model_id',
            return_value=updated_response
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))
            
            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            
            # Verify that _update_responses_api_response_id_with_model_id was called
            mock_update_id.assert_called_once_with(
                responses_api_response=mock_responses_api_response,
                litellm_metadata={"model_info": {"id": "model_123"}},
                custom_llm_provider="openai"
            )
            
            # Verify the completed response was stored
            assert iterator.completed_response == result
            
            # Verify the response was updated on the event
            assert result.response == updated_response

    def test_process_chunk_with_delta_event_no_id_update(self):
        """
        Test that _process_chunk correctly processes a delta event
        and does NOT call _update_responses_api_response_id_with_model_id.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create a mock OutputTextDeltaEvent (not a completed event)
        mock_delta_event = Mock(spec=OutputTextDeltaEvent)
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "Hello"
        # Delta events don't have a response attribute
        delattr(mock_delta_event, 'response') if hasattr(mock_delta_event, 'response') else None
        
        # Set up the mock transform method to return our delta event
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Prepare test chunk data for a delta event
        test_chunk_data = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0
        }
        
        with patch.object(
            ResponsesAPIRequestUtils, 
            '_update_responses_api_response_id_with_model_id'
        ) as mock_update_id:
            # Process the chunk
            result = iterator._process_chunk(json.dumps(test_chunk_data))
            
            # Assertions
            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
            
            # Verify that _update_responses_api_response_id_with_model_id was NOT called
            mock_update_id.assert_not_called()
            
            # Verify no completed response was stored (since this is not a completed event)
            assert iterator.completed_response is None

    def test_process_chunk_handles_invalid_json(self):
        """
        Test that _process_chunk gracefully handles invalid JSON.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
        )
        
        # Test with invalid JSON
        result = iterator._process_chunk("invalid json {")
        
        # Should return None for invalid JSON
        assert result is None
        assert iterator.completed_response is None

    def test_process_chunk_handles_done_marker(self):
        """
        Test that _process_chunk correctly handles the [DONE] marker.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        
        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
        )
        
        # Test with [DONE] marker
        result = iterator._process_chunk(STREAM_SSE_DONE_STRING)
        
        # Should return None and set finished flag
        assert result is None
        assert iterator.finished is True

    def test_process_chunk_handles_empty_chunk(self):
        """
        Test that _process_chunk correctly handles empty or None chunks.
        """
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = BaseResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj
        )

        # Test with empty chunk
        result = iterator._process_chunk("")
        assert result is None

        # Test with None chunk
        result = iterator._process_chunk(None)
        assert result is None

    def test_handle_logging_completed_response_with_unpickleable_objects(self):
        """
        Test that _handle_logging_completed_response handles responses containing
        objects that cannot be pickled (like Pydantic ValidatorIterator).

        This test verifies the fix for issue #17192 where streaming with tool_choice
        containing allowed_tools would fail with:
        "cannot pickle 'pydantic_core._pydantic_core.ValidatorIterator' object"

        The fix uses model_dump + model_validate instead of copy.deepcopy.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_success_handler = AsyncMock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )

        # Create a ResponseCompletedEvent with tool_choice that has model_dump
        mock_completed_response = Mock()
        mock_completed_response.model_dump.return_value = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [{"type": "function_call", "name": "search_web"}],
                "tool_choice": {"type": "function", "name": "search_web"}
            }
        }
        # model_validate should return a new mock (the copy)
        type(mock_completed_response).model_validate = Mock(return_value=Mock())

        iterator.completed_response = mock_completed_response

        # This should NOT raise an exception
        # Previously it would fail with: TypeError: cannot pickle 'ValidatorIterator'
        # Mock run_async_function since we're not in async context
        with patch('litellm.responses.streaming_iterator.run_async_function'):
            try:
                iterator._handle_logging_completed_response()
            except TypeError as e:
                if "pickle" in str(e):
                    pytest.fail(f"_handle_logging_completed_response failed with pickle error: {e}")
                raise

    @pytest.mark.asyncio
    async def test_stop_async_iteration_not_logged_as_failure(self):
        """
        Test that StopAsyncIteration is NOT logged as a failure.
        
        This test verifies that when streaming completes normally with StopAsyncIteration,
        the _handle_failure method is NOT called, preventing false error logs in Langfuse
        and other logging integrations.
        
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
        
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        
        # Create an async iterator that raises StopAsyncIteration after yielding one chunk
        async def mock_aiter_bytes():
            yield b'data: {"type": "response.output_text.delta", "delta": "test"}\n\n'
            # Normal end of stream - raise StopAsyncIteration
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_delta_event = Mock()
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "test"
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Consume the iterator until StopAsyncIteration
        chunks_received = []
        try:
            async for chunk in iterator:
                chunks_received.append(chunk)
        except StopAsyncIteration:
            pass  # This is expected
        
        # Verify we got the chunk
        assert len(chunks_received) == 1
        
        # CRITICAL: Verify that failure handlers were NOT called
        # StopAsyncIteration is a normal end of stream, not a failure
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()

    def test_stop_iteration_not_logged_as_failure(self):
        """
        Test that StopIteration is NOT logged as a failure in sync iterator.
        
        This test verifies that when streaming completes normally with StopIteration,
        the _handle_failure method is NOT called, preventing false error logs in Langfuse
        and other logging integrations.
        
        Regression test for: https://github.com/BerriAI/litellm/issues/XXXXX
        """
        from litellm.responses.streaming_iterator import SyncResponsesAPIStreamingIterator
        
        # Mock dependencies
        mock_response = Mock()
        mock_response.headers = {}
        
        # Create a sync iterator that raises StopIteration after yielding one chunk
        def mock_iter_bytes():
            yield b'data: {"type": "response.output_text.delta", "delta": "test"}\n\n'
            # Normal end of stream - raise StopIteration
        
        mock_response.iter_bytes = mock_iter_bytes
        
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_delta_event = Mock()
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_delta_event.delta = "test"
        mock_config.transform_streaming_response.return_value = mock_delta_event
        
        # Create the iterator instance
        iterator = SyncResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai"
        )
        
        # Consume the iterator until StopIteration
        chunks_received = []
        try:
            for chunk in iterator:
                chunks_received.append(chunk)
        except StopIteration:
            pass  # This is expected
        
        # Verify we got the chunk
        assert len(chunks_received) == 1
        
        # CRITICAL: Verify that failure handlers were NOT called
        # StopIteration is a normal end of stream, not a failure
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()
        mock_response.close.assert_called_once()
        assert iterator.response is None

    @pytest.mark.asyncio
    async def test_async_iterator_closes_response_on_normal_end(self):
        """
        Test that a normally exhausted async stream explicitly closes and drops
        the underlying httpx response.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aclose = AsyncMock()

        async def mock_aiter_bytes():
            yield b'data: {"type": "response.output_text.delta", "delta": "test"}\n\n'

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_config = Mock(spec=BaseResponsesAPIConfig)
        mock_delta_event = Mock()
        mock_delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        mock_config.transform_streaming_response.return_value = mock_delta_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        chunks_received = []
        async for chunk in iterator:
            chunks_received.append(chunk)

        assert len(chunks_received) == 1
        mock_response.aclose.assert_awaited_once()
        assert iterator.response is None

    @pytest.mark.asyncio
    async def test_async_iterator_closes_response_on_cancelled_error(self):
        """
        Test that client cancellation closes the upstream response without
        invoking failure logging.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        class CancelledAsyncIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aclose = AsyncMock()
        mock_response.aiter_bytes.return_value = CancelledAsyncIterator()

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        with pytest.raises(asyncio.CancelledError):
            await iterator.__anext__()

        mock_response.aclose.assert_awaited_once()
        assert iterator.response is None
        mock_logging_obj.async_failure_handler.assert_not_called()
        mock_logging_obj.failure_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_completed_logging_uses_bounded_worker(self):
        """
        Test that async Responses streaming success logging is enqueued through
        the bounded logging worker instead of spawning a naked asyncio task.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_success_handler = AsyncMock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )
        iterator.completed_response = Mock()
        iterator._run_post_success_hooks = Mock()

        def close_enqueued_coroutine(async_coroutine):
            async_coroutine.close()

        with patch(
            "litellm.responses.streaming_iterator.GLOBAL_LOGGING_WORKER"
        ) as mock_worker:
            mock_worker.ensure_initialized_and_enqueue.side_effect = (
                close_enqueued_coroutine
            )

            iterator._handle_logging_completed_response()

            mock_worker.ensure_initialized_and_enqueue.assert_called_once()
            mock_logging_obj.handle_sync_success_callbacks_for_async_calls.assert_called_once()

    def test_process_chunk_response_failed_calls_failure_handler(self):
        """
        Test that a RESPONSE_FAILED event routes to failure handlers,
        not success handlers. Failed responses represent genuine LLM-level
        errors and should be logged as failures.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {"retry-after": "60"}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_failed_123"
        mock_responses_api_response.error = {
            "type": "server_error",
            "message": "The model encountered an error",
        }
        mock_responses_api_response.usage = None

        mock_failed_event = Mock(spec=ResponseFailedEvent)
        mock_failed_event.type = ResponsesAPIStreamEvents.RESPONSE_FAILED
        mock_failed_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_failed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        test_chunk_data = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed_123",
                "error": {
                    "type": "server_error",
                    "message": "The model encountered an error",
                },
            },
        }

        with patch.object(
            ResponsesAPIRequestUtils,
            "_update_responses_api_response_id_with_model_id",
            return_value=mock_responses_api_response,
        ), patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async, patch(
            "litellm.responses.streaming_iterator.executor"
        ) as mock_executor:
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_FAILED
            assert iterator.completed_response == result

            # Failure handler should have been called via _handle_failure
            mock_run_async.assert_called_once()
            call_kwargs = mock_run_async.call_args
            assert (
                call_kwargs[1]["async_function"]
                == mock_logging_obj.async_failure_handler
            )

            mock_executor.submit.assert_called_once()
            submit_args = mock_executor.submit.call_args
            assert submit_args[0][0] == mock_logging_obj.failure_handler

    @pytest.mark.asyncio
    async def test_async_response_failed_logging_uses_bounded_worker(self):
        """
        Test that failure logging from async Responses streaming uses the
        bounded logging worker and avoids direct executor submission unless
        there are actual sync failure callbacks.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = AsyncMock()
        mock_logging_obj.failure_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_failed_123"
        mock_responses_api_response.error = {"message": "model failed"}
        mock_responses_api_response.usage = None

        mock_failed_event = Mock(spec=ResponseFailedEvent)
        mock_failed_event.type = ResponsesAPIStreamEvents.RESPONSE_FAILED
        mock_failed_event.response = mock_responses_api_response
        mock_config.transform_streaming_response.return_value = mock_failed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        test_chunk_data = {
            "type": "response.failed",
            "response": {"id": "resp_failed_123", "error": {"message": "model failed"}},
        }

        def close_enqueued_coroutine(async_coroutine):
            async_coroutine.close()

        with patch.object(
            ResponsesAPIRequestUtils,
            "_update_responses_api_response_id_with_model_id",
            return_value=mock_responses_api_response,
        ), patch(
            "litellm.responses.streaming_iterator.GLOBAL_LOGGING_WORKER"
        ) as mock_worker, patch(
            "litellm.responses.streaming_iterator.executor"
        ) as mock_executor:
            mock_worker.ensure_initialized_and_enqueue.side_effect = (
                close_enqueued_coroutine
            )

            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            mock_worker.ensure_initialized_and_enqueue.assert_called_once()
            mock_logging_obj.handle_sync_failure_callbacks_for_async_calls.assert_called_once()
            mock_executor.submit.assert_not_called()

    def test_process_chunk_response_incomplete_calls_success_handler(self):
        """
        Test that a RESPONSE_INCOMPLETE event routes to success handlers.
        Incomplete responses (e.g. max_output_tokens reached) are still valid
        responses with usage data — analogous to finish_reason='length' in chat.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_logging_obj.async_success_handler = Mock()
        mock_logging_obj.success_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_incomplete_123"
        mock_responses_api_response.incomplete_details = {
            "reason": "max_output_tokens"
        }
        mock_responses_api_response.usage = None

        mock_incomplete_event = Mock(spec=ResponseIncompleteEvent)
        mock_incomplete_event.type = ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
        mock_incomplete_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_incomplete_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="openai",
        )

        test_chunk_data = {
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete_123",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }

        with patch.object(
            ResponsesAPIRequestUtils,
            "_update_responses_api_response_id_with_model_id",
            return_value=mock_responses_api_response,
        ), patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async, patch(
            "litellm.responses.streaming_iterator.executor"
        ) as mock_executor:
            iterator._run_post_success_hooks = Mock()
            result = iterator._process_chunk(json.dumps(test_chunk_data))

            assert result is not None
            assert result.type == ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE
            assert iterator.completed_response == result

            # Success handler should have been called (via _handle_logging_completed_response)
            mock_run_async.assert_called_once()
            call_kwargs = mock_run_async.call_args
            assert (
                call_kwargs[1]["async_function"]
                == mock_logging_obj.async_success_handler
            )
            mock_logging_obj.handle_sync_success_callbacks_for_async_calls.assert_called_once()
            mock_executor.submit.assert_not_called()

            # Failure handlers should NOT have been called
            mock_logging_obj.async_failure_handler.assert_not_called()
            mock_logging_obj.failure_handler.assert_not_called()


    def test_process_chunk_response_failed_server_overloaded_raises_retryable(
        self, caplog
    ):
        """
        A RESPONSE_FAILED SSE event carrying error.code = server_is_overloaded (or
        slow_down) must surface as a retryable RateLimitError (429) instead of
        flowing the dead chunk downstream, so Router/retry logic can kick in.
        """
        import logging as _logging
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
        from litellm import RateLimitError

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_failed_123"
        mock_responses_api_response.error = {
            "code": "server_is_overloaded",
            "message": "Selected model is at capacity. Please try a different model.",
        }
        mock_responses_api_response.usage = None

        mock_failed_event = Mock(spec=ResponseFailedEvent)
        mock_failed_event.type = ResponsesAPIStreamEvents.RESPONSE_FAILED
        mock_failed_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_failed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="chatgpt",
        )

        test_chunk_data = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed_123",
                "error": {
                    "code": "server_is_overloaded",
                    "message": "Selected model is at capacity. Please try a different model.",
                },
            },
        }

        with caplog.at_level(_logging.WARNING, logger="LiteLLM"), patch.object(
            ResponsesAPIRequestUtils,
            "_update_responses_api_response_id_with_model_id",
            return_value=mock_responses_api_response,
        ), patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ), patch(
            "litellm.responses.streaming_iterator.executor"
        ):
            with pytest.raises(RateLimitError) as exc_info:
                iterator._process_chunk(json.dumps(test_chunk_data))

        assert exc_info.value.status_code == 429
        assert "Selected model is at capacity" in str(exc_info.value)
        assert "retry-after" not in exc_info.value.response.headers
        assert any(
            "server_is_overloaded" in rec.message
            and "RateLimitError" in rec.message
            for rec in caplog.records
        ), f"expected overload->429 warning, got: {[r.message for r in caplog.records]}"

    def test_process_chunk_response_failed_server_overloaded_preserves_retry_after_if_present(
        self,
    ):
        """
        If the upstream does include retry-after on the HTTP response, keep it on the
        surfaced RateLimitError so router cooldown logic can still honor it.
        """
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
        from litellm import RateLimitError

        mock_response = Mock()
        mock_response.headers = {"retry-after": "60"}
        mock_response.aiter_bytes = Mock()
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
        mock_logging_obj.async_failure_handler = Mock()
        mock_logging_obj.failure_handler = Mock()
        mock_config = Mock(spec=BaseResponsesAPIConfig)

        mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
        mock_responses_api_response.id = "resp_failed_123"
        mock_responses_api_response.error = {
            "code": "server_is_overloaded",
            "message": "Selected model is at capacity. Please try a different model.",
        }
        mock_responses_api_response.usage = None

        mock_failed_event = Mock(spec=ResponseFailedEvent)
        mock_failed_event.type = ResponsesAPIStreamEvents.RESPONSE_FAILED
        mock_failed_event.response = mock_responses_api_response

        mock_config.transform_streaming_response.return_value = mock_failed_event

        iterator = ResponsesAPIStreamingIterator(
            response=mock_response,
            model="gpt-4",
            responses_api_provider_config=mock_config,
            logging_obj=mock_logging_obj,
            litellm_metadata={"model_info": {"id": "model_123"}},
            custom_llm_provider="chatgpt",
        )

        test_chunk_data = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed_123",
                "error": {
                    "code": "server_is_overloaded",
                    "message": "Selected model is at capacity. Please try a different model.",
                },
            },
        }

        with patch.object(
            ResponsesAPIRequestUtils,
            "_update_responses_api_response_id_with_model_id",
            return_value=mock_responses_api_response,
        ), patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ), patch(
            "litellm.responses.streaming_iterator.executor"
        ):
            with pytest.raises(RateLimitError) as exc_info:
                iterator._process_chunk(json.dumps(test_chunk_data))

        assert exc_info.value.status_code == 429
        assert exc_info.value.response.headers["retry-after"] == "60"
