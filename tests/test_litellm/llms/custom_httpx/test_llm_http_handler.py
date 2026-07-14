import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams


class _OneSSEEventThenStallStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"type":"response.created"}\n\n'
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    async def aclose(self):
        self.closed.set()


class _CompleteSSEStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"type":"response.created"}\n\n'
        yield b'data: {"type":"response.completed"}\n\n'

    async def aclose(self):
        self.closed.set()


@pytest.mark.asyncio
async def test_responses_provider_headers_timeout_cancels_post():
    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://provider.example/v1/responses"
    provider_config.transform_responses_api_request.return_value = {
        "model": "test-model",
        "input": "hello",
        "stream": True,
    }
    cancelled = asyncio.Event()

    async def wait_for_headers(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = Mock(spec=AsyncHTTPHandler)
    client.post = wait_for_headers
    logging_obj = Mock()
    logging_obj.model_call_details = {}
    logging_obj.litellm_call_id = "timeout-test"

    with pytest.raises(litellm.Timeout, match="provider HTTP response headers"):
        await handler.async_response_api_handler(
            model="test-model",
            input="hello",
            responses_api_provider_config=provider_config,
            response_api_optional_request_params={"stream": True},
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(
                api_base="https://provider.example/v1/responses"
            ),
            logging_obj=logging_obj,
            client=client,
            provider_headers_timeout_seconds=0.01,
        )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_responses_provider_headers_timeout_stops_after_headers():
    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://provider.example/v1/responses"
    provider_config.transform_responses_api_request.return_value = {
        "model": "test-model",
        "input": "hello",
        "stream": True,
    }
    response = Mock()
    response.headers = {}
    client = Mock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=response)
    logging_obj = Mock()
    logging_obj.model_call_details = {}

    iterator = await handler.async_response_api_handler(
        model="test-model",
        input="hello",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={"stream": True},
        custom_llm_provider="openai",
        litellm_params=GenericLiteLLMParams(
            api_base="https://provider.example/v1/responses"
        ),
        logging_obj=logging_obj,
        client=client,
        provider_headers_timeout_seconds=0.01,
        provider_sse_event_timeout_seconds=300,
    )

    assert iterator.response is response
    assert iterator.provider_sse_event_timeout_seconds == 300


@pytest.mark.asyncio
async def test_nonstreaming_responses_timeout_only_waits_for_headers():
    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://provider.example/v1/responses"
    provider_config.transform_responses_api_request.return_value = {
        "model": "test-model",
        "input": "hello",
    }
    expected_result = Mock()
    provider_config.transform_response_api_response.return_value = expected_result
    body_read = asyncio.Event()

    response = Mock()

    async def slow_body_read():
        await asyncio.sleep(0.02)
        body_read.set()

    response.aread = slow_body_read
    client = Mock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=response)
    logging_obj = Mock()
    logging_obj.model_call_details = {}

    result = await handler.async_response_api_handler(
        model="test-model",
        input="hello",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={"stream": False},
        custom_llm_provider="openai",
        litellm_params=GenericLiteLLMParams(
            api_base="https://provider.example/v1/responses"
        ),
        logging_obj=logging_obj,
        client=client,
        provider_headers_timeout_seconds=0.01,
    )

    assert result is expected_result
    assert body_read.is_set()
    assert client.post.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_nonstreaming_provider_native_sse_event_timeout_closes_response():
    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = (
        "https://provider.example/v1/responses"
    )
    provider_config.transform_responses_api_request.return_value = {
        "model": "test-model",
        "input": "hello",
        "stream": True,
    }
    stream = _OneSSEEventThenStallStream()
    response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://provider.example/v1/responses"),
    )
    client = Mock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=response)
    logging_obj = Mock()
    logging_obj.model_call_details = {}
    logging_obj.litellm_call_id = "nonstream-sse-timeout-test"

    with pytest.raises(litellm.Timeout, match="next provider Responses SSE event"):
        await handler.async_response_api_handler(
            model="test-model",
            input="hello",
            responses_api_provider_config=provider_config,
            response_api_optional_request_params={"stream": False},
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(
                api_base="https://provider.example/v1/responses"
            ),
            logging_obj=logging_obj,
            client=client,
            provider_sse_event_timeout_seconds=0.02,
        )

    assert client.post.await_args.kwargs["stream"] is True
    assert stream.closed.is_set()
    assert stream.cancelled.is_set()


@pytest.mark.asyncio
async def test_nonstreaming_provider_native_sse_buffer_preserves_complete_body():
    stream = _CompleteSSEStream()
    response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://provider.example/v1/responses"),
    )
    logging_obj = Mock()
    logging_obj.litellm_call_id = "nonstream-sse-complete-test"

    buffered_response = (
        await BaseLLMHTTPHandler._read_provider_sse_body_with_event_timeout(
            response=response,
            timeout_seconds=0.1,
            model="test-model",
            custom_llm_provider="openai",
            logging_obj=logging_obj,
        )
    )

    assert buffered_response.status_code == 200
    assert buffered_response.text == (
        'data: {"type":"response.created"}\n\n'
        'data: {"type":"response.completed"}\n\n'
    )
    assert stream.closed.is_set()


def test_prepare_fake_stream_request():
    # Initialize the BaseLLMHTTPHandler
    handler = BaseLLMHTTPHandler()

    # Test case 1: fake_stream is True
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that "stream" key is removed from data
    assert "stream" not in result_data
    # Verify other data remains unchanged
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 2: fake_stream is False
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = False

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream remains True
    assert result_stream is True
    # Verify that data remains unchanged
    assert "stream" in result_data
    assert result_data["stream"] is True
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 3: data doesn't have stream key but fake_stream is True
    stream = True
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that data remains unchanged (since there was no stream key to remove)
    assert "stream" not in result_data
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_extra_headers():
    """
    Test that async_anthropic_messages_handler correctly extracts and merges
    extra_headers from kwargs with proper priority.
    """
    handler = BaseLLMHTTPHandler()
    
    # Mock the config
    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-3-opus-20240229", "messages": []}
    )
    
    # Mock the client
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)
    
    # Mock logging object
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False
    
    # Test case 1: Only extra_headers in kwargs
    kwargs = {
        "extra_headers": {
            "X-Custom-Header": "from-kwargs",
            "X-Auth-Token": "token123",
        }
    }
    
    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = None
        
        # Capture what headers are passed to validate_anthropic_messages_environment
        captured_headers = {}
        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")
        
        mock_config.validate_anthropic_messages_environment = capture_validate
        
        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass  # We're testing header extraction, not the full flow
        
        # Verify extra_headers were extracted and merged
        assert "X-Custom-Header" in captured_headers
        assert captured_headers["X-Custom-Header"] == "from-kwargs"
        assert "X-Auth-Token" in captured_headers
        assert captured_headers["X-Auth-Token"] == "token123"


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_passes_litellm_metadata():
    """Ensure litellm_metadata from kwargs is forwarded via update_from_kwargs.

    Routes like /messages store model_info under kwargs['litellm_metadata'].
    The handler must forward this so that use_custom_pricing_for_model can
    detect custom pricing. Regression test for #23185.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    custom_model_info = {
        "id": "claude-sonnet-4-custom-pricing",
        "input_cost_per_token": 0.0003,
        "output_cost_per_token": 0.0015,
    }
    kwargs = {
        "litellm_metadata": {
            "model_info": custom_model_info,
            "deployment": "anthropic/claude-sonnet-4-20250514",
        },
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs=kwargs,
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    kwargs_arg = call_kwargs.kwargs.get(
        "kwargs", call_kwargs[1].get("kwargs", {})
    ) if call_kwargs.kwargs else call_kwargs[1].get("kwargs", {})

    assert "litellm_metadata" in kwargs_arg
    assert kwargs_arg["litellm_metadata"]["model_info"] == custom_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_header_priority():
    """
    Test that async_anthropic_messages_handler respects header priority:
    forwarded < extra_headers < provider_specific
    """
    handler = BaseLLMHTTPHandler()
    
    # Mock the config
    mock_config = Mock()
    mock_client = AsyncMock()
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False
    
    # Test with all three header sources
    kwargs = {
        "headers": {"X-Priority": "forwarded", "X-Forwarded-Only": "keep"},
        "extra_headers": {"X-Priority": "extra", "X-Extra-Only": "also-keep"},
    }
    
    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = {
            "X-Priority": "provider",
            "X-Provider-Only": "keep-this-too"
        }
        
        captured_headers = {}
        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")
        
        mock_config.validate_anthropic_messages_environment = capture_validate
        mock_config.transform_anthropic_messages_request = Mock(
            return_value={"model": "claude-3-opus-20240229", "messages": []}
        )
        
        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass
        
        # Verify priority: provider_specific should win
        assert captured_headers["X-Priority"] == "provider"
        # Verify all unique headers from different sources are present
        assert captured_headers["X-Forwarded-Only"] == "keep"
        assert captured_headers["X-Extra-Only"] == "also-keep"
        assert captured_headers["X-Provider-Only"] == "keep-this-too"
