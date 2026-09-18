"""
Test for anthropic_endpoints/endpoints.py, focusing on handling dictionary objects in streaming responses
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import ProxyException
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


@pytest.mark.asyncio
async def test_should_preserve_proxy_exception_status_code():
    """Route errors must retain the status code carried by ProxyException."""
    from fastapi import Response

    from litellm.proxy.anthropic_endpoints.endpoints import anthropic_response

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    proxy_exception = ProxyException(
        message="upstream request was rate limited",
        type="rate_limit_error",
        param=None,
        code=429,
    )

    with (
        patch(
            "litellm.proxy.anthropic_endpoints.endpoints._read_request_body",
            new=AsyncMock(return_value={"model": "test-model"}),
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            new=AsyncMock(side_effect=proxy_exception),
        ),
        patch.object(
            ProxyBaseLLMRequestProcessing,
            "get_custom_headers",
            return_value={"x-litellm-model-id": "test-deployment"},
        ),
        patch.multiple(
            "litellm.proxy.proxy_server",
            proxy_logging_obj=proxy_logging_obj,
            general_settings={},
            llm_router=None,
            proxy_config=MagicMock(),
            user_api_base=None,
            user_max_tokens=None,
            user_model=None,
            user_request_timeout=None,
            user_temperature=None,
            version="test",
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await anthropic_response(
                fastapi_response=Response(),
                request=MagicMock(),
                user_api_key_dict=MagicMock(),
            )

    assert exc_info.value.code == "429"
    assert exc_info.value.headers["x-litellm-model-id"] == "test-deployment"
    proxy_logging_obj.post_call_failure_hook.assert_awaited_once()


class TestAnthropicEndpoints(unittest.TestCase):

    @patch("litellm.litellm_core_utils.safe_json_dumps.safe_dumps")
    @pytest.mark.asyncio
    async def test_async_data_generator_anthropic_dict_handling(self, mock_safe_dumps):
        """Test async_data_generator_anthropic handles dictionary chunks properly"""
        # Setup
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [
            {"type": "message_start", "message": {"id": "msg_123"}},
            "text chunk data",
            {"type": "content_block_delta", "delta": {"text": "more data"}},
            "text chunk data again",
        ]

        mock_user_api_key_dict = MagicMock()
        mock_request_data = {}
        mock_proxy_logging_obj = MagicMock()
        mock_proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
            side_effect=lambda **kwargs: kwargs["response"]
        )

        # Configure safe_dumps to return a properly formatted JSON string
        mock_safe_dumps.side_effect = lambda chunk: json.dumps(chunk)

        # Execute
        result = [
            chunk
            async for chunk in ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
                request_data=mock_request_data,
                proxy_logging_obj=mock_proxy_logging_obj,
            )
        ]

        # Verify
        expected_result = [
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n\n',
            "text chunk data",
            'data: {"type": "content_block_delta", "delta": {"text": "more data"}}\n\n',
            "text chunk data again",
        ]

        self.assertEqual(result, expected_result)

        # Assert safe_dumps was called for dictionary objects
        mock_safe_dumps.assert_any_call(
            {"type": "message_start", "message": {"id": "msg_123"}}
        )
        mock_safe_dumps.assert_any_call(
            {"type": "content_block_delta", "delta": {"text": "more data"}}
        )
        assert (
            mock_safe_dumps.call_count == 2
        )  # Called twice, once for each dict object


class TestEventLoggingBatchEndpoint:
    """Test the stubbed event logging batch endpoint"""

    def test_event_logging_batch_endpoint_exists(self):
        """Test that the event_logging_batch endpoint exists and returns 200"""
        from fastapi import FastAPI

        from litellm.proxy.anthropic_endpoints.endpoints import router

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.post("/api/event_logging/batch", json={"events": []})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
