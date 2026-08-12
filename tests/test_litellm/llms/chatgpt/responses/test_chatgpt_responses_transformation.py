"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
import litellm

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm import ModelResponse
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.chatgpt.responses.transformation import (
    CODEX_RESPONSES_LITE_HEADER,
    ChatGPTResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_headers(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="session-123")
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] == "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_prefers_request_auth_file_path(
        self, mock_authenticator_class
    ):
        default_authenticator = MagicMock()
        request_authenticator = MagicMock()
        request_authenticator.get_access_token.return_value = "access-123"
        request_authenticator.get_account_id.return_value = "acct-123"
        mock_authenticator_class.side_effect = [
            default_authenticator,
            request_authenticator,
        ]

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(
            litellm_session_id="session-123",
            chatgpt_auth_file_path="/tmp/chatgpt-account-b.json",
        )
        headers = config.validate_environment(
            headers={},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        mock_authenticator_class.assert_any_call(
            auth_file_path="/tmp/chatgpt-account-b.json",
            api_base=None,
        )
        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"] == ""

    def test_chatgpt_codex_responses_lite_forces_parallel_tool_calls_false(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.6-sol",
            input="hi",
            response_api_optional_request_params={"parallel_tool_calls": True},
            litellm_params=GenericLiteLLMParams(),
            headers={CODEX_RESPONSES_LITE_HEADER: "true"},
        )

        assert request["parallel_tool_calls"] is False
        assert "instructions" not in request

    @pytest.mark.parametrize(
        "text",
        [
            {
                "format": {
                    "type": "json_schema",
                    "name": "codex_output_schema",
                    "strict": False,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "enum": ["allow", "deny"],
                            }
                        },
                        "required": ["outcome"],
                    },
                }
            },
            {"format": {"type": "json_object"}},
            {"format": {"type": "text"}},
        ],
        ids=["json_schema", "json_object", "text"],
    )
    def test_chatgpt_preserves_responses_text_param(self, text):
        config = ChatGPTResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="chatgpt/codex-auto-review",
            input="Review this approval request",
            response_api_optional_request_params={"text": text},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["text"] == text

    def test_chatgpt_preserves_parallel_tool_calls_for_non_lite_requests(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.3-codex",
            input="hi",
            response_api_optional_request_params={"parallel_tool_calls": True},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["parallel_tool_calls"] is True

    def test_chatgpt_responses_extracts_system_and_developer_input_to_instructions(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be terse."}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Prefer patches."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix this bug"}],
                },
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix this bug"}],
            }
        ]
        assert request["instructions"] == "Be terse.\n\nPrefer patches."

    def test_chatgpt_responses_preserves_codex_responses_lite_input_items(self):
        config = ChatGPTResponsesAPIConfig()
        additional_tools = {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.6-sol",
            input=[
                additional_tools,
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Use tools."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "List files"}],
                },
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={CODEX_RESPONSES_LITE_HEADER: "true"},
        )

        assert request["input"] == [
            additional_tools,
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Use tools."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "List files"}],
            },
        ]
        assert "instructions" not in request

    def test_chatgpt_responses_strips_namespace_from_replayed_input_items(self):
        config = ChatGPTResponsesAPIConfig()
        function_call = {
            "type": "function_call",
            "call_id": "call_test",
            "name": "spawn_agent",
            "arguments": '{"task":"inspect"}',
            "namespace": "multi_agent_v1",
        }
        function_call_output = {
            "type": "function_call_output",
            "call_id": "call_test",
            "output": "done",
            "namespace": "multi_agent_v1",
        }

        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.6-sol",
            input=[function_call, function_call_output],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {
                "type": "function_call",
                "call_id": "call_test",
                "name": "spawn_agent",
                "arguments": '{"task":"inspect"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_test",
                "output": "done",
            },
        ]
        assert function_call["namespace"] == "multi_agent_v1"
        assert function_call_output["namespace"] == "multi_agent_v1"

    def test_chatgpt_responses_extracted_instructions_replace_existing_instructions(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Use markdown."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize this diff"}],
                },
            ],
            response_api_optional_request_params={
                "instructions": "Existing provider rule."
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Summarize this diff"}],
            }
        ]
        assert request["instructions"] == "Use markdown."
        assert "Existing provider rule." not in request["instructions"]

    def test_chatgpt_responses_keeps_existing_instructions_without_system_messages(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize this diff"}],
                }
            ],
            response_api_optional_request_params={
                "instructions": "Existing provider rule."
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["instructions"] == "Existing provider rule."

    def test_chatgpt_responses_sends_empty_instructions_without_system_messages(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello!"}],
                }
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["instructions"] == ""

    def test_chatgpt_responses_wraps_string_input_in_user_message_list(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.5",
            input="Hello!",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello!"}],
            }
        ]
        assert request["stream"] is True
        assert request["instructions"] == ""

    def test_chatgpt_responses_preserves_stream_for_non_stream_bridge_calls(self):
        """
        The ChatGPT backend requires `stream=true` on responses requests even
        when LiteLLM later aggregates the SSE response into a single object.
        """
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.5",
            input="Hello!",
            response_api_optional_request_params={"stream": False},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True

    @pytest.mark.parametrize(
        ("input_service_tier", "expected_service_tier"),
        [
            ("priority", "priority"),
            ("fast", "priority"),
        ],
    )
    def test_chatgpt_preserves_priority_service_tier(
        self, input_service_tier, expected_service_tier
    ):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.5",
            input="Hello!",
            response_api_optional_request_params={"service_tier": input_service_tier},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["service_tier"] == expected_service_tier

    def test_chatgpt_responses_never_fake_stream_for_unknown_models(self):
        """
        Fresh ChatGPT model launches may not yet exist in model metadata, but
        the provider still requires native SSE streaming.
        """
        config = ChatGPTResponsesAPIConfig()

        assert (
            config.should_fake_stream(
                model="chatgpt/gpt-5.5",
                stream=True,
                custom_llm_provider="chatgpt",
            )
            is False
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [{"type": "compaction", "compact_threshold": 200000}],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "parallel_tool_calls": False,
                "reasoning": {"effort": "medium"},
                "text": {"format": {"type": "json_object"}},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["parallel_tool_calls"] is False
        assert request["reasoning"] == {"effort": "medium"}
        assert request["text"] == {"format": {"type": "json_object"}}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {"type": "function", "function": {"name": "hello"}}

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    def test_chatgpt_non_stream_sse_reconstructs_empty_completed_output(self):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
        }
        sse_events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg_123",
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant",
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_123",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "text": "",
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_123",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_123",
                "output_index": 0,
                "content_index": 0,
                "delta": "!",
            },
            {
                "type": "response.completed",
                "response": response_payload,
            },
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert len(parsed.output) == 1
        assert parsed.output[0].type == "message"
        assert parsed.output[0].role == "assistant"
        assert parsed.output[0].content[0].text == "Hello!"
        assert parsed.output_text == "Hello!"

    def test_chatgpt_non_stream_sse_reconstructed_output_transforms_to_chat_completion(
        self,
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        }
        sse_events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg_456",
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant",
                },
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_456",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "text": "",
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_456",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_456",
                "output_index": 0,
                "content_index": 0,
                "delta": " there",
            },
            {
                "type": "response.completed",
                "response": response_payload,
            },
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        model_response = LiteLLMResponsesTransformationHandler().transform_response(
            model="chatgpt/gpt-5.4",
            raw_response=parsed,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "Hello!"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert model_response.choices[0].message.content == "Hi there"

    def test_chatgpt_non_stream_sse_failed_server_overloaded_is_retryable(
        self, caplog
    ):
        import logging as _logging

        config = ChatGPTResponsesAPIConfig()
        sse_events = [
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "status": "failed",
                    "error": {
                        "code": "server_is_overloaded",
                        "message": "Selected model is at capacity. Please try a different model.",
                    },
                },
            }
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with caplog.at_level(_logging.WARNING, logger="LiteLLM"):
            with pytest.raises(OpenAIError) as exc_info:
                config.transform_response_api_response(
                    model="chatgpt/gpt-5.4",
                    raw_response=raw_response,
                    logging_obj=logging_obj,
                )

        assert exc_info.value.status_code == 429
        assert "Selected model is at capacity" in exc_info.value.message
        assert any(
            "server_is_overloaded/slow_down" in rec.message
            and "mapped to 429" in rec.message
            for rec in caplog.records
        ), f"expected mapped-to-429 warning, got: {[r.message for r in caplog.records]}"

    def test_chatgpt_non_stream_sse_failed_preserves_retry_after_headers(self):
        config = ChatGPTResponsesAPIConfig()
        sse_events = [
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "status": "failed",
                    "error": {
                        "code": "server_is_overloaded",
                        "message": "Selected model is at capacity. Please try a different model.",
                    },
                },
            }
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "retry-after": "60",
            },
            text=sse_body,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.response.headers["retry-after"] == "60"
        assert exc_info.value.headers is not None
        assert exc_info.value.headers["retry-after"] == "60"

    def test_chatgpt_non_stream_sse_failed_invalid_prompt_stays_bad_request(self):
        config = ChatGPTResponsesAPIConfig()
        sse_events = [
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "status": "failed",
                    "error": {
                        "code": "invalid_prompt",
                        "message": "Invalid request.",
                    },
                },
            }
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.message == "Invalid request."

    def test_should_summarize_chatgpt_non_stream_sse_content_filter(self):
        config = ChatGPTResponsesAPIConfig()
        sse_events = [
            {
                "type": "response.created",
                "response": {"id": "resp_filtered", "status": "in_progress"},
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_filtered",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                },
            },
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4-mini",
                raw_response=raw_response,
                logging_obj=MagicMock(),
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.message == (
            "The upstream response was blocked by the content filter "
            "(content_policy_violation)."
        )
        assert exc_info.value.body == {
            "error": {
                "code": "content_policy_violation",
                "message": (
                    "The upstream response was blocked by the content filter "
                    "(content_policy_violation)."
                ),
                "param": None,
                "type": "invalid_request_error",
            }
        }

        with pytest.raises(litellm.ContentPolicyViolationError):
            litellm.exception_type(
                model="gpt-5.4-mini",
                original_exception=exc_info.value,
                custom_llm_provider="chatgpt",
            )

    def test_chatgpt_non_stream_error_event_uses_top_level_status_code(self):
        config = ChatGPTResponsesAPIConfig()
        sse_events = [
            {
                "type": "error",
                "status": 429,
                "error": {
                    "type": "rate_limit_error",
                    "message": "Too many requests.",
                },
            }
        ]
        sse_body = "\n".join(
            [f"data: {json.dumps(event)}" for event in sse_events]
            + ["data: [DONE]", ""]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.message == "Too many requests."
