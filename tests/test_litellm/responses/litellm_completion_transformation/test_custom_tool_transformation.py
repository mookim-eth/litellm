from unittest.mock import MagicMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import ModelResponse


CUSTOM_TOOL = {
    "type": "custom",
    "name": "local_shell",
    "description": "Run a shell command.",
}


def _tool_call_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-custom",
        model="test-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_custom_1",
                            "type": "function",
                            "function": {
                                "name": "local_shell",
                                "arguments": '{"input":"pwd"}',
                            },
                        }
                    ],
                },
            }
        ],
    )


def test_custom_tool_declaration_is_wrapped_as_chat_function():
    request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="test-model",
        input="list files",
        responses_api_request={
            "tools": [CUSTOM_TOOL],
            "tool_choice": {"type": "custom", "name": "local_shell"},
        },
    )

    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "local_shell",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Raw custom tool input.",
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "local_shell"},
    }


def test_custom_tool_call_history_is_replayed_as_chat_tool_messages():
    messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {
                "type": "custom_tool_call",
                "call_id": "call_custom_1",
                "name": "local_shell",
                "input": "pwd",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_custom_1",
                "output": "/workspace",
            },
        ]
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "local_shell"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == ('{"input":"pwd"}')
    assert messages[1] == {
        "role": "tool",
        "content": "/workspace",
        "tool_call_id": "call_custom_1",
    }


def test_chat_function_call_is_restored_as_custom_tool_call():
    response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="list files",
        responses_api_request={"tools": [CUSTOM_TOOL]},
        chat_completion_response=_tool_call_response(),
    )

    assert len(response.output) == 1
    custom_call = next(
        item for item in response.output if item.type == "custom_tool_call"
    )
    assert custom_call.call_id == "call_custom_1"
    assert custom_call.name == "local_shell"
    assert custom_call.input == "pwd"


def test_streaming_tool_events_use_custom_tool_protocol():
    stream_wrapper = MagicMock()
    stream_wrapper.logging_obj = None
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=stream_wrapper,
        request_input="list files",
        responses_api_request={"tools": [CUSTOM_TOOL]},
    )

    iterator._queue_final_tool_call_done_events(_tool_call_response())
    events = [event.model_dump() for event in iterator._pending_tool_events]

    assert events[0]["item"]["type"] == "custom_tool_call"
    assert events[0]["item"]["input"] == ""
    assert events[1]["type"] == "response.custom_tool_call_input.done"
    assert events[1]["input"] == "pwd"
    assert events[2]["item"]["type"] == "custom_tool_call"
    assert events[2]["item"]["input"] == "pwd"
