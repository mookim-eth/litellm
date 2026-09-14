"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "hello"}, {"type": "input_text", "text": " world"}],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


def test_image_function_call_output_uses_gemini_function_response_parts():
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    image_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    transform_input = (
        LiteLLMCompletionResponsesConfig
        ._transform_response_input_param_to_chat_completion_message
    )
    messages = transform_input(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the image"}],
            },
            {
                "type": "function_call",
                "name": "inspect_image",
                "call_id": "call_image",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    }
                ],
            },
        ]
    )

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3.8-flash"
    )
    function_response = contents[-1]["parts"][0]["function_response"]

    assert contents[-1]["role"] == "user"
    assert function_response["name"] == "inspect_image"
    assert function_response["parts"][0]["inline_data"] == {
        "mime_type": "image/png",
        "data": image_base64,
    }
