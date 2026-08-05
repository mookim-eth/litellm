from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


def _prepare(output_config, *, thinking=None):
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=128,
        messages=[{"role": "user", "content": "hello"}],
        model="glm-5.2",
        thinking=thinking,
        extra_kwargs={
            "custom_llm_provider": "custom_openai",
            "output_config": output_config,
        },
    )
    return completion_kwargs


def test_should_translate_adaptive_effort_without_forwarding_output_config():
    completion_kwargs = _prepare({"effort": "high"}, thinking={"type": "adaptive"})

    assert completion_kwargs["reasoning_effort"] == "high"
    assert "output_config" not in completion_kwargs


def test_should_drop_raw_output_config_when_thinking_is_not_requested():
    completion_kwargs = _prepare({"effort": "high"})

    assert "reasoning_effort" not in completion_kwargs
    assert "output_config" not in completion_kwargs
    assert completion_kwargs["custom_llm_provider"] == "custom_openai"


def test_should_translate_output_config_format_without_forwarding_raw_key():
    completion_kwargs = _prepare(
        {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            }
        }
    )

    assert completion_kwargs["response_format"]["type"] == "json_schema"
    assert "output_config" not in completion_kwargs
