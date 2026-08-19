"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


def test_zai_chat_api_base_keeps_coding_endpoint_for_native_responses_deployment():
    """Chat Completions must remain on Coding API when Responses uses /api/v1."""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    api_base, _ = ZAIChatConfig()._get_openai_compatible_provider_info(
        api_base="https://open.bigmodel.cn/api/v1", api_key="test-key"
    )

    assert api_base == "https://open.bigmodel.cn/api/coding/paas/v4"


def test_zai_responses_api_uses_native_endpoint():
    """Responses uses /api/v1 even when the deployment stores the Coding base."""
    from litellm.llms.zai.responses.transformation import ZAIResponsesAPIConfig

    config = ZAIResponsesAPIConfig()

    assert (
        config.get_complete_url(
            api_base="https://open.bigmodel.cn/api/coding/paas/v4", litellm_params={}
        )
        == "https://open.bigmodel.cn/api/v1/responses"
    )


def test_zai_responses_streaming_events_include_sequence_number():
    """Native Z.AI events are serialized with contiguous Responses sequence fields."""
    from litellm.llms.zai.responses.transformation import ZAIResponsesAPIConfig

    config = ZAIResponsesAPIConfig()
    event = config.transform_streaming_response(
        model="glm-5.3",
        parsed_chunk={
            "type": "response.output_text.delta",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
        },
        logging_obj=None,
    )

    assert event.model_dump(mode="json", exclude_none=True)["sequence_number"] == 0


def test_zai_models_in_model_cost():
    """Test that ZAI models are in the model cost map"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    zai_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
        "zai/glm-4.5-x",
        "zai/glm-4.5-air",
        "zai/glm-4.5-airx",
        "zai/glm-4-32b-0414-128k",
        "zai/glm-4.5-flash",
    ]

    for model in zai_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "zai"


def test_zai_glm46_cost_calculation():
    """Test the cost calculation for glm-4.6"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.6"
    info = litellm.model_cost[key]

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_flash_model_is_free():
    """Test that glm-4.5-flash has zero cost"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.5-flash"
    info = litellm.model_cost[key]

    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0


def test_glm47_supports_reasoning():
    """Test that GLM-4.7 supports reasoning"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.7"
    assert key in litellm.model_cost, f"Model {key} not found in model_cost"

    info = litellm.model_cost[key]
    assert info["supports_reasoning"] is True


def test_glm47_cost_calculation():
    """Test cost calculation for GLM-4.7"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.7",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.7: $0.6/M input, $2.2/M output (same as GLM-4.6)
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_converts_responses_builtin_tools_to_functions():
    """ZAI chat endpoint rejects non-function tool types; convert them."""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "regular_tool",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "local_shell"},
                {
                    "type": "apply_patch",
                    "description": "Apply a patch",
                    "parameters": {"properties": {"patch": {"type": "string"}}},
                },
            ],
        },
        optional_params={},
        model="glm-5.2",
        drop_params=False,
    )

    tools = optional_params["tools"]
    assert [tool["type"] for tool in tools] == ["function", "function", "function"]
    assert tools[0]["function"]["name"] == "regular_tool"
    assert tools[1]["function"]["name"] == "local_shell"
    assert tools[1]["function"]["parameters"] == {
        "type": "object",
        "additionalProperties": True,
    }
    assert tools[2]["function"]["name"] == "apply_patch"
    assert tools[2]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


def test_zai_anthropic_messages_config_is_registered():
    """ZAI should use its native Anthropic-compatible Messages endpoint."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.2",
        provider=LlmProviders.ZAI,
    )

    assert config is not None
    assert config.custom_llm_provider == "zai"
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="glm-5.2",
            optional_params={},
            litellm_params={},
        )
        == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
            api_key=None,
            model="glm-5.2",
            optional_params={},
            litellm_params={},
        )
        == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.z.ai/api/paas/v4",
            api_key=None,
            model="glm-5.2",
            optional_params={},
            litellm_params={},
        )
        == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    )


@pytest.mark.asyncio
async def test_zai_anthropic_messages_uses_native_endpoint(respx_mock, monkeypatch):
    """Anthropic Messages calls for ZAI should not be bridged to chat/completions."""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://open.bigmodel.cn/api/anthropic/v1/messages").respond(
        json={
            "id": "msg_zai_123",
            "type": "message",
            "role": "assistant",
            "model": "glm-5.2",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 9,
                "output_tokens": 2,
                "cache_read_input_tokens": 1,
            },
        }
    )

    response = await litellm.anthropic_messages(
        model="zai/glm-5.2",
        max_tokens=16,
        messages=[{"role": "user", "content": "只回复 ok"}],
    )

    assert response["content"][0]["text"] == "ok"
    assert response["usage"]["cache_read_input_tokens"] == 1
    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert str(request.url) == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    assert request.headers["Authorization"] == "Bearer test-api-key"
    assert request.headers["anthropic-version"] == "2023-06-01"

    request_body = json.loads(request.content)
    assert request_body["model"] == "glm-5.2"
    assert request_body["max_tokens"] == 16
    assert request_body["messages"] == [{"role": "user", "content": "只回复 ok"}]
    assert "tools" not in request_body
