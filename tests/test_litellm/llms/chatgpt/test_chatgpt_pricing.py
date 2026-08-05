import copy

import litellm
import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm import completion_cost
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import Usage
from litellm.utils import _invalidate_model_cost_lowercase_map


def _load_local_model_cost_map() -> None:
    litellm.model_cost = litellm.get_model_cost_map(url="")
    _invalidate_model_cost_lowercase_map()


def test_chatgpt_model_info_uses_base_model_pricing_when_prefixed_entry_is_missing_costs():
    _load_local_model_cost_map()

    original_chatgpt_entry = copy.deepcopy(litellm.model_cost["chatgpt/gpt-5.4"])
    original_base_entry = copy.deepcopy(litellm.model_cost["gpt-5.4"])

    try:
        litellm.model_cost["chatgpt/gpt-5.4"]["input_cost_per_token"] = None
        litellm.model_cost["chatgpt/gpt-5.4"]["output_cost_per_token"] = None
        _invalidate_model_cost_lowercase_map()

        model_info = litellm.get_model_info(
            model="chatgpt/gpt-5.4", custom_llm_provider="chatgpt"
        )

        assert model_info["input_cost_per_token"] == original_base_entry["input_cost_per_token"]
        assert model_info["output_cost_per_token"] == original_base_entry["output_cost_per_token"]
        assert model_info["litellm_provider"] == "chatgpt"
    finally:
        litellm.model_cost["chatgpt/gpt-5.4"] = original_chatgpt_entry
        _invalidate_model_cost_lowercase_map()


def test_chatgpt_deployment_alias_uses_stripped_base_model_pricing():
    _load_local_model_cost_map()

    model_info = litellm.get_model_info(
        model="chatgpt/gpt-5.4-1", custom_llm_provider="chatgpt"
    )
    base_model_info = litellm.get_model_info(
        model="chatgpt/gpt-5.4", custom_llm_provider="chatgpt"
    )

    assert model_info["input_cost_per_token"] == base_model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] == base_model_info["output_cost_per_token"]
    assert model_info["litellm_provider"] == "chatgpt"


def test_chatgpt_special_alias_reuses_openai_pricing():
    _load_local_model_cost_map()

    model_info = litellm.get_model_info(
        model="chatgpt/gpt-5.3-codex-spark", custom_llm_provider="chatgpt"
    )
    base_model_info = litellm.get_model_info(model="gpt-5.3-codex")

    assert model_info["input_cost_per_token"] == base_model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] == base_model_info["output_cost_per_token"]


def test_chatgpt_gpt_5_4_mini_exposes_cached_input_pricing():
    _load_local_model_cost_map()

    model_info = litellm.get_model_info(
        model="chatgpt/gpt-5.4-mini", custom_llm_provider="chatgpt"
    )

    assert model_info["input_cost_per_token"] == 7.5e-07
    assert model_info["cache_read_input_token_cost"] == 7.5e-08
    assert model_info["output_cost_per_token"] == 4.5e-06
    assert model_info["litellm_provider"] == "chatgpt"
    assert model_info["supports_prompt_caching"] is True


def test_chatgpt_gpt_5_4_mini_cached_input_cost_calculation():
    _load_local_model_cost_map()

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        cache_read_input_tokens=800,
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="chatgpt/gpt-5.4-mini",
        usage=usage,
        custom_llm_provider="chatgpt",
    )

    expected_prompt_cost = (200 * 7.5e-07) + (800 * 7.5e-08)
    expected_completion_cost = 100 * 4.5e-06

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_chatgpt_gpt_5_5_exposes_cached_input_pricing():
    _load_local_model_cost_map()

    model_info = litellm.get_model_info(
        model="chatgpt/gpt-5.5", custom_llm_provider="chatgpt"
    )

    assert model_info["input_cost_per_token"] == 5e-06
    assert model_info["cache_read_input_token_cost"] == 5e-07
    assert model_info["output_cost_per_token"] == 3e-05
    assert model_info["litellm_provider"] == "chatgpt"
    assert model_info["supports_prompt_caching"] is True


def test_chatgpt_gpt_5_5_cached_input_cost_calculation():
    _load_local_model_cost_map()

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        cache_read_input_tokens=800,
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="chatgpt/gpt-5.5",
        usage=usage,
        custom_llm_provider="chatgpt",
    )

    expected_prompt_cost = (200 * 5e-06) + (800 * 5e-07)
    expected_completion_cost = 100 * 3e-05

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


@pytest.mark.parametrize(
    ("model", "input_cost", "cache_read_cost", "output_cost"),
    [
        ("chatgpt/gpt-5.6-sol", 5e-6, 5e-7, 3e-5),
        ("chatgpt/gpt-5.6-terra", 2e-6, 2e-7, 1.2e-5),
        ("chatgpt/gpt-5.6-luna", 2e-7, 2e-8, 1.2e-6),
    ],
)
def test_chatgpt_gpt_5_6_cached_input_pricing(
    model: str,
    input_cost: float,
    cache_read_cost: float,
    output_cost: float,
):
    _load_local_model_cost_map()

    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="chatgpt",
    )
    assert model_info["input_cost_per_token"] == input_cost
    assert model_info["cache_read_input_token_cost"] == cache_read_cost
    assert model_info["output_cost_per_token"] == output_cost

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        cache_read_input_tokens=800,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="chatgpt",
    )

    expected_prompt_cost = (200 * input_cost) + (800 * cache_read_cost)
    expected_completion_cost = 100 * output_cost
    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_chatgpt_auto_review_uses_response_model_pricing():
    _load_local_model_cost_map()

    response = ResponsesAPIResponse(
        id="resp_auto_review",
        created_at=1700000000,
        model="gpt-5.6-luna",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
        usage=ResponseAPIUsage(
            input_tokens=10018,
            output_tokens=83,
            total_tokens=10101,
            input_tokens_details={"cached_tokens": 7680},
        ),
    )

    cost = completion_cost(
        completion_response=response,
        model="codex-auto-review",
        custom_llm_provider="chatgpt",
        optional_params={},
        call_type="responses",
    )

    expected_cost = (2338 * 2e-7) + (7680 * 2e-8) + (83 * 1.2e-6)
    assert round(cost, 12) == round(expected_cost, 12)


def test_chatgpt_gpt_5_4_priority_cost_calculation():
    _load_local_model_cost_map()

    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)

    prompt_cost, completion_cost = generic_cost_per_token(
        model="chatgpt/gpt-5.4",
        usage=usage,
        custom_llm_provider="chatgpt",
        service_tier="priority",
    )

    assert round(prompt_cost, 12) == round(1000 * 5e-06, 12)
    assert round(completion_cost, 12) == round(100 * 3e-05, 12)


def test_chatgpt_gpt_5_5_fast_alias_uses_priority_cost_calculation():
    _load_local_model_cost_map()

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        cache_read_input_tokens=800,
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="chatgpt/gpt-5.5",
        usage=usage,
        custom_llm_provider="chatgpt",
        service_tier="fast",
    )

    expected_prompt_cost = (200 * 1.25e-05) + (800 * 1.25e-06)
    expected_completion_cost = 100 * 7.5e-05

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_chatgpt_response_default_service_tier_overrides_priority_request_pricing():
    _load_local_model_cost_map()

    response = ResponsesAPIResponse(
        id="resp_test",
        created_at=1700000000,
        model="chatgpt/gpt-5.4",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
        usage=ResponseAPIUsage(input_tokens=1000, output_tokens=100, total_tokens=1100),
        service_tier="default",
    )

    cost = completion_cost(
        completion_response=response,
        model="chatgpt/gpt-5.4",
        custom_llm_provider="chatgpt",
        optional_params={"service_tier": "priority"},
        call_type="responses",
    )

    expected_standard_cost = (1000 * 2.5e-06) + (100 * 1.5e-05)
    assert round(cost, 12) == round(expected_standard_cost, 12)


def test_chatgpt_response_priority_service_tier_overrides_default_request_pricing():
    _load_local_model_cost_map()

    response = ResponsesAPIResponse(
        id="resp_test",
        created_at=1700000000,
        model="chatgpt/gpt-5.4",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
        usage=ResponseAPIUsage(input_tokens=1000, output_tokens=100, total_tokens=1100),
        service_tier="priority",
    )

    cost = completion_cost(
        completion_response=response,
        model="chatgpt/gpt-5.4",
        custom_llm_provider="chatgpt",
        optional_params={"service_tier": "default"},
        call_type="responses",
    )

    expected_priority_cost = (1000 * 5e-06) + (100 * 3e-05)
    assert round(cost, 12) == round(expected_priority_cost, 12)
