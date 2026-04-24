import copy

import litellm

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
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
