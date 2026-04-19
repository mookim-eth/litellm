import copy

import litellm

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
