from typing import Any, Optional


def _get_usage_value(usage_obj: Any, key: str) -> Any:
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        return usage_obj.get(key)
    return getattr(usage_obj, key, None)


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed_value = int(value)
        except ValueError:
            return None
        return parsed_value if parsed_value >= 0 else None
    return None


def get_cache_read_input_tokens_from_usage(usage_obj: Any) -> int:
    """
    Extract cache-read tokens from a LiteLLM/OpenAI-compatible usage object.

    LiteLLM's daily usage tables track cache reads in `cache_read_input_tokens`.
    OpenAI-compatible backends, including ChatGPT-style backends, often return the
    same value as `prompt_tokens_details.cached_tokens` instead. Prefer an
    explicit positive top-level value when present, otherwise fall back to the
    nested OpenAI field.
    """
    top_level_cache_read_tokens = _coerce_non_negative_int(
        _get_usage_value(usage_obj, "cache_read_input_tokens")
    )
    if top_level_cache_read_tokens is not None and top_level_cache_read_tokens > 0:
        return top_level_cache_read_tokens

    prompt_tokens_details = _get_usage_value(usage_obj, "prompt_tokens_details")
    cached_tokens = _coerce_non_negative_int(
        _get_usage_value(prompt_tokens_details, "cached_tokens")
    )
    if cached_tokens is not None and cached_tokens > 0:
        return cached_tokens

    return top_level_cache_read_tokens or 0


def normalize_usage_dict_for_cache_read_tokens(usage_dict: dict) -> dict:
    """
    Return a usage dict with `cache_read_input_tokens` populated when the source
    only provided `prompt_tokens_details.cached_tokens`.

    The original dict is returned unchanged unless a positive derived value needs
    to be added, preserving existing exact-shape behavior for usage objects with
    no cache-read tokens.
    """
    cache_read_input_tokens = get_cache_read_input_tokens_from_usage(usage_dict)
    existing_cache_read_input_tokens = _coerce_non_negative_int(
        usage_dict.get("cache_read_input_tokens")
    )

    if (
        cache_read_input_tokens > 0
        and existing_cache_read_input_tokens != cache_read_input_tokens
    ):
        usage_dict = dict(usage_dict)
        usage_dict["cache_read_input_tokens"] = cache_read_input_tokens

    return usage_dict
