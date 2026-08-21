import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _reject_url_valued_destinations,
    _strip_client_pricing_overrides,
    _strip_untrusted_request_controls,
)


def test_untrusted_proxy_control_fields_are_removed():
    data = {
        "proxy_server_request": {"body": {"forged": True}},
        "mock_response": "free response",
        "_litellm_proxy_max_parallel_request_lease": {
            "counter_keys": ["{api_key:attacker-chosen}:max_parallel_requests"],
            "released": False,
        },
        "metadata": {
            "user_api_key_metadata": {"allow_client_tags": True},
            "disable_global_guardrails": True,
        },
    }

    _strip_untrusted_request_controls(data, UserAPIKeyAuth())

    assert "proxy_server_request" not in data
    assert "mock_response" not in data
    assert "_litellm_proxy_max_parallel_request_lease" not in data
    assert data["metadata"] == {}


def test_client_pricing_and_model_info_are_removed():
    data = {
        "input_cost_per_token": 0,
        "metadata": {"model_info": {"input_cost_per_token": 0}},
    }

    _strip_client_pricing_overrides(data)

    assert "input_cost_per_token" not in data
    assert data["metadata"] == {}


@pytest.mark.parametrize("field", ["model", "file_id"])
def test_url_valued_provider_destinations_are_rejected(field):
    with pytest.raises(HTTPException) as exc_info:
        _reject_url_valued_destinations({field: "http://127.0.0.1/admin"})

    assert exc_info.value.status_code == 400
