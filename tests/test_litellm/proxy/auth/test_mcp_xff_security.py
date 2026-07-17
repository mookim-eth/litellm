from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.auth.ip_address_utils import IPAddressUtils


def _request(xff):
    request = MagicMock()
    request.client.host = "10.0.0.10"
    request.headers = {"x-forwarded-for": xff}
    return request


def test_xff_hop_count_ignores_attacker_prepended_internal_ip():
    result = IPAddressUtils.get_mcp_client_ip(
        _request("10.0.0.99, 203.0.113.9"),
        {
            "use_x_forwarded_for": True,
            "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            "mcp_xff_num_trusted_hops": 1,
        },
    )
    assert result == "203.0.113.9"


@pytest.mark.parametrize("value", [0, -1, "bad", 1.5])
def test_invalid_xff_hop_count_fails_closed(value):
    assert (
        IPAddressUtils.get_mcp_client_ip(
            _request("10.0.0.99, 203.0.113.9"),
            {
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
                "mcp_xff_num_trusted_hops": value,
            },
        )
        == ""
    )


def test_config_rejects_non_positive_hop_count():
    with pytest.raises(ValidationError):
        ConfigGeneralSettings(mcp_xff_num_trusted_hops=0)
