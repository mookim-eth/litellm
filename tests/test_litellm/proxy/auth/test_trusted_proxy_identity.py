from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request


def _request(client_ip: str) -> MagicMock:
    request = MagicMock()
    request.client = SimpleNamespace(host=client_ip)
    request.headers = {"x-auth-user": "user@example.com"}
    return request


def test_should_accept_identity_headers_from_trusted_direct_peer():
    require_trusted_proxy_request(
        request=_request("172.20.0.5"),
        general_settings={"trusted_proxy_ranges": ["172.20.0.0/16"]},
        feature_name="test identity auth",
    )


@pytest.mark.parametrize(
    "client_ip,settings",
    [
        ("203.0.113.8", {"trusted_proxy_ranges": ["172.20.0.0/16"]}),
        ("172.20.0.5", {}),
        ("not-an-ip", {"trusted_proxy_ranges": ["172.20.0.0/16"]}),
    ],
)
def test_should_reject_untrusted_or_unconfigured_identity_headers(
    client_ip, settings
):
    with pytest.raises(ValueError):
        require_trusted_proxy_request(
            request=_request(client_ip),
            general_settings=settings,
            feature_name="test identity auth",
        )


@pytest.mark.asyncio
async def test_should_gate_oauth2_proxy_auth_before_consuming_identity_header():
    from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "trusted_proxy_ranges": ["172.20.0.0/16"],
            "oauth2_config_mappings": {"user_id": "x-auth-user"},
        },
    ):
        with pytest.raises(ValueError, match="not trusted"):
            await handle_oauth2_proxy_request(_request("203.0.113.8"))

        auth = await handle_oauth2_proxy_request(_request("172.20.0.5"))

    assert auth.user_id == "user@example.com"
