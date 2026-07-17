from unittest.mock import MagicMock, patch

import pytest

from litellm.litellm_core_utils.url_utils import (
    SSRFError,
    assert_same_origin,
    safe_get,
    validate_url,
)
from litellm.llms.black_forest_labs.common_utils import (
    BlackForestLabsError,
    assert_bfl_polling_url,
)


def test_same_origin_normalizes_default_port():
    assert_same_origin(
        "https://api.example.test:443/jobs/1",
        "https://API.EXAMPLE.TEST/v1",
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "http://api.example.test/jobs/1",
        "https://evil.example.test/jobs/1",
        "https://api.example.test:8443/jobs/1",
    ],
)
def test_same_origin_rejects_changed_destination(candidate):
    with pytest.raises(SSRFError):
        assert_same_origin(candidate, "https://api.example.test/v1")


def test_validate_url_rejects_private_resolution():
    with patch(
        "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("127.0.0.1", 80))],
    ):
        with pytest.raises(SSRFError):
            validate_url("http://example.test/private")


def test_safe_get_preserves_headers_across_redirects():
    redirect = MagicMock(is_redirect=True, headers={"location": "/next"})
    complete = MagicMock(is_redirect=False)
    client = MagicMock()
    client.get.side_effect = [redirect, complete]
    public_dns = [(2, 1, 6, "", ("93.184.216.34", 80))]

    with patch(
        "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
        return_value=public_dns,
    ):
        safe_get(client, "http://example.test/start", headers={"X-Test": "value"})

    assert client.get.call_args_list[1].kwargs["headers"]["X-Test"] == "value"


def test_bfl_polling_url_accepts_owned_subdomain_only():
    assert_bfl_polling_url("https://gateway.bfl.ai/v1/get_result?id=1")
    with pytest.raises(BlackForestLabsError):
        assert_bfl_polling_url("https://bfl.ai.attacker.test/steal")
