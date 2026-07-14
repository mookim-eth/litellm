import math

import pytest

from litellm.responses.provider_headers_timeout import (
    RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG,
    apply_provider_headers_timeout_to_request,
)


def _request_data(requester_ip: str | None) -> dict:
    return {"litellm_metadata": {"requester_ip_address": requester_ip}}


@pytest.mark.parametrize(
    "requester_ip,allowlist",
    [
        ("203.0.113.8", ["203.0.113.8"]),
        ("203.0.113.8", ["203.0.113.0/24"]),
        ("2001:db8::8", ["2001:db8::/32"]),
    ],
)
def test_provider_headers_timeout_bypasses_allowlisted_ip(
    requester_ip: str, allowlist: list[str]
):
    data = _request_data(requester_ip)

    apply_provider_headers_timeout_to_request(
        data=data,
        general_settings={
            "responses_provider_headers_timeout_seconds": 110,
            "responses_provider_headers_timeout_ip_allowlist": allowlist,
        },
    )

    assert RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG not in data


@pytest.mark.parametrize("requester_ip", ["198.51.100.4", None, "not-an-ip"])
def test_provider_headers_timeout_applies_when_ip_is_not_allowlisted(
    requester_ip: str | None,
):
    data = _request_data(requester_ip)

    apply_provider_headers_timeout_to_request(
        data=data,
        general_settings={
            "responses_provider_headers_timeout_seconds": 110,
            "responses_provider_headers_timeout_ip_allowlist": [
                "203.0.113.0/24",
                "invalid-cidr",
            ],
        },
    )

    assert data[RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG] == 110.0


@pytest.mark.parametrize(
    "configured_timeout", [None, False, True, 0, -1, math.nan, "110"]
)
def test_provider_headers_timeout_ignores_missing_or_invalid_values(
    configured_timeout,
):
    data = _request_data("198.51.100.4")
    data[RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG] = 1

    apply_provider_headers_timeout_to_request(
        data=data,
        general_settings={
            "responses_provider_headers_timeout_seconds": configured_timeout,
        },
    )

    assert RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG not in data


def test_provider_headers_timeout_cannot_be_set_by_request():
    data = _request_data("203.0.113.8")
    data[RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG] = 1

    apply_provider_headers_timeout_to_request(
        data=data,
        general_settings={
            "responses_provider_headers_timeout_seconds": 110,
            "responses_provider_headers_timeout_ip_allowlist": ["203.0.113.8"],
        },
    )

    assert RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG not in data
