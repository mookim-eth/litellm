import ipaddress
import math
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

from litellm._logging import verbose_proxy_logger

RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG = (
    "_responses_provider_headers_timeout_seconds"
)

IPAddressNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@lru_cache(maxsize=64)
def _parse_ip_allowlist(entries: Tuple[str, ...]) -> Tuple[IPAddressNetwork, ...]:
    networks = []
    for entry in entries:
        try:
            networks.append(ipaddress.ip_network(entry.strip(), strict=False))
        except ValueError:
            verbose_proxy_logger.warning(
                "Invalid entry in responses_provider_headers_timeout_ip_allowlist: %s; skipping",
                entry,
            )
    return tuple(networks)


def _is_ip_allowlisted(client_ip: Optional[str], allowlist: Any) -> bool:
    if not client_ip or not isinstance(allowlist, (list, tuple)):
        return False
    try:
        address = ipaddress.ip_address(client_ip.split(",", 1)[0].strip())
    except ValueError:
        return False
    entries = tuple(entry for entry in allowlist if isinstance(entry, str))
    return any(address in network for network in _parse_ip_allowlist(entries))


def apply_provider_headers_timeout_to_request(
    *, data: Dict[str, Any], general_settings: Dict[str, Any]
) -> None:
    """Add the server-controlled timeout to a Responses request unless IP-exempt."""
    data.pop(RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG, None)
    configured_timeout = general_settings.get(
        "responses_provider_headers_timeout_seconds"
    )
    if isinstance(configured_timeout, bool) or not isinstance(
        configured_timeout, (int, float)
    ):
        return
    timeout_seconds = float(configured_timeout)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        verbose_proxy_logger.warning(
            "Ignoring invalid responses_provider_headers_timeout_seconds=%r; expected a positive finite number",
            configured_timeout,
        )
        return

    metadata = data.get("litellm_metadata")
    requester_ip = (
        metadata.get("requester_ip_address") if isinstance(metadata, dict) else None
    )
    allowlist = general_settings.get(
        "responses_provider_headers_timeout_ip_allowlist", []
    )
    if _is_ip_allowlisted(requester_ip, allowlist):
        verbose_proxy_logger.debug(
            "Bypassing Responses provider headers timeout for allowlisted requester IP=%s",
            requester_ip,
        )
        return

    data[RESPONSES_PROVIDER_HEADERS_TIMEOUT_KWARG] = timeout_seconds
