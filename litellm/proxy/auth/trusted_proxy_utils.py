import ipaddress
from typing import Any, Dict, List, Optional, Union

from fastapi import Request

from litellm._logging import verbose_proxy_logger

TRUSTED_PROXY_RANGES_KEY = "trusted_proxy_ranges"
TrustedProxyNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def _normalize_cidr_ranges(configured_ranges: Any, *, setting_name: str) -> List[str]:
    if not configured_ranges:
        return []
    if isinstance(configured_ranges, str):
        return [item.strip() for item in configured_ranges.split(",") if item.strip()]
    if isinstance(configured_ranges, (list, tuple, set)):
        return [str(item).strip() for item in configured_ranges if str(item).strip()]
    verbose_proxy_logger.warning(
        "Invalid %s value: expected CIDR ranges, got %s",
        setting_name,
        type(configured_ranges).__name__,
    )
    return []


def parse_trusted_proxy_ranges(
    configured_ranges: Any,
    *,
    setting_name: str = TRUSTED_PROXY_RANGES_KEY,
) -> List[TrustedProxyNetwork]:
    networks: List[TrustedProxyNetwork] = []
    for cidr in _normalize_cidr_ranges(configured_ranges, setting_name=setting_name):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            verbose_proxy_logger.warning(
                "Invalid CIDR in %s: %s, skipping", setting_name, cidr
            )
    return networks


def require_trusted_proxy_request(
    *,
    request: Request,
    general_settings: Optional[Dict[str, Any]],
    feature_name: str,
) -> None:
    """Trust identity headers only when the direct TCP peer is allowlisted."""
    settings = general_settings or {}
    trusted_networks = parse_trusted_proxy_ranges(
        settings.get(TRUSTED_PROXY_RANGES_KEY)
    )
    if not trusted_networks:
        raise ValueError(
            f"{feature_name} requires general_settings.{TRUSTED_PROXY_RANGES_KEY} "
            "before trusting identity headers."
        )

    client = getattr(request, "client", None)
    client_host = getattr(client, "host", None)
    try:
        client_ip = ipaddress.ip_address(str(client_host).strip())
    except ValueError as exc:
        raise ValueError(
            f"{feature_name} received identity headers from an invalid direct client IP."
        ) from exc
    if not any(client_ip in network for network in trusted_networks):
        raise ValueError(
            f"{feature_name} only accepts identity headers from configured trusted "
            f"proxy ranges. Direct client IP {client_host!r} is not trusted."
        )
