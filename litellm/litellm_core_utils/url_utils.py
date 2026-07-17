"""SSRF-safe helpers for URLs influenced by proxy callers or providers."""

import socket
from ipaddress import ip_address, ip_network
from typing import Any, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse, urlunparse

import httpx

import litellm

_ALLOWED_SCHEMES = ("http", "https")
_CLOUD_METADATA_EXCEPTIONS = (ip_network("168.63.129.16/32"),)
_MAX_REDIRECTS = 10


class SSRFError(ValueError):
    """Raised when a URL cannot be proven safe for a server-side fetch."""


def encode_url_path_segment(value: Any, *, field_name: str = "path parameter") -> str:
    if value is None or str(value) == "":
        raise ValueError(f"{field_name} is required")
    value_str = str(value)
    if value_str in {".", ".."}:
        raise ValueError(f"{field_name} cannot be a dot path segment")
    return quote(value_str, safe="")


def encode_url_path_segments(value: Any, *, field_name: str = "path") -> str:
    if value is None or str(value) == "":
        raise ValueError(f"{field_name} is required")
    return "/".join(
        encode_url_path_segment(segment, field_name=field_name)
        for segment in str(value).split("/")
    )


def _normalize_host(host: str) -> str:
    return host.lower().rstrip(".")


def _is_blocked_ip(address: str) -> bool:
    try:
        parsed = ip_address(address)
    except ValueError:
        return True
    if parsed.version == 6 and getattr(parsed, "ipv4_mapped", None):
        parsed = parsed.ipv4_mapped
    return (
        not parsed.is_global
        or parsed.is_multicast
        or any(parsed in network for network in _CLOUD_METADATA_EXCEPTIONS)
    )


def _effective_port(scheme: str, port: Optional[int]) -> int:
    return port if port is not None else (443 if scheme == "https" else 80)


def _format_host_header(hostname: str, port: int, default_port: int) -> str:
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    return bracketed if port == default_port else f"{bracketed}:{port}"


def _is_host_allowlisted(hostname: str, port: int) -> bool:
    configured: List[str] = getattr(litellm, "user_url_allowed_hosts", []) or []
    normalized = _normalize_host(hostname)
    host_repr = f"[{normalized}]" if ":" in normalized else normalized
    candidates: Set[str] = {host_repr, f"{host_repr}:{port}"}
    return bool(candidates & {_normalize_host(item) for item in configured if item})


def validate_url(url: str) -> Tuple[str, str]:
    """Resolve once, reject internal addresses, and pin plaintext HTTP to the IP."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError("URL scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise SSRFError("URL userinfo is not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise SSRFError("URL has an invalid port") from exc
    port = _effective_port(parsed.scheme, parsed_port)
    try:
        addresses = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError("DNS resolution failed") from exc
    if not addresses:
        raise SSRFError("DNS resolution returned no addresses")

    if not _is_host_allowlisted(hostname, port):
        for address in addresses:
            resolved = address[4][0]
            if not isinstance(resolved, str) or _is_blocked_ip(resolved):
                raise SSRFError("URL targets a blocked address")

    default_port = 443 if parsed.scheme == "https" else 80
    host_header = _format_host_header(hostname, port, default_port)
    if parsed.scheme == "https" and getattr(litellm, "ssl_verify", True) is not False:
        return url, host_header

    resolved_ip = addresses[0][4][0]
    if not isinstance(resolved_ip, str):
        raise SSRFError("DNS returned an invalid address")
    ip_host = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
    netloc = ip_host if parsed_port is None else f"{ip_host}:{parsed_port}"
    return (
        urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, "")),
        host_header,
    )


def assert_same_origin(candidate_url: str, expected_url: str) -> None:
    """Require scheme, normalized host, and effective port to match."""
    candidate = urlparse(candidate_url)
    expected = urlparse(expected_url)
    if candidate.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError("URL scheme is not allowed")
    if candidate.username is not None or candidate.password is not None:
        raise SSRFError("URL userinfo is not allowed")
    if candidate.scheme != expected.scheme:
        raise SSRFError("Origin scheme mismatch")
    if _normalize_host(candidate.hostname or "") != _normalize_host(
        expected.hostname or ""
    ):
        raise SSRFError("Origin host mismatch")
    try:
        candidate_port = candidate.port
        expected_port = expected.port
    except ValueError as exc:
        raise SSRFError("Origin URL has an invalid port") from exc
    if _effective_port(candidate.scheme, candidate_port) != _effective_port(
        expected.scheme, expected_port
    ):
        raise SSRFError("Origin port mismatch")


def is_url_destination_allowed_by_host(url: str, allowed_hosts: List[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    host = _normalize_host(parsed.hostname)
    try:
        port = _effective_port(parsed.scheme, parsed.port)
    except ValueError:
        return False
    for entry in allowed_hosts or []:
        if not isinstance(entry, str):
            continue
        configured = urlparse(entry if "://" in entry else f"//{entry}")
        if _normalize_host(configured.hostname or "") != host:
            continue
        if configured.scheme and configured.scheme != parsed.scheme:
            continue
        try:
            configured_port = configured.port
        except ValueError:
            continue
        if configured.scheme and configured_port is None:
            configured_port = _effective_port(configured.scheme, None)
        if configured_port is not None and configured_port != port:
            continue
        return True
    return False


def safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    if getattr(litellm, "user_url_validation", True) is False:
        return client.get(url, follow_redirects=True, **kwargs)
    kwargs.pop("follow_redirects", None)
    base_headers = dict(kwargs.pop("headers", {}) or {})
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validated, host = validate_url(current)
        headers = dict(base_headers)
        headers["Host"] = host
        response = client.get(validated, headers=headers, follow_redirects=False, **kwargs)
        if not response.is_redirect:
            return response
        location = response.headers.get("location")
        if not location:
            raise SSRFError("Redirect response has no Location header")
        current = str(httpx.URL(current).join(location))
    raise SSRFError("Too many redirects")


async def async_safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    if getattr(litellm, "user_url_validation", True) is False:
        return await client.get(url, follow_redirects=True, **kwargs)
    kwargs.pop("follow_redirects", None)
    base_headers = dict(kwargs.pop("headers", {}) or {})
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validated, host = validate_url(current)
        headers = dict(base_headers)
        headers["Host"] = host
        response = await client.get(
            validated, headers=headers, follow_redirects=False, **kwargs
        )
        if not response.is_redirect:
            return response
        location = response.headers.get("location")
        if not location:
            raise SSRFError("Redirect response has no Location header")
        current = str(httpx.URL(current).join(location))
    raise SSRFError("Too many redirects")
