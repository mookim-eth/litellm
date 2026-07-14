import os
import time
from typing import Any, Dict, List, Optional

from aiohttp import TraceConfig

from litellm._logging import verbose_proxy_logger


def _str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


def _outbound_trace_enabled() -> bool:
    return _str_to_bool(os.getenv("LITELLM_OUTBOUND_TRACE_ENABLED", "False"))


def _outbound_trace_min_duration_ms() -> float:
    try:
        return max(0.0, float(os.getenv("LITELLM_OUTBOUND_TRACE_MIN_DURATION_MS", "0")))
    except ValueError:
        return 0.0


def _ctx_data(trace_config_ctx: Any) -> Dict[str, Any]:
    data = getattr(trace_config_ctx, "litellm_outbound_trace", None)
    if not isinstance(data, dict):
        data = {"started_at": time.monotonic(), "durations_ms": {}}
        setattr(trace_config_ctx, "litellm_outbound_trace", data)
    return data


def _mark_start(trace_config_ctx: Any, key: str) -> None:
    _ctx_data(trace_config_ctx)[f"{key}_started_at"] = time.monotonic()


def _mark_end(trace_config_ctx: Any, key: str) -> None:
    data = _ctx_data(trace_config_ctx)
    started_at = data.pop(f"{key}_started_at", None)
    if started_at is None:
        return
    durations = data.setdefault("durations_ms", {})
    durations[key] = round((time.monotonic() - started_at) * 1000, 2)


def _request_ctx(trace_config_ctx: Any) -> Dict[str, Any]:
    raw = getattr(trace_config_ctx, "trace_request_ctx", None)
    return raw if isinstance(raw, dict) else {}


def _finalize_request_body_timings(trace_config_ctx: Any) -> None:
    """Split request upload time from provider wait-for-headers time."""
    data = _ctx_data(trace_config_ctx)
    first_chunk_sent_at = data.get("request_body_first_chunk_sent_at")
    last_chunk_sent_at = data.get("request_body_last_chunk_sent_at")
    if not isinstance(first_chunk_sent_at, (int, float)) or not isinstance(
        last_chunk_sent_at, (int, float)
    ):
        return
    now = time.monotonic()
    durations = data.setdefault("durations_ms", {})
    durations["request_body_upload"] = round(
        max(0.0, last_chunk_sent_at - first_chunk_sent_at) * 1000, 2
    )
    durations["body_sent_to_headers"] = round(
        max(0.0, now - last_chunk_sent_at) * 1000, 2
    )


def _safe_url_fields(params: Any) -> Dict[str, Optional[str]]:
    url = getattr(params, "url", None)
    if url is None:
        return {"host": None, "path": None, "scheme": None}
    return {
        "host": getattr(url, "host", None),
        "path": getattr(url, "path", None),
        "scheme": getattr(url, "scheme", None),
    }


def _log_trace(trace_config_ctx: Any, params: Any, *, error: Optional[str] = None) -> None:
    data = _ctx_data(trace_config_ctx)
    total_ms = round((time.monotonic() - data["started_at"]) * 1000, 2)
    if error is None and total_ms < _outbound_trace_min_duration_ms():
        return

    request_ctx = _request_ctx(trace_config_ctx)
    url_fields = _safe_url_fields(params)
    response = getattr(params, "response", None)
    status_code = getattr(response, "status", None)

    verbose_proxy_logger.warning(
        "litellm_outbound_trace method=%s scheme=%s host=%s path=%s "
        "status=%s stream=%s total_ms=%s durations_ms=%s "
        "connection_reused=%s call_id=%s model=%s error=%s",
        getattr(params, "method", None),
        url_fields["scheme"],
        url_fields["host"],
        url_fields["path"],
        status_code,
        request_ctx.get("stream"),
        total_ms,
        data.get("durations_ms", {}),
        data.get("connection_reused", False),
        request_ctx.get("call_id"),
        request_ctx.get("model"),
        error,
    )


def create_outbound_trace_configs() -> Optional[List[TraceConfig]]:  # noqa: PLR0915
    if not _outbound_trace_enabled():
        return None

    trace_config = TraceConfig()

    async def on_request_start(session: Any, trace_config_ctx: Any, params: Any) -> None:
        data = _ctx_data(trace_config_ctx)
        data["started_at"] = time.monotonic()
        _mark_start(trace_config_ctx, "request_headers")
        _mark_start(trace_config_ctx, "request_body_first_chunk")

    async def on_connection_queued_start(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_start(trace_config_ctx, "connection_queued")

    async def on_connection_queued_end(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_end(trace_config_ctx, "connection_queued")

    async def on_dns_resolvehost_start(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_start(trace_config_ctx, "dns")

    async def on_dns_resolvehost_end(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_end(trace_config_ctx, "dns")

    async def on_connection_create_start(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_start(trace_config_ctx, "connection_create")

    async def on_connection_create_end(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_end(trace_config_ctx, "connection_create")

    async def on_connection_reuseconn(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _ctx_data(trace_config_ctx)["connection_reused"] = True

    async def on_request_headers_sent(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _mark_end(trace_config_ctx, "request_headers")

    async def on_request_chunk_sent(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        data = _ctx_data(trace_config_ctx)
        now = time.monotonic()
        if not data.get("first_request_chunk_sent"):
            data["first_request_chunk_sent"] = True
            _mark_end(trace_config_ctx, "request_body_first_chunk")
            data["request_body_first_chunk_sent_at"] = now
        data["request_body_last_chunk_sent_at"] = now

    async def on_request_end(session: Any, trace_config_ctx: Any, params: Any) -> None:
        _finalize_request_body_timings(trace_config_ctx)
        _log_trace(trace_config_ctx, params)

    async def on_request_exception(
        session: Any, trace_config_ctx: Any, params: Any
    ) -> None:
        _finalize_request_body_timings(trace_config_ctx)
        exception = getattr(params, "exception", None)
        error = type(exception).__name__ if exception is not None else "unknown"
        _log_trace(trace_config_ctx, params, error=error)

    trace_config.on_request_start.append(on_request_start)
    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_exception.append(on_request_exception)
    return [trace_config]
