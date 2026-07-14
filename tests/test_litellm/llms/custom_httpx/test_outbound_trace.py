from types import SimpleNamespace
from unittest.mock import patch

import pytest

from litellm.llms.custom_httpx.outbound_trace import create_outbound_trace_configs


def test_outbound_trace_configs_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_OUTBOUND_TRACE_ENABLED", raising=False)

    assert create_outbound_trace_configs() is None


def test_outbound_trace_configs_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_OUTBOUND_TRACE_ENABLED", "true")

    trace_configs = create_outbound_trace_configs()

    assert trace_configs is not None
    assert len(trace_configs) == 1
    assert len(trace_configs[0].on_connection_queued_start) == 1
    assert len(trace_configs[0].on_dns_resolvehost_start) == 1
    assert len(trace_configs[0].on_connection_create_start) == 1
    assert len(trace_configs[0].on_connection_reuseconn) == 1
    assert len(trace_configs[0].on_request_end) == 1


@pytest.mark.asyncio
async def test_outbound_trace_splits_body_upload_from_header_wait(monkeypatch):
    monkeypatch.setenv("LITELLM_OUTBOUND_TRACE_ENABLED", "true")
    monkeypatch.setenv("LITELLM_OUTBOUND_TRACE_MIN_DURATION_MS", "0")
    trace_config = create_outbound_trace_configs()[0]
    trace_config_ctx = SimpleNamespace(
        trace_request_ctx={
            "stream": True,
            "call_id": "trace-call-id",
            "model": "test-model",
        }
    )
    params = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(scheme="https", host="provider.test", path="/v1"),
        response=SimpleNamespace(status=200),
    )

    await trace_config.on_request_start[0](None, trace_config_ctx, params)
    await trace_config.on_request_chunk_sent[0](None, trace_config_ctx, params)
    await trace_config.on_request_chunk_sent[0](None, trace_config_ctx, params)
    with patch(
        "litellm.llms.custom_httpx.outbound_trace.verbose_proxy_logger"
    ) as logger:
        await trace_config.on_request_end[0](None, trace_config_ctx, params)

    durations = trace_config_ctx.litellm_outbound_trace["durations_ms"]
    assert durations["request_body_upload"] >= 0
    assert durations["body_sent_to_headers"] >= 0
    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[10] == "trace-call-id"
