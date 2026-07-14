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
