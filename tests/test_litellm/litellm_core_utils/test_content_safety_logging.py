import json

from litellm.litellm_core_utils.content_safety_logging import (
    close_content_safety_log,
    write_content_safety_event,
)


def test_should_write_content_safety_event_as_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "content-safety.jsonl"
    monkeypatch.setenv("LITELLM_CONTENT_SAFETY_LOG_PATH", str(path))
    close_content_safety_log()

    assert write_content_safety_event(
        event_type="biological_risk",
        request_id="request-1",
        model="gpt-5.6-sol",
        reason="biological_risk",
        request_input="investigate this request",
        upstream_error="This content was flagged for possible biological risk.",
    )

    record = json.loads(path.read_text())
    assert record["event_type"] == "biological_risk"
    assert record["request_id"] == "request-1"
    assert record["request_input"] == "investigate this request"
    assert record["upstream_error"].startswith("This content was flagged")


def test_should_not_write_when_audit_path_is_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_CONTENT_SAFETY_LOG_PATH", raising=False)
    close_content_safety_log()

    assert not write_content_safety_event(
        event_type="content_filter",
        request_id="request-2",
        model="gpt-5.6-sol",
        reason="content_filter",
        request_input="blocked input",
    )
    assert list(tmp_path.iterdir()) == []
