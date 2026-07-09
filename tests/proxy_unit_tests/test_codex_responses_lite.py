from types import SimpleNamespace

from starlette.datastructures import Headers

from litellm.proxy.response_api_endpoints.endpoints import (
    CODEX_RESPONSES_LITE_HEADER,
    _apply_codex_responses_lite_request_overrides,
)


def test_codex_responses_lite_overrides_add_header_and_disables_parallel_tools():
    data = {"model": "gpt-5.6-sol", "parallel_tool_calls": True}
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert data["extra_headers"] == {CODEX_RESPONSES_LITE_HEADER: "true"}
    assert data["parallel_tool_calls"] is False


def test_forward_codex_responses_lite_header_preserves_existing_extra_headers():
    data = {
        "model": "gpt-5.6-sol",
        "extra_headers": {"x-existing": "value"},
    }
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert data["extra_headers"] == {
        "x-existing": "value",
        CODEX_RESPONSES_LITE_HEADER: "true",
    }
    assert data["parallel_tool_calls"] is False


def test_forward_codex_responses_lite_header_ignores_missing_header():
    data = {"model": "gpt-5.5"}
    request = SimpleNamespace(headers=Headers({}))

    _apply_codex_responses_lite_request_overrides(data=data, request=request)

    assert "extra_headers" not in data
    assert "parallel_tool_calls" not in data
