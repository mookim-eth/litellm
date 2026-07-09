from types import SimpleNamespace

from starlette.datastructures import Headers

from litellm.proxy.response_api_endpoints.endpoints import (
    CODEX_RESPONSES_LITE_HEADER,
    _forward_codex_responses_lite_header,
)


def test_forward_codex_responses_lite_header_adds_extra_headers():
    data = {"model": "gpt-5.6-sol"}
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _forward_codex_responses_lite_header(data=data, request=request)

    assert data["extra_headers"] == {CODEX_RESPONSES_LITE_HEADER: "true"}


def test_forward_codex_responses_lite_header_preserves_existing_extra_headers():
    data = {
        "model": "gpt-5.6-sol",
        "extra_headers": {"x-existing": "value"},
    }
    request = SimpleNamespace(
        headers=Headers({CODEX_RESPONSES_LITE_HEADER: "true"})
    )

    _forward_codex_responses_lite_header(data=data, request=request)

    assert data["extra_headers"] == {
        "x-existing": "value",
        CODEX_RESPONSES_LITE_HEADER: "true",
    }


def test_forward_codex_responses_lite_header_ignores_missing_header():
    data = {"model": "gpt-5.5"}
    request = SimpleNamespace(headers=Headers({}))

    _forward_codex_responses_lite_header(data=data, request=request)

    assert "extra_headers" not in data
