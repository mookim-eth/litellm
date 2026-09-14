import json
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from starlette.responses import StreamingResponse


def _as_error_mapping(error: Any) -> Mapping[str, Any]:
    """Extract an OpenAI-style error object without exposing exception internals."""
    value: Any = error
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("event:"):
            candidate = "\n".join(
                line.removeprefix("data: ")
                for line in candidate.splitlines()
                if line.startswith("data: ")
            )
        elif candidate.startswith("data: "):
            candidate = candidate.removeprefix("data: ").strip()
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return {"message": error}

    if isinstance(value, Mapping):
        nested_error = value.get("error")
        if isinstance(nested_error, Mapping):
            return nested_error
        return value

    detail = getattr(value, "detail", None)
    if isinstance(detail, Mapping):
        nested_error = detail.get("error")
        if isinstance(nested_error, Mapping):
            return nested_error
        return detail

    return {
        "code": getattr(value, "code", None) or getattr(value, "status_code", None),
        "message": getattr(value, "message", None) or detail or str(value),
        "param": getattr(value, "param", None),
    }


def _string_value(value: Any, *, default: str) -> str:
    if value is None or value == "None":
        return default
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def build_responses_error_event(
    error: Any, *, sequence_number: int = 0
) -> Dict[str, Any]:
    """Build the top-level ``error`` event defined by the Responses API."""
    error_mapping = _as_error_mapping(error)
    param = error_mapping.get("param")
    if param == "None":
        param = None
    elif param is not None:
        param = _string_value(param, default="")
    return {
        "type": "error",
        "code": _string_value(error_mapping.get("code"), default="server_error"),
        "message": _string_value(
            error_mapping.get("message") or error_mapping.get("error"),
            default="An unknown error occurred.",
        ),
        "param": param,
        "sequence_number": sequence_number,
    }


def serialize_responses_error_event(error: Any, *, sequence_number: int = 0) -> str:
    event = build_responses_error_event(error, sequence_number=sequence_number)
    return (
        "event: error\n"
        f"data: {json.dumps(event, separators=(',', ':'), ensure_ascii=False)}\n\n"
    )


def responses_error_streaming_response(
    error: Any,
    *,
    headers: Optional[Mapping[str, str]] = None,
    sequence_number: int = 0,
) -> StreamingResponse:
    """Return a terminal, spec-compliant Responses error SSE stream."""
    body = serialize_responses_error_event(
        error, sequence_number=sequence_number
    ).encode("utf-8")

    async def _body():
        yield body

    response_headers = {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() not in {"content-length", "content-type"}
    }
    response_headers["Cache-Control"] = "no-cache"
    response_headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(
        _body(),
        status_code=200,
        media_type="text/event-stream",
        headers=response_headers,
    )
