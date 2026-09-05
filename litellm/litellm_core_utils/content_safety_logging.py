"""Durable audit logging for upstream content-safety decisions.

The regular LiteLLM log stream is intentionally kept for operational debugging,
but it is backed by container stdout and is therefore subject to Docker log
rotation.  This module writes selected safety events to a separate JSONL file
when ``LITELLM_CONTENT_SAFETY_LOG_PATH`` is configured.  The writer is best
effort: an unavailable audit destination must never change the request result.
"""

import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from litellm._logging import redact_secrets
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


_LOGGER_NAME = "LiteLLM Content Safety"
_HANDLER_LOCK = threading.Lock()
_HANDLER: Optional[WatchedFileHandler] = None
_HANDLER_PATH: Optional[str] = None


def _configured_path() -> Optional[str]:
    value = os.getenv("LITELLM_CONTENT_SAFETY_LOG_PATH", "").strip()
    return value or None


def _get_handler(path: str) -> Optional[WatchedFileHandler]:
    global _HANDLER, _HANDLER_PATH

    with _HANDLER_LOCK:
        if _HANDLER is not None and _HANDLER_PATH == path:
            return _HANDLER

        if _HANDLER is not None:
            try:
                _HANDLER.close()
            except Exception:
                pass
            _HANDLER = None
            _HANDLER_PATH = None

        try:
            log_path = Path(path)
            log_path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
            handler = WatchedFileHandler(
                log_path, mode="a", encoding="utf-8", delay=True
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _HANDLER = handler
            _HANDLER_PATH = path
            return handler
        except (OSError, ValueError):
            return None


def write_content_safety_event(
    *,
    event_type: str,
    request_id: Optional[str],
    model: Optional[str],
    reason: Optional[str] = None,
    request_input: Any = None,
    provider: Optional[str] = None,
    status_code: Optional[int] = None,
    key_alias: Optional[str] = None,
    user_id: Optional[str] = None,
    upstream_event: Any = None,
    upstream_error: Optional[str] = None,
) -> bool:
    """Append one durable content-safety event, returning whether it was written."""

    path = _configured_path()
    if path is None:
        return False

    event: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "request_id": request_id,
        "model": model,
        "reason": reason,
        "provider": provider,
        "status_code": status_code,
    }
    if key_alias is not None:
        event["key_alias"] = key_alias
    if user_id is not None:
        event["user_id"] = user_id
    if request_input is not None:
        event["request_input"] = request_input
    if upstream_event is not None:
        event["upstream_event"] = upstream_event
    if upstream_error is not None:
        event["upstream_error"] = upstream_error

    # Redact credential-shaped values while preserving the requested prompt
    # context.  safe_dumps also handles Pydantic objects and circular values.
    payload = redact_secrets(safe_dumps(event))
    handler = _get_handler(path)
    if handler is None:
        return False

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if handler not in logger.handlers:
        logger.handlers.clear()
        logger.addHandler(handler)

    try:
        logger.info(payload)
        # Keep the audit file private even when the process umask is permissive.
        try:
            os.chmod(path, 0o640)
        except OSError:
            pass
        return True
    except Exception:
        return False


def close_content_safety_log() -> None:
    """Close the current handler (used by graceful shutdown and tests)."""

    global _HANDLER, _HANDLER_PATH
    with _HANDLER_LOCK:
        if _HANDLER is not None:
            try:
                _HANDLER.close()
            except Exception:
                pass
        _HANDLER = None
        _HANDLER_PATH = None
