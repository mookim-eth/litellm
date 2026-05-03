"""Lightweight runtime diagnostics for LiteLLM proxy memory retention.

Enabled only when LITELLM_MEM_DEBUG is truthy. The signal handlers intentionally
avoid gc.get_objects() and object serialization; they collect queue sizes, task
name counts, process RSS/swap, and optional tracemalloc top frames.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import sys
import threading
import time
import tracemalloc
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


_installed = False
_dump_lock = threading.Lock()
_context_provider: Optional[Callable[[], Mapping[str, Any]]] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _stderr_line(message: str) -> None:
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_enabled() -> bool:
    return os.getenv("LITELLM_MEM_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _read_proc_status() -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                result[key] = value.strip()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _read_smaps_rollup() -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open("/proc/self/smaps_rollup", "r", encoding="utf-8") as rollup_file:
            for line in rollup_file:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                result[key] = value.strip()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _fd_count() -> Optional[int]:
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return None


def _safe_qsize(queue: Any) -> Optional[int]:
    if queue is None:
        return None
    try:
        return int(queue.qsize())
    except Exception:
        return None


def _safe_maxsize(queue: Any) -> Optional[int]:
    if queue is None:
        return None
    try:
        maxsize = getattr(queue, "maxsize", None)
        if maxsize is None:
            maxsize = getattr(queue, "_maxsize", None)
        return int(maxsize) if maxsize is not None else None
    except Exception:
        return None


def _task_coro_name(task: asyncio.Task[Any]) -> str:
    try:
        coro = task.get_coro()
    except Exception:
        return "<unknown>"

    module = getattr(coro, "__module__", None)
    qualname = getattr(coro, "__qualname__", None)
    if qualname:
        return f"{module}.{qualname}" if module else qualname

    code = getattr(coro, "cr_code", None) or getattr(coro, "ag_code", None)
    if code is not None:
        code_name = getattr(code, "co_name", "<unknown>")
        filename = getattr(code, "co_filename", "")
        return f"{filename}:{code_name}" if filename else code_name

    return type(coro).__name__


def _asyncio_task_summary(limit: int) -> Dict[str, Any]:
    loop = _event_loop
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            return {"loop_available": False}

    try:
        tasks = list(asyncio.all_tasks(loop))
    except Exception as exc:
        return {"loop_available": True, "error": str(exc)}

    by_coro: Counter[str] = Counter()
    by_state: Counter[str] = Counter()
    for task in tasks:
        by_coro[_task_coro_name(task)] += 1
        if task.cancelled():
            by_state["cancelled"] += 1
        elif task.done():
            by_state["done"] += 1
        else:
            by_state["pending"] += 1

    return {
        "loop_available": True,
        "total": len(tasks),
        "by_state": dict(by_state),
        "by_coro": dict(by_coro.most_common(limit)),
    }


def _logging_worker_stats() -> Dict[str, Any]:
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    worker = GLOBAL_LOGGING_WORKER
    queue = getattr(worker, "_queue", None)
    worker_task = getattr(worker, "_worker_task", None)
    running_tasks = getattr(worker, "_running_tasks", None)
    sem = getattr(worker, "_sem", None)

    running_total: Optional[int] = None
    running_done: Optional[int] = None
    if running_tasks is not None:
        try:
            running_total = len(running_tasks)
            running_done = sum(1 for task in running_tasks if task.done())
        except Exception:
            running_total = None
            running_done = None

    try:
        sem_value = getattr(sem, "_value", None) if sem is not None else None
    except Exception:
        sem_value = None

    return {
        "available": True,
        "timeout": getattr(worker, "timeout", None),
        "max_queue_size": getattr(worker, "max_queue_size", None),
        "concurrency": getattr(worker, "concurrency", None),
        "queue_initialized": queue is not None,
        "queue_size": _safe_qsize(queue),
        "queue_maxsize": _safe_maxsize(queue),
        "running_tasks_total": running_total,
        "running_tasks_done": running_done,
        "semaphore_value": sem_value,
        "worker_task_exists": worker_task is not None,
        "worker_task_done": worker_task.done() if worker_task is not None else None,
        "aggressive_clear_in_progress": getattr(
            worker, "_aggressive_clear_in_progress", None
        ),
    }


def _thread_pool_executor_stats() -> Dict[str, Any]:
    try:
        from litellm.litellm_core_utils.thread_pool_executor import executor
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    work_queue = getattr(executor, "_work_queue", None)
    threads = getattr(executor, "_threads", None)
    return {
        "available": True,
        "max_workers": getattr(executor, "_max_workers", None),
        "threads_count": len(threads) if threads is not None else None,
        "work_queue_size": _safe_qsize(work_queue),
        "shutdown": getattr(executor, "_shutdown", None),
    }


def _proxy_db_queue_stats() -> Dict[str, Any]:
    """Collect DB spend/update queue sizes without importing proxy_server again."""

    proxy_server_module = sys.modules.get("litellm.proxy.proxy_server")
    if proxy_server_module is None:
        return {"available": False, "reason": "proxy_server module not loaded"}

    proxy_logging_obj = getattr(proxy_server_module, "proxy_logging_obj", None)
    writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
    if writer is None:
        return {"available": False, "reason": "db_spend_update_writer missing"}

    queue_attrs = [
        "spend_update_queue",
        "daily_spend_update_queue",
        "daily_team_spend_update_queue",
        "daily_end_user_spend_update_queue",
        "daily_agent_spend_update_queue",
        "daily_org_spend_update_queue",
        "daily_tag_spend_update_queue",
        "tool_discovery_queue",
    ]
    queues: Dict[str, Any] = {}
    for attr in queue_attrs:
        queue_obj = getattr(writer, attr, None)
        update_queue = getattr(queue_obj, "update_queue", None)
        queues[attr] = {
            "queue_size": _safe_qsize(update_queue),
            "queue_maxsize": _safe_maxsize(update_queue),
        }

    return {"available": True, "queues": queues}


def _tracemalloc_top(limit: int) -> list[Dict[str, Any]]:
    if limit <= 0 or not tracemalloc.is_tracing():
        return []

    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        )
    )
    top_items = []
    for stat in snapshot.statistics("traceback")[:limit]:
        top_items.append(
            {
                "size_kb": round(stat.size / 1024, 1),
                "count": stat.count,
                "traceback": [
                    f"{frame.filename}:{frame.lineno}" for frame in stat.traceback
                ],
            }
        )
    return top_items


def _safe_context() -> Dict[str, Any]:
    if _context_provider is None:
        return {}
    try:
        raw_context = dict(_context_provider())
    except Exception as exc:
        return {"error": str(exc)}

    safe_context: Dict[str, Any] = {}
    for key, value in raw_context.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_context[key] = value
        elif isinstance(value, Mapping):
            safe_context[key] = {
                str(nested_key): nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool))
                or nested_value is None
            }
        else:
            safe_context[key] = str(type(value))
    return safe_context


def _write_dump(mode: str = "lite") -> str:
    dump_dir = Path(os.getenv("LITELLM_MEM_DUMP_DIR", "/tmp/litellm-mem"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = dump_dir / f"litellm-mem-{mode}-{timestamp}-{os.getpid()}"
    json_path = prefix.with_suffix(".json")
    text_path = prefix.with_suffix(".txt")

    task_limit = _env_int("LITELLM_MEM_TASK_LIMIT", 80)
    trace_limit = _env_int("LITELLM_MEM_TRACEMALLOC_LIMIT", 25)

    proc_status = _read_proc_status()
    smaps_rollup = _read_smaps_rollup()
    runtime = {
        "logging_worker": _logging_worker_stats(),
        "thread_pool_executor": _thread_pool_executor_stats(),
        "proxy_db_queues": _proxy_db_queue_stats(),
        "asyncio_tasks": _asyncio_task_summary(task_limit),
    }

    payload: Dict[str, Any] = {
        "created_at": timestamp,
        "mode": mode,
        "pid": os.getpid(),
        "python": sys.version,
        "dump_elapsed_seconds": None,
        "tracemalloc": {
            "enabled": tracemalloc.is_tracing(),
            "limit": trace_limit,
            "top": _tracemalloc_top(trace_limit),
        },
        "proc_status": proc_status,
        "smaps_rollup": smaps_rollup,
        "fd_count": _fd_count(),
        "thread_count": threading.active_count(),
        "gc_generation_counts": gc.get_count(),
        "litellm_context": _safe_context(),
        "runtime": runtime,
    }
    payload["dump_elapsed_seconds"] = round(time.monotonic() - started, 3)

    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_text = text_path.with_suffix(".txt.tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"created_at: {payload['created_at']}",
        f"mode: {payload['mode']}",
        f"pid: {payload['pid']}",
        f"VmRSS: {proc_status.get('VmRSS')}",
        f"VmSwap: {proc_status.get('VmSwap')}",
        f"Rss: {smaps_rollup.get('Rss')}",
        f"Swap: {smaps_rollup.get('Swap')}",
        f"fd_count: {payload['fd_count']}",
        f"thread_count: {payload['thread_count']}",
        f"dump_elapsed_seconds: {payload['dump_elapsed_seconds']}",
        "",
        "runtime:",
        json.dumps(runtime, indent=2, sort_keys=True),
        "",
        "litellm_context:",
        json.dumps(payload["litellm_context"], indent=2, sort_keys=True),
        "",
        "top_tracemalloc:",
    ]
    for item in payload["tracemalloc"]["top"]:
        lines.append(f"- {item['size_kb']} KiB in {item['count']} blocks")
        for frame in item["traceback"]:
            lines.append(f"  {frame}")

    tmp_text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_json.replace(json_path)
    tmp_text.replace(text_path)
    return str(text_path)


def dump_now(*_args: Any) -> None:
    if not _dump_lock.acquire(blocking=False):
        _stderr_line("LiteLLM memory debug dump already running")
        return
    try:
        path = _write_dump(mode="lite")
        _stderr_line(f"LiteLLM memory debug dump written: {path}")
    except Exception as exc:
        _stderr_line(f"LiteLLM memory debug dump failed: {exc}")
    finally:
        _dump_lock.release()


def install(context_provider: Optional[Callable[[], Mapping[str, Any]]] = None) -> bool:
    global _context_provider, _event_loop, _installed
    if not _is_enabled():
        return False
    if _installed:
        return True

    _context_provider = context_provider
    try:
        _event_loop = asyncio.get_running_loop()
    except Exception:
        _event_loop = None

    if os.getenv("LITELLM_MEM_TRACEMALLOC", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        if not tracemalloc.is_tracing():
            frame_count = _env_int("LITELLM_MEM_TRACEMALLOC_FRAMES", 8)
            tracemalloc.start(frame_count)

    # SIGUSR2 is intentionally mapped to the same lightweight dump so an
    # accidental SIGUSR2 does not terminate the process or run a heavy GC scan.
    signal.signal(signal.SIGUSR1, dump_now)
    signal.signal(signal.SIGUSR2, dump_now)
    _installed = True
    _stderr_line(
        "LiteLLM memory debug enabled; send SIGUSR1 or SIGUSR2 for lightweight dump"
    )
    return True
