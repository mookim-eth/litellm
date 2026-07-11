import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy.proxy_server import async_data_generator


def _format_await_chain(awaitable: Any) -> list[str]:
    chain = []
    seen = set()
    while awaitable is not None and id(awaitable) not in seen:
        seen.add(id(awaitable))
        frame = getattr(awaitable, "cr_frame", None) or getattr(
            awaitable, "ag_frame", None
        )
        code = getattr(awaitable, "cr_code", None) or getattr(
            awaitable, "ag_code", None
        )
        if code is not None:
            location = f"{code.co_filename}:{frame.f_lineno}" if frame else code.co_filename
            chain.append(f"AWAIT {code.co_name} {location}")
        awaitable = (
            getattr(awaitable, "cr_await", None)
            or getattr(awaitable, "ag_await", None)
            or getattr(awaitable, "gi_yieldfrom", None)
        )
    return chain


@dataclass
class Timeline:
    started: float = field(default_factory=time.monotonic)
    events: list[tuple[str, float]] = field(default_factory=list)

    def mark(self, name: str) -> None:
        self.events.append((name, time.monotonic() - self.started))

    def at(self, name: str) -> float:
        return next(timestamp for event, timestamp in self.events if event == name)


class ControlledProvider:
    def __init__(self, timeline: Timeline, mode: str, close_gate: asyncio.Event):
        self.timeline = timeline
        self.mode = mode
        self.close_gate = close_gate
        self.first_chunk_sent = asyncio.Event()
        self.close_started = asyncio.Event()
        self.finish_gate = asyncio.Event()
        self.closed = False
        self._chunks = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        if self._chunks == 0:
            self._chunks += 1
            self.timeline.mark("provider.first_chunk")
            self.first_chunk_sent.set()
            return {"choices": [{"delta": {"content": "first"}}]}
        try:
            if self.mode == "continuous":
                await asyncio.sleep(0)
                self._chunks += 1
                return {"choices": [{"delta": {"content": "next"}}]}
            await self.finish_gate.wait()
            raise StopAsyncIteration
        except asyncio.CancelledError:
            self.timeline.mark("provider.cancelled")
            raise

    async def aclose(self):
        if self.closed:
            return
        self.timeline.mark("provider.aclose.start")
        self.close_started.set()
        try:
            await self.close_gate.wait()
        except asyncio.CancelledError:
            self.timeline.mark("provider.aclose.cancelled")
            raise
        self.closed = True
        self.timeline.mark("provider.aclose.end")


class PassthroughProxyLogging:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline

    async def async_post_call_streaming_iterator_hook(self, response, **_kwargs):
        try:
            async for chunk in response:
                yield chunk
        finally:
            self.timeline.mark("hook.finally")

    async def async_post_call_streaming_hook(self, response, **_kwargs):
        return response

    async def post_call_failure_hook(self, **_kwargs):
        self.timeline.mark("failure_hook")


async def _run_scenario(  # noqa: PLR0915
    mode: str, blocked_close: bool = False
):
    timeline = Timeline()
    close_gate = asyncio.Event()
    if not blocked_close:
        close_gate.set()
    provider = ControlledProvider(timeline, mode, close_gate)
    disconnect = asyncio.Event()
    first_body = asyncio.Event()
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        timeline.mark("asgi.http.disconnect")
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]):
        if message["type"] == "http.response.body" and message.get("body"):
            timeline.mark("asgi.first_body")
            first_body.set()

    inner = async_data_generator(provider, object(), {"model": "test"})

    async def traced_outer():
        try:
            async for chunk in inner:
                yield chunk
        except asyncio.CancelledError:
            timeline.mark("outer.cancelled")
            raise
        finally:
            timeline.mark("outer.finally")

    async def background_cleanup():
        timeline.mark("background.cleanup")

    response = StreamingResponse(
        traced_outer(),
        media_type="text/event-stream",
        background=BackgroundTask(background_cleanup),
    )
    app = InFlightRequestsMiddleware(response)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/chat/completions",
        "headers": [],
        "asgi": {"version": "3.0", "spec_version": "2.3"},
    }

    logging = PassthroughProxyLogging(timeline)
    InFlightRequestsMiddleware._in_flight = 0
    with (
        patch("litellm.proxy.proxy_server.proxy_logging_obj", logging),
        patch("litellm.proxy.proxy_server.STREAM_CLOSE_TIMEOUT_SECONDS", 0.05),
    ):
        task = asyncio.create_task(app(scope, receive, send), name=f"asgi-{mode}")
        await asyncio.wait_for(first_body.wait(), 1)
        timeline.mark("test.disconnect.set")
        if mode == "completion_race":
            provider.finish_gate.set()
        disconnect.set()
        if blocked_close:
            await asyncio.wait_for(provider.close_started.wait(), 1)
        else:
            await asyncio.sleep(0.05)
        stack = []
        if not task.done():
            for pending in asyncio.all_tasks():
                if pending.done() or pending is asyncio.current_task():
                    continue
                stack.append(f"TASK {pending.get_name()}")
                stack.extend(_format_await_chain(pending.get_coro()))
                for frame in pending.get_stack():
                    stack.extend(traceback.format_stack(frame))
        await asyncio.wait_for(task, 1)

    timeline.mark("asgi.app.done")
    backlog = InFlightRequestsMiddleware.get_count()
    InFlightRequestsMiddleware._in_flight = 0
    return timeline, stack, backlog


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["waiting", "continuous", "completion_race"])
async def test_disconnect_cancels_stream_and_closes_provider_immediately(mode):
    timeline, stack, backlog = await _run_scenario(mode)

    assert stack == []
    assert backlog == 0
    assert timeline.at("provider.aclose.end") - timeline.at(
        "asgi.http.disconnect"
    ) < 0.2
    assert timeline.at("asgi.app.done") - timeline.at("asgi.http.disconnect") < 0.2
    assert sum(name == "provider.aclose.start" for name, _ in timeline.events) == 1
    assert sum(name == "hook.finally" for name, _ in timeline.events) == 1
    assert timeline.at("background.cleanup") >= timeline.at("outer.finally")


@pytest.mark.asyncio
async def test_blocked_aclose_times_out_once_and_releases_request():
    timeline, stack, backlog = await _run_scenario("waiting", blocked_close=True)

    assert backlog == 0
    assert timeline.at("provider.aclose.cancelled") >= timeline.at(
        "asgi.http.disconnect"
    ) + 0.05
    assert any("stream_response" in line or "__call__" in line for line in stack)
    assert timeline.at("asgi.app.done") >= timeline.at("provider.aclose.cancelled")
    assert timeline.at("asgi.app.done") - timeline.at("asgi.http.disconnect") < 0.2
    assert sum(name == "provider.aclose.start" for name, _ in timeline.events) == 1
    assert timeline.at("background.cleanup") >= timeline.at("outer.finally")
