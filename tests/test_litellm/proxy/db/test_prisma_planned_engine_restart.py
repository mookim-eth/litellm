"""Regression coverage for coordinated Prisma query-engine replacement."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


def _wrapper(pid: int = 111) -> PrismaWrapper:
    prisma = MagicMock()
    prisma.is_connected.return_value = True
    prisma._engine.process.pid = pid
    prisma.disconnect = AsyncMock()
    return PrismaWrapper(prisma, iam_token_db_auth=False)


@pytest.mark.asyncio
async def test_planned_recreate_records_pid_and_advances_generation(mock_prisma_binary):
    wrapper = _wrapper()
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        assert await wrapper.recreate_prisma_client("postgresql://new") is True

    assert wrapper._expected_engine_deaths == {111}
    assert wrapper._engine_generation == 1


@pytest.mark.asyncio
async def test_planned_recreate_gracefully_disconnects_without_killing_engine(
    mock_prisma_binary,
):
    wrapper = _wrapper()
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())
    old_prisma = wrapper._original_prisma

    with patch.object(wrapper, "_kill_engine_process", new_callable=AsyncMock) as kill:
        assert await wrapper.recreate_prisma_client("postgresql://new") is True

    old_prisma.disconnect.assert_awaited_once()
    kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_planned_recreate_kills_engine_after_disconnect_timeout(
    mock_prisma_binary,
):
    wrapper = _wrapper()
    wrapper.PLANNED_DISCONNECT_TIMEOUT_SECONDS = 0.01

    async def stalled_disconnect():
        await asyncio.Event().wait()

    wrapper._original_prisma.disconnect = AsyncMock(side_effect=stalled_disconnect)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with patch.object(wrapper, "_kill_engine_process", new_callable=AsyncMock) as kill:
        assert await wrapper.recreate_prisma_client("postgresql://new") is True

    kill.assert_awaited_once_with(111)


@pytest.mark.asyncio
async def test_concurrent_guarded_recreates_collapse_to_one(mock_prisma_binary):
    wrapper = _wrapper()
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        results = await asyncio.gather(
            wrapper.recreate_prisma_client("postgresql://new", expected_generation=0),
            wrapper.recreate_prisma_client("postgresql://new", expected_generation=0),
        )

    assert sorted(results) == [False, True]
    assert mock_prisma_binary.Prisma.call_count == 1
    assert wrapper._engine_generation == 1


@pytest.mark.asyncio
async def test_recreate_recovers_from_disconnected_client(mock_prisma_binary):
    wrapper = _wrapper()
    wrapper._original_prisma.is_connected.return_value = False
    type(wrapper._original_prisma)._engine = property(
        lambda _self: (_ for _ in ()).throw(RuntimeError("not connected"))
    )
    replacement = MagicMock(connect=AsyncMock())
    mock_prisma_binary.Prisma.return_value = replacement

    assert await wrapper.recreate_prisma_client("postgresql://new") is True
    assert wrapper._original_prisma is replacement


@pytest.mark.asyncio
async def test_failed_heavy_reconnect_keeps_dead_state():
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    mock_proxy_logging = AsyncMock(spec=ProxyLogging)
    client = PrismaClient("mock://test", proxy_logging_obj=mock_proxy_logging)
    client._engine_confirmed_dead = True
    client.db.recreate_prisma_client = AsyncMock(side_effect=RuntimeError("failed"))

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        with pytest.raises(RuntimeError, match="failed"):
            await client._run_reconnect_cycle(timeout_seconds=1)

    assert client._engine_confirmed_dead is True
