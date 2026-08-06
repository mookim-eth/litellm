import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.proxy import proxy_server


@pytest.mark.asyncio
async def test_stale_low_counter_is_floored_to_database_spend():
    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_get_cache = AsyncMock(return_value=2.0)
    cache.redis_cache.async_set_max = AsyncMock(return_value=12.0)

    with (
        patch.object(proxy_server, "spend_counter_cache", cache),
        patch.object(
            proxy_server.SpendCounterReseed,
            "from_db",
            AsyncMock(return_value=12.0),
        ),
    ):
        spend = await proxy_server.get_current_spend(
            counter_key="spend:key:test",
            fallback_spend=5.0,
            max_budget=10.0,
        )

    assert spend == 12.0
    cache.redis_cache.async_set_max.assert_awaited_once_with(
        key="spend:key:test", value=12.0
    )


@pytest.mark.asyncio
async def test_healthy_counter_does_not_read_database():
    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_get_cache = AsyncMock(return_value=5.0)
    from_db = AsyncMock(return_value=99.0)

    with (
        patch.object(proxy_server, "spend_counter_cache", cache),
        patch.object(proxy_server.SpendCounterReseed, "from_db", from_db),
    ):
        spend = await proxy_server.get_current_spend(
            counter_key="spend:key:test",
            fallback_spend=5.0,
            max_budget=10.0,
        )

    assert spend == 5.0
    from_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_unavailable_floor_reads_database_once():
    cache = DualCache()
    query_started = asyncio.Event()
    release_query = asyncio.Event()

    async def unavailable_from_db(*args, **kwargs):
        query_started.set()
        await release_query.wait()
        return None

    from_db = AsyncMock(side_effect=unavailable_from_db)

    with (
        patch.object(proxy_server, "spend_counter_cache", cache),
        patch.object(proxy_server.SpendCounterReseed, "from_db", from_db),
    ):
        spends_task = asyncio.gather(
            *(
                proxy_server._authoritative_floor_spend(
                    counter_key="spend:user:test-unavailable"
                )
                for _ in range(5)
            )
        )
        await query_started.wait()
        await asyncio.sleep(0)
        assert from_db.await_count == 1
        release_query.set()
        spends = await spends_task

    assert spends == [None] * 5
    from_db.assert_awaited_once_with(
        prisma_client=proxy_server.prisma_client,
        counter_key="spend:user:test-unavailable",
    )


@pytest.mark.asyncio
async def test_fail_closed_rejects_unverifiable_budget():
    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_get_cache = AsyncMock(side_effect=RuntimeError("down"))

    with (
        patch.object(proxy_server, "spend_counter_cache", cache),
        patch.object(proxy_server, "general_settings", {"fail_closed_budget_enforcement": True}),
        patch.object(
            proxy_server.SpendCounterReseed,
            "coalesced",
            AsyncMock(return_value=None),
        ),
        patch.object(
            proxy_server.SpendCounterReseed,
            "from_db",
            AsyncMock(return_value=None),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        await proxy_server.get_current_spend(
            counter_key="spend:key:test",
            fallback_spend=1.0,
            max_budget=10.0,
        )

    assert exc.value.status_code == 503
