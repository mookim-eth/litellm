from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import (
    _set_user_keys_blocked,
    delete_user,
    patch_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOp,
    SCIMPatchOperation,
)


@pytest.mark.asyncio
async def test_set_user_keys_blocked_updates_rows_and_invalidates_cache():
    prisma = MagicMock()
    key_row = MagicMock(token="hashed-token", metadata={"existing": True})
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[key_row]
    )
    prisma.db.litellm_verificationtoken.update = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            new_callable=AsyncMock,
        ) as delete_cache,
    ):
        assert await _set_user_keys_blocked("user-1", blocked=True) == 1

    prisma.db.litellm_verificationtoken.find_many.assert_awaited_once_with(
        where={
            "user_id": "user-1",
            "OR": [{"blocked": False}, {"blocked": None}],
        }
    )
    update = prisma.db.litellm_verificationtoken.update.await_args.kwargs
    assert update["where"] == {"token": "hashed-token"}
    assert update["data"]["blocked"] is True
    assert '"scim_blocked": true' in update["data"]["metadata"]
    assert delete_cache.await_args.kwargs["hashed_token"] == "hashed-token"


@pytest.mark.asyncio
async def test_scim_reactivation_does_not_unblock_admin_blocked_key():
    prisma = MagicMock()
    scim_key = MagicMock(
        token="scim-key", metadata={"scim_blocked": True}, blocked=True
    )
    admin_key = MagicMock(token="admin-key", metadata={}, blocked=True)
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[scim_key, admin_key]
    )
    prisma.db.litellm_verificationtoken.update = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            new_callable=AsyncMock,
        ) as delete_cache,
    ):
        assert await _set_user_keys_blocked("user-1", blocked=False) == 1

    update = prisma.db.litellm_verificationtoken.update.await_args.kwargs
    assert update["where"] == {"token": "scim-key"}
    assert update["data"]["blocked"] is False
    assert "scim_blocked" not in update["data"]["metadata"]
    assert delete_cache.await_args.kwargs["hashed_token"] == "scim-key"


@pytest.mark.asyncio
async def test_scim_delete_blocks_keys_before_deleting_owner():
    prisma = MagicMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=LiteLLM_UserTable(user_id="user-1", teams=[])
    )
    prisma.db.litellm_usertable.delete = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._set_user_keys_blocked",
            new_callable=AsyncMock,
        ) as set_blocked,
    ):
        await delete_user(user_id="user-1")

    set_blocked.assert_awaited_once_with(user_id="user-1", blocked=True)
    prisma.db.litellm_usertable.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_scim_patch_inactive_blocks_existing_keys():
    existing = LiteLLM_UserTable(user_id="user-1", teams=[], metadata={})
    updated = LiteLLM_UserTable(
        user_id="user-1", teams=[], metadata={"scim_active": False}
    )
    prisma = MagicMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=existing)
    prisma.db.litellm_usertable.update = AsyncMock(return_value=updated)
    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path="active", value=False)]
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._set_user_keys_blocked",
            new_callable=AsyncMock,
        ) as set_blocked,
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            new_callable=AsyncMock,
            return_value=MagicMock(active=False),
        ),
    ):
        await patch_user(user_id="user-1", patch_ops=patch_ops)

    set_blocked.assert_awaited_once_with(user_id="user-1", blocked=True)
