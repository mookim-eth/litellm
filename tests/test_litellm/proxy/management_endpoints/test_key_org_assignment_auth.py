from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _validate_caller_can_assign_key_org,
)


def _prisma_with_memberships(*organization_ids: str) -> MagicMock:
    prisma = MagicMock()
    user_row = MagicMock()
    user_row.organization_memberships = [
        MagicMock(organization_id=organization_id)
        for organization_id in organization_ids
    ]
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)
    return prisma


@pytest.mark.asyncio
async def test_should_allow_key_org_assignment_for_member():
    prisma = _prisma_with_memberships("org-a", "org-b")
    caller = UserAPIKeyAuth(
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    await _validate_caller_can_assign_key_org(caller, "org-b", prisma)

    prisma.db.litellm_usertable.find_unique.assert_awaited_once_with(
        where={"user_id": "internal-user"},
        include={"organization_memberships": True},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("memberships", [(), ("org-a",)])
async def test_should_reject_key_org_assignment_for_non_member(memberships):
    prisma = _prisma_with_memberships(*memberships)
    caller = UserAPIKeyAuth(
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(caller, "other-org", prisma)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_should_reject_key_org_assignment_without_caller_identity():
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER.value)

    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(caller, "org-a", MagicMock())

    assert exc_info.value.status_code == 403
