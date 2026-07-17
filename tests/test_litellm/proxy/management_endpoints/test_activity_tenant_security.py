from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _team_row(team_id, admins):
    row = MagicMock()
    row.team_id = team_id
    row.team_alias = team_id
    row.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": team_id,
        "admins": admins,
        "members_with_roles": [
            {"user_id": user_id, "role": "admin"} for user_id in admins
        ],
        "members": [],
    }
    return row


@pytest.mark.asyncio
async def test_team_activity_requires_admin_on_every_requested_team():
    from litellm.proxy.management_endpoints import team_endpoints

    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_many = AsyncMock(
        return_value=[_team_row("team-a", ["alice"]), _team_row("team-b", ["bob"])]
    )
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[MagicMock(token="alice-key")]
    )
    user_info = MagicMock(teams=["team-a", "team-b"])
    captured = {}

    async def capture_activity(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new=AsyncMock(return_value=user_info),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity",
            new=AsyncMock(side_effect=capture_activity),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
    ):
        await team_endpoints.get_team_daily_activity(
            team_ids="team-a,team-b",
            user_api_key_dict=caller,
        )

    assert captured["api_key"] == ["alice-key"]


@pytest.mark.asyncio
async def test_agent_activity_intersects_requested_ids_with_permissions():
    from litellm.proxy.agent_endpoints import endpoints

    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    prisma = MagicMock()
    prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
    captured = {}

    async def capture_activity(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.agent_endpoints.endpoints.check_feature_access_for_user",
            new=AsyncMock(),
        ),
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler."
            "AgentRequestHandler.get_allowed_agents",
            new=AsyncMock(return_value=["allowed-agent"]),
        ),
        patch(
            "litellm.proxy.agent_endpoints.endpoints.get_daily_activity",
            new=AsyncMock(side_effect=capture_activity),
        ),
    ):
        await endpoints.get_agent_daily_activity(
            agent_ids="allowed-agent,foreign-agent",
            user_api_key_dict=caller,
        )

    assert captured["entity_id"] == ["allowed-agent"]
