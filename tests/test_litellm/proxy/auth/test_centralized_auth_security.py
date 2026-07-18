from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.auth.user_api_key_auth import _run_centralized_common_checks


@pytest.mark.asyncio
async def test_centralized_checks_skip_public_readiness_route():
    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/health/readiness",
            "headers": [],
        }
    )

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks",
        new=AsyncMock(),
    ) as common:
        await _run_centralized_common_checks(
            user_api_key_auth_obj=UserAPIKeyAuth(),
            request=request,
            request_data={},
            route="/health/readiness",
        )

    common.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_builder_return_runs_centralized_common_checks():
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [],
        }
    )
    token = UserAPIKeyAuth(token="hashed-token", user_id="internal-user")
    centralized = AsyncMock()

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth._read_request_body",
            new=AsyncMock(return_value={"model": "restricted-model"}),
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth._user_api_key_auth_builder",
            new=AsyncMock(return_value=token),
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth."
            "_run_centralized_common_checks",
            new=centralized,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.RouteChecks.should_call_route"
        ),
    ):
        result = await user_api_key_auth(
            request=request,
            api_key="Bearer key",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )

    assert result is token
    centralized.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_virtual_team_preserves_dashboard_session_user():
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/key/generate",
            "headers": [],
        }
    )
    token = UserAPIKeyAuth(
        token="hashed-token",
        team_id="litellm-dashboard",
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    user = LiteLLM_UserTable(
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_team_object",
            new=AsyncMock(side_effect=HTTPException(status_code=404)),
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_user_object",
            new=AsyncMock(return_value=user),
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_global_proxy_spend",
            new=AsyncMock(return_value=0.0),
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new=AsyncMock(),
        ) as common,
        patch("litellm.proxy.proxy_server.master_key", "test-master-key"),
    ):
        await _run_centralized_common_checks(
            user_api_key_auth_obj=token,
            request=request,
            request_data={},
            route="/key/generate",
        )

    assert common.await_args.kwargs["user_object"] is user
    assert common.await_args.kwargs["team_object"].team_id == "litellm-dashboard"
