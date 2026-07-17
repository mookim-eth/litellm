from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


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
