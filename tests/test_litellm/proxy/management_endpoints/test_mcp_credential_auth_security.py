from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_MCPServerTable, UserAPIKeyAuth
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    _authorize_user_mcp_server_state,
)


@pytest.mark.asyncio
async def test_non_admin_cannot_manage_credentials_for_forbidden_server():
    server = LiteLLM_MCPServerTable(
        server_id="forbidden-server",
        alias="forbidden",
        transport="sse",
    )
    caller = UserAPIKeyAuth(user_id="internal-user")

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints."
            "get_mcp_server",
            new=AsyncMock(return_value=server),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints."
            "build_effective_auth_contexts",
            new=AsyncMock(return_value=[caller]),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints."
            "global_mcp_server_manager.get_allowed_mcp_servers",
            new=AsyncMock(return_value=[]),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _authorize_user_mcp_server_state(
                MagicMock(), caller, "forbidden-server"
            )

    assert exc_info.value.status_code == 403
