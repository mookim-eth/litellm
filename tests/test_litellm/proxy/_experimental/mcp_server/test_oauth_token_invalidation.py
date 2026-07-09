from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    mcp_oauth_token_identity,
    purge_user_oauth_credentials_for_server,
)


def _server(**overrides):
    values = {
        "url": "https://up.example/mcp",
        "auth_type": "oauth2",
        "oauth2_flow": "authorization_code",
        "authorization_url": "https://idp.example/authorize",
        "token_url": "https://idp.example/token",
        "registration_url": "https://idp.example/register",
        "credentials": {
            "client_id": "client",
            "client_secret": "secret",
            "scopes": ["read"],
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_oauth_identity_changes_when_resource_changes():
    assert mcp_oauth_token_identity(_server()) != mcp_oauth_token_identity(
        _server(url="https://new-target.example/mcp")
    )


@pytest.mark.asyncio
async def test_purge_removes_database_rows_and_cached_tokens(monkeypatch):
    rows = [SimpleNamespace(user_id="u1"), SimpleNamespace(user_id="u2")]
    prisma = MagicMock()
    table = prisma.db.litellm_mcpusercredentials
    table.find_many = AsyncMock(return_value=rows)
    table.delete_many = AsyncMock()

    from litellm.proxy._experimental.mcp_server import oauth2_token_cache

    delete = AsyncMock()
    monkeypatch.setattr(oauth2_token_cache.mcp_per_user_token_cache, "delete", delete)

    assert await purge_user_oauth_credentials_for_server(prisma, "server-1") == 2
    table.delete_many.assert_awaited_once_with(where={"server_id": "server-1"})
    assert delete.await_count == 2
