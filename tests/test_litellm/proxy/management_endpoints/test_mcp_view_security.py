from litellm.proxy._types import LiteLLM_MCPServerTable
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    _sanitize_mcp_server_for_non_admin,
)


def test_non_admin_mcp_view_redacts_connection_and_command_fields():
    server = LiteLLM_MCPServerTable(
        server_id="server-id",
        alias="safe-alias",
        transport="sse",
        url="https://example.test/token-in-path",
        spec_path="https://example.test/spec?token=secret",
        static_headers={"Authorization": "Bearer secret"},
        extra_headers=["Authorization"],
        env={"API_KEY": "secret"},
        command="python",
        args=["server.py", "--token", "secret"],
        authorization_url="https://idp.test/authorize?secret=1",
        token_url="https://idp.test/token",
        registration_url="https://idp.test/register",
    )

    sanitized = _sanitize_mcp_server_for_non_admin(server)

    assert sanitized.alias == "safe-alias"
    assert sanitized.url is None
    assert sanitized.spec_path is None
    assert sanitized.static_headers is None
    assert sanitized.extra_headers == []
    assert sanitized.env == {}
    assert sanitized.command is None
    assert sanitized.args == []
    assert sanitized.authorization_url is None
    assert sanitized.token_url is None
    assert sanitized.registration_url is None
