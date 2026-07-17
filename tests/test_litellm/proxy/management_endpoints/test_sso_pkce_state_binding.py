import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse

from litellm.proxy._types import ProxyException
from litellm.proxy.management_endpoints.ui_sso import (
    SSOAuthenticationHandler,
    _validate_pkce_state_cookie,
)


def _mock_sso_redirect(state: str):
    response = RedirectResponse(f"https://idp.example/authorize?state={state}")
    sso = MagicMock()
    sso.__enter__ = MagicMock(return_value=sso)
    sso.__exit__ = MagicMock(return_value=None)
    sso.get_login_redirect = AsyncMock(return_value=response)
    return sso


@pytest.mark.asyncio
async def test_should_set_secure_browser_state_cookie_for_pkce_redirect():
    cache = MagicMock()
    cache.async_set_cache = AsyncMock()
    with (
        patch.dict(
            os.environ,
            {"GENERIC_CLIENT_STATE": "browser-state", "GENERIC_CLIENT_USE_PKCE": "true"},
        ),
        patch("litellm.proxy.proxy_server.redis_usage_cache", None),
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
    ):
        response = await SSOAuthenticationHandler.get_generic_sso_redirect_response(
            generic_sso=_mock_sso_redirect("browser-state"),
            generic_authorization_endpoint="https://idp.example/authorize",
        )

    assert response is not None
    cookie = "\n".join(response.headers.getlist("set-cookie"))
    assert "litellm_oauth_state=browser-state" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie


@pytest.mark.parametrize(
    "query_state,cookie_state",
    [("state-a", None), ("state-a", "state-b"), (None, "state-a")],
)
def test_should_reject_pkce_callback_without_matching_browser_cookie(
    query_state, cookie_state
):
    request = MagicMock(spec=Request)
    request.query_params = {"state": query_state} if query_state else {}
    request.cookies = (
        {"litellm_oauth_state": cookie_state} if cookie_state is not None else {}
    )

    with pytest.raises(ProxyException) as exc_info:
        _validate_pkce_state_cookie(request)

    assert exc_info.value.code == "400"


def test_should_accept_pkce_callback_with_matching_browser_cookie():
    request = MagicMock(spec=Request)
    request.query_params = {"state": "matching-state"}
    request.cookies = {"litellm_oauth_state": "matching-state"}

    _validate_pkce_state_cookie(request)
