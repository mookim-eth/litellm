import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from prisma import errors as prisma_errors
from prisma.errors import (
    ClientNotConnectedError,
    DataError,
    ForeignKeyViolationError,
    HTTPClientClosedError,
    MissingRequiredValueError,
    PrismaError,
    RawQueryError,
    RecordNotFoundError,
    TableNotFoundError,
    UniqueViolationError,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        DataError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        UniqueViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        ForeignKeyViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        MissingRequiredValueError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RawQueryError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        TableNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RecordNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        HTTPClientClosedError(),
        ClientNotConnectedError(),
    ],
)
async def test_handle_authentication_error_db_unavailable(prisma_error):
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with DB connection error when requests are allowed
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await handler._handle_authentication_error(
            prisma_error,
            mock_request,
            mock_request_data,
            mock_route,
            mock_span,
            mock_api_key,
        )
        assert result.key_name == "failed-to-connect-to-db"
        assert result.token == "failed-to-connect-to-db"


@pytest.mark.asyncio
async def test_handle_authentication_error_budget_exceeded():
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with budget exceeded error
    with pytest.raises(ProxyException) as exc_info:
        from litellm.exceptions import BudgetExceededError

        budget_error = BudgetExceededError(
            message="Budget exceeded", current_cost=100, max_budget=100
        )
        await handler._handle_authentication_error(
            budget_error,
            mock_request,
            mock_request_data,
            mock_route,
            mock_span,
            mock_api_key,
        )

    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded


@pytest.mark.asyncio
async def test_handle_authentication_error_db_transport_error_returns_503_not_401():
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/v1/responses"
    mock_span = None
    mock_api_key = "test-key"
    db_error = Exception(
        "Client is not connected to the query engine, you must call `connect()` "
        "before attempting to query data."
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
    ) as mock_post_call_failure_hook:
        mock_post_call_failure_hook.return_value = None

        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                db_error,
                mock_request,
                mock_request_data,
                mock_route,
                mock_span,
                mock_api_key,
            )

    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert exc_info.value.code == str(status.HTTP_503_SERVICE_UNAVAILABLE)
    assert "Authentication Error" not in exc_info.value.message
    assert "Database connection error during authentication" in exc_info.value.message


@pytest.mark.asyncio
async def test_auth_failure_logging_includes_request_ip_and_user_agent_context():
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    mock_request.headers = {
        "authorization": "Bearer secret-token",
        "user-agent": "codex-tui/0.142.4",
        "x-forwarded-for": "198.51.100.10",
        "x-litellm-tags": "team:dev, env:test",
    }
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()
    request_data = {"model": "gpt-test"}

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "allow_requests_on_db_unavailable": False,
            "use_x_forwarded_for": True,
        },
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
    ) as mock_post_call_failure_hook:
        mock_post_call_failure_hook.return_value = None

        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                HTTPException(status_code=401, detail="bad auth"),
                mock_request,
                request_data,
                "/v1/responses",
                None,
                "test-key",
            )

    mock_post_call_failure_hook.assert_called_once()
    logged_request_data = mock_post_call_failure_hook.call_args.kwargs["request_data"]
    metadata = logged_request_data["metadata"]
    proxy_server_request = logged_request_data["proxy_server_request"]

    assert metadata["requester_ip_address"] == "198.51.100.10"
    assert metadata["user_agent"] == "codex-tui/0.142.4"
    assert metadata["tags"] == ["team:dev", "env:test"]
    assert proxy_server_request["headers"]["user-agent"] == "codex-tui/0.142.4"
    assert "authorization" not in {
        header.lower() for header in proxy_server_request["headers"].keys()
    }


@pytest.mark.asyncio
async def test_route_passed_to_post_call_failure_hook():
    """
    This route is used by proxy track_cost_callback's async_post_call_failure_hook to check if the route is an LLM route
    """
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    test_route = "/custom/route"
    mock_span = None
    mock_api_key = "test-key"

    # Mock proxy_logging_obj.post_call_failure_hook
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
    ) as mock_post_call_failure_hook:
        # Test with DB connection error
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ):
            try:
                await handler._handle_authentication_error(
                    PrismaError(),
                    mock_request,
                    mock_request_data,
                    test_route,
                    mock_span,
                    mock_api_key,
                )
            except Exception as e:
                pass
            asyncio.sleep(1)
            # Verify post_call_failure_hook was called with the correct route
            mock_post_call_failure_hook.assert_called_once()
            call_args = mock_post_call_failure_hook.call_args[1]
            assert call_args["user_api_key_dict"].request_route == test_route
