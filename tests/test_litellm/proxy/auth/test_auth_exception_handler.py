import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
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
from litellm.proxy.auth.auth_exception_handler import (
    MalformedAPIKeyError,
    MissingAPIKeyError,
    UserAPIKeyAuthExceptionHandler,
)


def test_auth_failure_uses_server_ingress_time_over_injected_time():
    from starlette.requests import Request

    request = Request({
        "type": "http", "method": "POST", "path": "/v1/responses",
        "headers": [], "query_string": b"", "scheme": "http",
        "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
        "state": {"_litellm_request_start_time": 1700000000.0},
    })
    data = {"proxy_server_request": {"request_start_time": 1, "arrival_time": 1}}
    UserAPIKeyAuthExceptionHandler._add_request_context_to_failure_logging_data(
        request, data, {}
    )
    assert data["proxy_server_request"]["request_start_time"] == 1700000000.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error", [HTTPClientClosedError(), ClientNotConnectedError()]
)
async def test_db_transport_error_uses_restricted_fallback(prisma_error):
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
            prisma_error,
            MagicMock(),
            {},
            "/test",
            None,
            "test-key",
        )

    assert result.key_name == "failed-to-connect-to-db"
    assert result.user_id == "__db_unavailable_fallback__"
    assert result.user_role == "internal_user"


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
    ],
)
async def test_db_data_errors_do_not_use_fallback(prisma_error):
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
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                prisma_error,
                mock_request,
                mock_request_data,
                mock_route,
                mock_span,
                mock_api_key,
            )


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
    with patch.object(verbose_proxy_logger, "exception") as mock_exception:
        from litellm.exceptions import BudgetExceededError

        budget_error = BudgetExceededError(
            message="Budget exceeded", current_cost=100, max_budget=100
        )
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                budget_error,
                mock_request,
                mock_request_data,
                mock_route,
                mock_span,
                mock_api_key,
            )

    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
    assert exc_info.value.code == "400"
    mock_exception.assert_not_called()


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
    ) as mock_post_call_failure_hook, patch.object(
        verbose_proxy_logger, "exception"
    ) as mock_exception:
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
    mock_exception.assert_called_once()


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
    assert metadata["tags"] == [
        "team:dev",
        "env:test",
        "User-Agent: codex-tui",
        "User-Agent: codex-tui/0.142.4",
    ]
    assert proxy_server_request["headers"]["user-agent"] == "codex-tui/0.142.4"
    assert "authorization" not in {
        header.lower() for header in proxy_server_request["headers"].keys()
    }


@pytest.mark.asyncio
async def test_malformed_key_http_400_does_not_log_exception_traceback():
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                HTTPException(
                    status_code=400,
                    detail=(
                        "LiteLLM Virtual Key expected. Received=tid=****fa4c, "
                        "expected to start with 'sk-'."
                    ),
                ),
                mock_request,
                {},
                "/v1/responses",
                None,
                "tid=masked",
            )

    assert exc_info.value.code == "400"
    assert "tid=****fa4c" in exc_info.value.message
    mock_exception.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth_error", "expected_message"),
    [
        (
            HTTPException(status_code=401, detail="Invalid API key"),
            "Invalid API key",
        ),
        (
            ProxyException(
                message="Authentication Error, Invalid proxy server token passed.",
                type=ProxyErrorTypes.token_not_found_in_db,
                param="key",
                code=status.HTTP_401_UNAUTHORIZED,
            ),
            "Authentication Error, Invalid proxy server token passed.",
        ),
    ],
    ids=["http_exception", "proxy_exception"],
)
async def test_unauthorized_client_error_does_not_log_exception_traceback(
    auth_error, expected_message
):
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                auth_error,
                mock_request,
                {},
                "/v1/responses",
                None,
                "test-key",
            )

    assert exc_info.value.code == "401"
    assert exc_info.value.message == expected_message
    mock_exception.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (ProxyErrorTypes.expired_key, "Authentication Error - Expired Key."),
        (ProxyErrorTypes.bad_request_error, "Invalid request metadata."),
    ],
)
async def test_expected_proxy_http_400_does_not_log_exception_traceback(
    error_type, message
):
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()
    auth_error = ProxyException(
        message=message,
        type=error_type,
        param=None,
        code=status.HTTP_400_BAD_REQUEST,
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                auth_error,
                mock_request,
                {},
                "/v1/responses",
                None,
                "test-key",
            )

    assert exc_info.value.code == "400"
    assert exc_info.value.message == message
    mock_exception.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "error_code"),
    [
        (ProxyErrorTypes.no_db_connection, status.HTTP_400_BAD_REQUEST),
        (ProxyErrorTypes.bad_request_error, status.HTTP_500_INTERNAL_SERVER_ERROR),
    ],
)
async def test_unexpected_proxy_error_logs_exception_traceback(
    error_type, error_code
):
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()
    auth_error = ProxyException(
        message="Authentication infrastructure failure.",
        type=error_type,
        param=None,
        code=error_code,
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                auth_error,
                mock_request,
                {},
                "/v1/responses",
                None,
                "test-key",
            )

    mock_exception.assert_called_once()


@pytest.mark.asyncio
async def test_forbidden_auth_error_does_not_log_exception_traceback():
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "GET"
    mock_request.url = "http://testserver/config/list"
    mock_request.state = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                HTTPException(status_code=403, detail="forbidden route"),
                mock_request,
                {},
                "/config/list",
                None,
                "test-key",
            )

    assert exc_info.value.code == "403"
    assert exc_info.value.message == "forbidden route"
    mock_exception.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth_error", "expected_message"),
    [
        (
            MissingAPIKeyError("No api key passed in."),
            "Authentication Error, No api key passed in.",
        ),
        (
            MalformedAPIKeyError(
                "Malformed API Key passed in. Ensure Key has `Bearer ` prefix."
            ),
            "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
        ),
    ],
    ids=["missing", "malformed"],
)
async def test_expected_api_key_error_does_not_log_exception_traceback(
    auth_error, expected_message
):
    handler = UserAPIKeyAuthExceptionHandler()
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "203.0.113.5"
    mock_request.method = "POST"
    mock_request.url = "http://testserver/v1/responses"
    mock_request.state = MagicMock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
        return_value=None,
    ), patch.object(verbose_proxy_logger, "exception") as mock_exception:
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                auth_error,
                mock_request,
                {},
                "/v1/responses",
                None,
                "",
            )

    assert exc_info.value.code == "401"
    assert exc_info.value.message == expected_message
    mock_exception.assert_not_called()


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
            with pytest.raises(ProxyException):
                await handler._handle_authentication_error(
                    PrismaError(),
                    mock_request,
                    mock_request_data,
                    test_route,
                    mock_span,
                    mock_api_key,
                )
            # Verify post_call_failure_hook was called with the correct route
            mock_post_call_failure_hook.assert_called_once()
            call_args = mock_post_call_failure_hook.call_args[1]
            assert call_args["user_api_key_dict"].request_route == test_route
