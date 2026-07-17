from typing import Union

from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.secret_managers.main import str_to_bool


class PrismaDBExceptionHandler:
    """
    Class to handle DB Exceptions or Connection Errors
    """

    @staticmethod
    def _exception_message(e: Exception) -> str:
        try:
            return str(e).lower()
        except Exception:
            return ""

    @staticmethod
    def is_prisma_client_not_connected_error(e: Exception) -> bool:
        """
        Returns True for Prisma query-engine-not-connected failures.

        In production this can surface either as prisma.errors.ClientNotConnectedError
        or as a plain/wrapped Exception string. Treat both forms as DB transport
        errors so auth can reconnect instead of returning a misleading 401.
        """
        import prisma

        if isinstance(e, prisma.errors.ClientNotConnectedError):
            return True

        error_message = PrismaDBExceptionHandler._exception_message(e)
        return (
            "client is not connected to the query engine" in error_message
            or "must call `connect()` before attempting to query data"
            in error_message
            or "must call connect() before attempting to query data" in error_message
            or (
                "query engine" in error_message
                and "not connected" in error_message
                and "connect" in error_message
            )
        )

    @staticmethod
    def is_prisma_http_client_closed_error(e: Exception) -> bool:
        """
        Returns True for Prisma HTTP client closed failures, including wrapped
        string variants.
        """
        import prisma

        if isinstance(e, prisma.errors.HTTPClientClosedError):
            return True

        error_message = PrismaDBExceptionHandler._exception_message(e)
        return (
            "http client is closed" in error_message
            or "httpx client has been closed" in error_message
        )

    @staticmethod
    def should_allow_request_on_db_unavailable() -> bool:
        """
        Returns True if the request should be allowed to proceed despite the DB connection error
        """
        from litellm.proxy.proxy_server import general_settings

        _allow_requests_on_db_unavailable: Union[bool, str] = general_settings.get(
            "allow_requests_on_db_unavailable", False
        )
        if isinstance(_allow_requests_on_db_unavailable, bool):
            return _allow_requests_on_db_unavailable
        if str_to_bool(_allow_requests_on_db_unavailable) is True:
            return True
        return False

    @staticmethod
    def is_database_connection_error(e: Exception) -> bool:
        """Match connectivity failures, never ordinary Prisma data errors."""
        if PrismaDBExceptionHandler.is_database_transport_error(e):
            return True
        if isinstance(e, ProxyException) and e.type == ProxyErrorTypes.no_db_connection:
            return True
        return False

    @staticmethod
    def is_database_transport_error(e: Exception) -> bool:
        """
        Returns True only for transport/connectivity failures where a reconnect
        attempt makes sense (e.g. DB is unreachable, connection dropped).

        Use this for reconnect logic — data-layer errors like UniqueViolationError
        mean the DB IS reachable, so reconnecting would be pointless.
        """
        import prisma

        if isinstance(e, DB_CONNECTION_ERROR_TYPES):
            return True
        if PrismaDBExceptionHandler.is_prisma_client_not_connected_error(e):
            return True
        if PrismaDBExceptionHandler.is_prisma_http_client_closed_error(e):
            return True
        if isinstance(
            e,
            (
                prisma.errors.HTTPClientClosedError,
            ),
        ):
            return True
        if isinstance(e, prisma.errors.PrismaError):
            error_message = PrismaDBExceptionHandler._exception_message(e)
            connection_keywords = (
                "can't reach database server",
                "cannot reach database server",
                "can't connect",
                "cannot connect",
                "connection error",
                "connection closed",
                "timed out",
                "timeout",
                "connection refused",
                "network is unreachable",
                "no route to host",
                "broken pipe",
            )
            if any(keyword in error_message for keyword in connection_keywords):
                return True
        if isinstance(e, ProxyException) and e.type == ProxyErrorTypes.no_db_connection:
            return True
        return False

    @staticmethod
    def handle_db_exception(e: Exception):
        """
        Primary handler for `allow_requests_on_db_unavailable` flag. Decides whether to raise a DB Exception or not based on the flag.

        - If exception is a DB Connection Error, and `allow_requests_on_db_unavailable` is True,
            - Do not raise an exception, return None
        - Else, raise the exception
        """
        if (
            PrismaDBExceptionHandler.is_database_connection_error(e)
            and PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
        ):
            return None
        raise e
