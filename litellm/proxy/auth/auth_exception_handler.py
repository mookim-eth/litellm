"""
Handles Authentication Errors
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_utils import _get_request_ip_address
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.types.services import ServiceTypes

DB_UNAVAILABLE_FALLBACK_USER_ID = "__db_unavailable_fallback__"

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class MissingAPIKeyError(Exception):
    """Expected client authentication failure when no API key is supplied."""


class MalformedAPIKeyError(Exception):
    """Expected client authentication failure for a malformed API key."""


class UserAPIKeyAuthExceptionHandler:
    @staticmethod
    def _get_header_value(headers: Dict[str, Any], header_name: str) -> Optional[Any]:
        header_name_lower = header_name.lower()
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == header_name_lower:
                return value
        return None

    @staticmethod
    def _merge_tags(existing_tags: Any, tags_to_add: Optional[List[str]]) -> List[str]:
        final_tags: List[str] = []
        if isinstance(existing_tags, list):
            final_tags.extend(str(tag) for tag in existing_tags)
        if tags_to_add:
            for tag in tags_to_add:
                if tag not in final_tags:
                    final_tags.append(tag)
        return final_tags

    @staticmethod
    def _get_request_tags_from_headers_and_body(
        headers: Dict[str, Any], request_data: dict
    ) -> Optional[List[str]]:
        tags: Optional[List[str]] = None

        header_tags = UserAPIKeyAuthExceptionHandler._get_header_value(
            headers=headers, header_name="x-litellm-tags"
        )
        if isinstance(header_tags, str):
            tags = [tag.strip() for tag in header_tags.split(",") if tag.strip()]
        elif isinstance(header_tags, list):
            tags = [str(tag).strip() for tag in header_tags if str(tag).strip()]

        body_tags = request_data.get("tags")
        if isinstance(body_tags, list):
            tags = [str(tag) for tag in body_tags]

        return tags

    @staticmethod
    def _get_user_agent_tags(user_agent: Optional[Any]) -> Optional[List[str]]:
        if user_agent is None or litellm.disable_add_user_agent_to_request_tags is True:
            return None

        user_agent_str = str(user_agent).strip()
        if not user_agent_str:
            return None

        user_agent_tags: List[str] = []
        if "/" in user_agent_str:
            user_agent_tags.append("User-Agent: " + user_agent_str.split("/")[0])
        user_agent_tags.append("User-Agent: " + user_agent_str)
        return user_agent_tags

    @staticmethod
    def _add_request_context_to_failure_logging_data(
        request: Request,
        request_data: dict,
        general_settings: dict,
    ) -> str:
        """
        Add the same request context used by normal LLM calls to auth/proxy-only
        failure logs, without storing raw request bodies or auth headers.
        """
        requester_ip = (
            _get_request_ip_address(
                request=request,
                use_x_forwarded_for=general_settings.get(
                    "use_x_forwarded_for", False
                ),
                use_cloudflare_header=general_settings.get(
                    "use_cloudflare_header", False
                ),
            )
            or ""
        )
        requester_ip = str(requester_ip)

        raw_headers = _safe_get_request_headers(request)
        user_agent = UserAPIKeyAuthExceptionHandler._get_header_value(
            headers=raw_headers, header_name="user-agent"
        )

        metadata = request_data.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            request_data["metadata"] = metadata

        metadata["requester_ip_address"] = requester_ip
        if user_agent is not None:
            metadata["user_agent"] = str(user_agent)

        request_tags = (
            UserAPIKeyAuthExceptionHandler._get_request_tags_from_headers_and_body(
                headers=raw_headers, request_data=request_data
            )
            or []
        )
        user_agent_tags = (
            UserAPIKeyAuthExceptionHandler._get_user_agent_tags(user_agent) or []
        )
        tags = [*request_tags, *user_agent_tags]
        if tags:
            metadata["tags"] = UserAPIKeyAuthExceptionHandler._merge_tags(
                existing_tags=metadata.get("tags"), tags_to_add=tags
            )

        if "proxy_server_request" not in request_data:
            from litellm.proxy.litellm_pre_call_utils import clean_headers

            lower_header_names = {
                key.lower() for key in raw_headers.keys() if isinstance(key, str)
            }
            if "x-litellm-api-key" in lower_header_names:
                authenticated_with_header = "x-litellm-api-key"
            elif "authorization" in lower_header_names:
                authenticated_with_header = "authorization"
            else:
                authenticated_with_header = "x-api-key"

            request_data["proxy_server_request"] = {
                "url": str(getattr(request, "url", "")),
                "method": str(getattr(request, "method", "")),
                "headers": clean_headers(
                    raw_headers,
                    litellm_key_header_name=general_settings.get(
                        "litellm_key_header_name"
                    ),
                    # Auth failures should never log provider/auth credentials,
                    # even if normal successful calls are configured to forward
                    # provider auth headers.
                    forward_llm_provider_auth_headers=False,
                    authenticated_with_header=authenticated_with_header,
                ),
                # Keep auth-failure logging lightweight and avoid persisting
                # unauthenticated request bodies.
                "body": {},
                "arrival_time": time.time(),
            }

        return requester_ip

    @staticmethod
    async def _handle_authentication_error(
        e: Exception,
        request: Request,
        request_data: dict,
        route: str,
        parent_otel_span: Optional[Span],
        api_key: str,
    ) -> UserAPIKeyAuth:
        """
        Handles Connection Errors when reading a Virtual Key from LiteLLM DB
        Use this if you don't want failed DB queries to block LLM API reqiests

        Reliability scenarios this covers:
        - DB is down and having an outage
        - Unable to read / recover a key from the DB

        Returns:
            - UserAPIKeyAuth: If general_settings.allow_requests_on_db_unavailable is True

        Raises:
            - Original Exception in all other cases
        """
        from litellm.proxy.proxy_server import (
            general_settings,
            proxy_logging_obj,
        )

        if (
            PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
            and PrismaDBExceptionHandler.is_database_connection_error(e)
        ):
            # log this as a DB failure on prometheus
            proxy_logging_obj.service_logging_obj.service_failure_hook(
                service=ServiceTypes.DB,
                call_type="get_key_object",
                error=e,
                duration=0.0,
            )

            return UserAPIKeyAuth(
                key_name="failed-to-connect-to-db",
                token="failed-to-connect-to-db",
                user_id=DB_UNAVAILABLE_FALLBACK_USER_ID,
                user_role=LitellmUserRoles.INTERNAL_USER,
                request_route=route,
            )
        else:
            # raise the exception to the caller
            requester_ip = UserAPIKeyAuthExceptionHandler._add_request_context_to_failure_logging_data(
                request=request,
                request_data=request_data,
                general_settings=general_settings,
            )
            # Expected client-input failures are still reported through the
            # normal failure hooks and HTTP response, but are not application
            # exceptions and should not emit ERROR tracebacks.
            is_expected_client_error = (
                isinstance(
                    e,
                    (
                        litellm.BudgetExceededError,
                        MissingAPIKeyError,
                        MalformedAPIKeyError,
                    ),
                )
                or (
                    isinstance(e, HTTPException)
                    and e.status_code
                    in (
                        status.HTTP_400_BAD_REQUEST,
                        status.HTTP_401_UNAUTHORIZED,
                        status.HTTP_403_FORBIDDEN,
                    )
                )
                or (
                    isinstance(e, ProxyException)
                    and (
                        e.code == str(status.HTTP_401_UNAUTHORIZED)
                        or (
                            e.code == str(status.HTTP_400_BAD_REQUEST)
                            and e.type
                            in (
                                ProxyErrorTypes.expired_key,
                                ProxyErrorTypes.bad_request_error,
                            )
                        )
                    )
                )
            )
            if not is_expected_client_error:
                verbose_proxy_logger.exception(
                    "litellm.proxy.proxy_server.user_api_key_auth(): Exception occured - {}\nRequester IP Address:{}".format(
                        str(e),
                        requester_ip,
                    ),
                    extra={"requester_ip": requester_ip},
                )

            # Log this exception to OTEL, Datadog etc
            user_api_key_dict = UserAPIKeyAuth(
                parent_otel_span=parent_otel_span,
                api_key=api_key,
                request_route=route,
            )
            # Allow callbacks to transform the error response
            transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=e,
                user_api_key_dict=user_api_key_dict,
                error_type=ProxyErrorTypes.auth_error,
                route=route,
            )
            # Use transformed exception if callback returned one, otherwise use original
            if transformed_exception is not None:
                e = transformed_exception

            if isinstance(e, litellm.BudgetExceededError):
                raise ProxyException(
                    message=e.message,
                    type=ProxyErrorTypes.budget_exceeded,
                    param=None,
                    code=400,
                )
            if isinstance(e, HTTPException):
                raise ProxyException(
                    message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                    type=ProxyErrorTypes.auth_error,
                    param=getattr(e, "param", "None"),
                    code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
                )
            elif isinstance(e, ProxyException):
                raise e
            if PrismaDBExceptionHandler.is_database_transport_error(e):
                raise ProxyException(
                    message=(
                        "Database connection error during authentication, "
                        + str(e)
                    ),
                    type=ProxyErrorTypes.no_db_connection,
                    param=getattr(e, "param", "None"),
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            raise ProxyException(
                message="Authentication Error, " + str(e),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=status.HTTP_401_UNAUTHORIZED,
            )
