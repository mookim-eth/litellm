from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError

from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.litellm_core_utils.disabled_observability_integrations import (
    DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE,
    ensure_observability_integration_enabled,
)
from litellm.litellm_core_utils.litellm_logging import (
    _init_custom_logger_compatible_class,
    get_custom_logger_compatible_class,
    set_callbacks,
)
from litellm.proxy._types import AddTeamCallback, ProxyException
from litellm.proxy.health_endpoints._health_endpoints import health_services_endpoint
from litellm.proxy.vertex_ai_endpoints.langfuse_endpoints import langfuse_proxy_route


@pytest.mark.parametrize("integration_name", ["langfuse", "langfuse_otel", "langsmith"])
def test_disabled_observability_integrations_are_rejected(integration_name):
    with pytest.raises(ValueError, match="disabled by local security policy"):
        ensure_observability_integration_enabled(integration_name)


@pytest.mark.parametrize("callback_name", ["langfuse", "langfuse_otel", "langsmith"])
def test_disabled_key_and_team_callbacks_are_rejected(callback_name):
    with pytest.raises((PydanticValidationError, ValueError)) as exc_info:
        AddTeamCallback(
            callback_name=callback_name,
            callback_vars={},
        )

    assert DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE in str(exc_info.value)


def test_unrelated_observability_integrations_remain_enabled():
    ensure_observability_integration_enabled("datadog")
    callback = AddTeamCallback(
        callback_name="gcs",
        callback_vars={"gcs_bucket_name": "test-bucket"},
    )

    assert callback.callback_name == "gcs"


def test_dynamic_langfuse_logger_creation_is_rejected():
    with pytest.raises(ValueError, match="disabled by local security policy"):
        LangFuseHandler.get_langfuse_logger_for_request(
            standard_callback_dynamic_params={},
            in_memory_dynamic_logger_cache=MagicMock(),
        )


def test_legacy_langfuse_callback_initialization_is_rejected():
    with pytest.raises(ValueError, match="disabled by local security policy"):
        set_callbacks(["langfuse"])


@pytest.mark.parametrize("integration_name", ["langfuse", "langfuse_otel", "langsmith"])
def test_custom_logger_initialization_does_not_create_disabled_integration(
    integration_name,
):
    assert (
        _init_custom_logger_compatible_class(
            logging_integration=integration_name,
            internal_usage_cache=None,
            llm_router=None,
        )
        is None
    )
    assert get_custom_logger_compatible_class(integration_name) is None


@pytest.mark.asyncio
async def test_langfuse_passthrough_is_rejected_before_processing_request():
    with pytest.raises(HTTPException) as exc_info:
        await langfuse_proxy_route(
            endpoint="api/public/ingestion",
            request=MagicMock(),
            fastapi_response=MagicMock(),
        )

    assert exc_info.value.status_code == 403
    assert DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["langfuse", "langfuse_otel", "langsmith"])
async def test_disabled_observability_health_checks_are_rejected(service):
    with pytest.raises(ProxyException) as exc_info:
        await health_services_endpoint(
            user_api_key_dict=MagicMock(),
            service=service,
        )

    assert exc_info.value.code == "403"
    assert DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE in str(exc_info.value.message)
