import pytest

from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _enforce_delegated_model_ceiling,
)


@pytest.mark.asyncio
async def test_omitted_models_inherit_restricted_caller_models():
    request = GenerateKeyRequest()
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["allowed-model"],
    )

    await _enforce_delegated_model_ceiling(
        data=request,
        user_api_key_dict=caller,
        inherit_when_omitted=True,
    )

    assert request.models == ["allowed-model"]


@pytest.mark.asyncio
async def test_empty_models_inherit_restricted_caller_models_on_generate():
    request = GenerateKeyRequest(models=[])
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["allowed-model"],
    )

    await _enforce_delegated_model_ceiling(
        data=request,
        user_api_key_dict=caller,
        inherit_when_omitted=True,
    )

    assert request.models == ["allowed-model"]
