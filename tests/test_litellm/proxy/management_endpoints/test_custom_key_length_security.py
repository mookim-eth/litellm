import pytest
from fastapi import HTTPException

from litellm.proxy._types import RegenerateKeyRequest
from litellm.proxy.auth.auth_utils import abbreviate_api_key
from litellm.proxy.management_endpoints.key_management_endpoints import get_new_token


@pytest.mark.asyncio
async def test_regeneration_rejects_short_custom_key():
    with pytest.raises(HTTPException) as exc_info:
        await get_new_token(RegenerateKeyRequest(new_key="sk-short"))

    assert exc_info.value.status_code == 400


def test_short_key_abbreviation_does_not_reveal_suffix():
    assert abbreviate_api_key("sk-short") == "sk-..."
