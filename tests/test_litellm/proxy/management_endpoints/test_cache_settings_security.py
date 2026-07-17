import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.cache_settings_endpoints import (
    get_cache_settings,
)


@pytest.mark.asyncio
async def test_cache_settings_read_masks_discrete_and_url_credentials():
    cache_row = MagicMock(
        cache_settings=json.dumps(
            {
                "type": "redis",
                "password": "plain-password",
                "url": "redis://user:url-password@host:6379/1",
                "namespace": "safe-namespace",
            }
        )
    )
    prisma = MagicMock()
    prisma.db.litellm_cacheconfig.find_unique = AsyncMock(return_value=cache_row)
    proxy_config = MagicMock()
    proxy_config._decrypt_db_variables.side_effect = lambda variables_dict: dict(
        variables_dict
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
    ):
        result = await get_cache_settings(user_api_key_dict=UserAPIKeyAuth())

    assert result.current_values["password"] == "***REDACTED***"
    assert result.current_values["url"] == "***REDACTED***"
    assert result.current_values["namespace"] == "safe-namespace"
