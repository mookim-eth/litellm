from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _enforce_key_and_fallback_model_access


@pytest.mark.asyncio
async def test_teamless_all_team_models_key_is_unrestricted():
    """Legacy teamless keys treat all-team-models as all proxy models."""
    token = UserAPIKeyAuth(models=["all-team-models"], team_id=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
        new_callable=AsyncMock,
    ) as mock_can_key_call_model:
        await _enforce_key_and_fallback_model_access(
            valid_token=token,
            request_data={"model": "gpt-5.4-mini"},
            route="/chat/completions",
            llm_model_list=None,
            llm_router=None,
            request=None,
        )

    mock_can_key_call_model.assert_not_awaited()
