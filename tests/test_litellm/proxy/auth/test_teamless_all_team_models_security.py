from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _enforce_key_and_fallback_model_access


@pytest.mark.asyncio
async def test_teamless_all_team_models_key_does_not_skip_model_check():
    token = UserAPIKeyAuth(models=["all-team-models"], team_id=None)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
        new=AsyncMock(side_effect=RuntimeError("model check reached")),
    ):
        with pytest.raises(RuntimeError, match="model check reached"):
            await _enforce_key_and_fallback_model_access(
                valid_token=token,
                request_data={"model": "restricted-model"},
                route="/chat/completions",
                llm_model_list=None,
                llm_router=None,
            )
