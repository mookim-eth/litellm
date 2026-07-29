from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils import debug_utils
from litellm.proxy.management_endpoints import internal_user_endpoints, team_endpoints
from litellm.proxy.prompts.prompt_endpoints import convert_prompt_file_to_json
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def _auth(role=LitellmUserRoles.INTERNAL_USER, user_id="user-1", team_id=None):
    return UserAPIKeyAuth(user_role=role, user_id=user_id, team_id=team_id)


@pytest.mark.asyncio
async def test_global_spend_reset_requires_proxy_admin():
    with pytest.raises(HTTPException) as exc:
        await spend_management_endpoints.global_spend_reset(user_api_key_dict=_auth())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_provider_budgets_requires_admin_view():
    with pytest.raises(HTTPException) as exc:
        await spend_management_endpoints.provider_budgets(user_api_key_dict=_auth())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_debug_asyncio_tasks_requires_admin_view():
    with pytest.raises(HTTPException) as exc:
        await debug_utils.get_active_tasks_stats(user_api_key_dict=_auth())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_team_filter_ui_requires_admin_view(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    with pytest.raises(HTTPException) as exc:
        await team_endpoints.ui_view_teams(user_api_key_dict=_auth())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_user_filter_ui_defaults_to_self_for_non_admin(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    fake_prisma = MagicMock()
    fake_prisma.db.litellm_usertable.find_many = AsyncMock()
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
    monkeypatch.setattr(
        internal_user_endpoints,
        "_resolve_org_filter_for_user_search",
        AsyncMock(return_value=None),
    )

    result = await internal_user_endpoints.ui_view_users(
        user_id="other-user", page=1, page_size=50, user_api_key_dict=_auth(user_id="caller-user")
    )

    assert result == []
    fake_prisma.db.litellm_usertable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_dotprompt_converter_rejects_path_filenames():
    upload = UploadFile(filename="../evil.prompt", file=BytesIO(b"ignored"))

    with pytest.raises(HTTPException) as exc:
        await convert_prompt_file_to_json(file=upload, user_api_key_dict=_auth())

    assert exc.value.status_code == 400


def test_vector_store_access_check_includes_rag_retrieval_config_id():
    registry = VectorStoreRegistry()

    assert registry.get_vector_store_ids_to_run(
        {"retrieval_config": {"vector_store_id": "vs-private"}}
    ) == ["vs-private"]
