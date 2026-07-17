from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _process_single_key_update,
)
from litellm.types.proxy.management_endpoints.key_management_endpoints import (
    BulkUpdateKeyRequestItem,
)


@pytest.mark.asyncio
async def test_internal_user_cannot_set_tags_through_bulk_key_update():
    item = BulkUpdateKeyRequestItem(key="target-key", tags=["privileged-route"])
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal-user",
    )

    with pytest.raises(HTTPException) as exc:
        await _process_single_key_update(
            key_update_item=item,
            user_api_key_dict=caller,
            litellm_changed_by=None,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
            llm_router=None,
        )

    assert exc.value.status_code == 403
    assert "tags" in str(exc.value.detail)
