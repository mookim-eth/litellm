import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, UserAPIKeyAuth
from litellm.proxy.management_endpoints.common_utils import (
    _check_passthrough_routes_caller_permission,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"allowed_passthrough_routes": []},
        {"metadata": {"allowed_passthrough_routes": []}},
    ],
)
def test_non_admin_cannot_set_team_passthrough_routes(payload):
    data = NewTeamRequest(team_alias="security-test", **payload)
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    with pytest.raises(HTTPException) as exc_info:
        _check_passthrough_routes_caller_permission(
            data=data,
            user_api_key_dict=caller,
            entity="team",
        )

    assert exc_info.value.status_code == 403
