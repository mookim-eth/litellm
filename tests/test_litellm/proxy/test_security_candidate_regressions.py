from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.prompts.prompt_endpoints import convert_prompt_file_to_json


def _auth(role=LitellmUserRoles.INTERNAL_USER, user_id="user-1", team_id=None):
    return UserAPIKeyAuth(user_role=role, user_id=user_id, team_id=team_id)


@pytest.mark.asyncio
async def test_dotprompt_converter_rejects_path_filenames():
    upload = UploadFile(filename="../evil.prompt", file=BytesIO(b"ignored"))

    with pytest.raises(HTTPException) as exc:
        await convert_prompt_file_to_json(file=upload, user_api_key_dict=_auth())

    assert exc.value.status_code == 400
