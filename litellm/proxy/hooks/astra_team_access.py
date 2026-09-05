from typing import Any, Dict, Optional

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.types.utils import CallTypes


# Pin the existing team's ID: renaming it must not change the security boundary.
ASTRA_TEAM_ID = "3880a488-52e7-44cc-9d91-a679452027f8"
ASTRA_MODELS = frozenset(("gpt-6-astra", "gpt-6-astra-1", "gpt-6-astra-2"))


class AstraTeamAccess(CustomLogger):
    """Require astra_team membership in addition to normal model permissions."""

    @staticmethod
    async def _check_access(model: Any, auth: Optional[UserAPIKeyAuth]) -> None:
        if not isinstance(model, str) or model.rsplit("/", 1)[-1] not in ASTRA_MODELS:
            return

        denied = HTTPException(
            status_code=403,
            detail="Astra requires the authenticated user or API key to belong to astra_team.",
        )
        if not isinstance(auth, UserAPIKeyAuth):
            raise denied

        from litellm.proxy.proxy_server import prisma_client

        try:
            # Read current membership: the general Team cache is not reliably
            # invalidated by every member-removal path. Only Astra queries DB.
            if prisma_client is None:
                raise denied
            row = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": ASTRA_TEAM_ID}
            )
            if row is None:
                raise denied
            team = LiteLLM_TeamTable(**row.model_dump())
        except Exception:
            # Missing/deleted team or unavailable authorization data fails closed.
            raise denied from None
        if team.blocked:
            raise denied
        if auth.team_id == ASTRA_TEAM_ID or (
            auth.user_id is not None
            and any(
                member.user_id == auth.user_id for member in team.members_with_roles
            )
        ):
            return
        raise denied

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> None:
        # ProxyBaseLLMRequestProcessing creates this object, replacing any
        # client-supplied logging object. Keep identity outside request metadata
        # so fallback kwargs cannot replace it with a forged user/team mapping.
        logging_obj = data.get("litellm_logging_obj")
        if logging_obj is not None:
            logging_obj._astra_request_auth = user_api_key_dict
        await self._check_access(data.get("model"), user_api_key_dict)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> None:
        # Runs before provider calls/cache reads, for every retry/fallback and
        # after aliases or direct deployment IDs have resolved to a real model.
        auth = getattr(kwargs.get("litellm_logging_obj"), "_astra_request_auth", None)
        await self._check_access(kwargs.get("model"), auth)
