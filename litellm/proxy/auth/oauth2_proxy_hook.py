from typing import Any, Dict, FrozenSet

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request


ALLOWED_OAUTH2_PROXY_FIELDS: FrozenSet[str] = frozenset(
    {
        "user_id",
        "user_email",
        "team_id",
        "team_alias",
        "org_id",
        "models",
    }
)


async def handle_oauth2_proxy_request(request: Request) -> UserAPIKeyAuth:
    """
    Handle request from oauth2 proxy.
    """
    from litellm.proxy.proxy_server import general_settings

    verbose_proxy_logger.debug("Handling oauth2 proxy request")
    require_trusted_proxy_request(
        request=request,
        general_settings=general_settings,
        feature_name="OAuth2 proxy auth",
    )
    # Define the OAuth2 config mappings
    oauth2_config_mappings: Dict[str, str] = (
        general_settings.get("oauth2_config_mappings") or {}
    )
    verbose_proxy_logger.debug(f"Oauth2 config mappings: {oauth2_config_mappings}")

    if not oauth2_config_mappings:
        raise ValueError("Oauth2 config mappings not found in general_settings")

    disallowed = sorted(
        set(oauth2_config_mappings.keys()) - ALLOWED_OAUTH2_PROXY_FIELDS
    )
    if disallowed:
        raise ValueError(
            "Oauth2 proxy auth refuses to map non-identity UserAPIKeyAuth "
            f"fields from request headers: {disallowed}. Only identity "
            f"fields are accepted ({sorted(ALLOWED_OAUTH2_PROXY_FIELDS)}); "
            "anything else would let a caller forge enforcement parameters "
            "by spoofing the matching header. If you need a trusted upstream "
            "to assert anything beyond identity, use JWT auth."
        )

    # Initialize a dictionary to store the mapped values
    auth_data: Dict[str, Any] = {}

    # Extract values from headers based on the mappings
    for key, header in oauth2_config_mappings.items():
        value = request.headers.get(header)
        if value:
            # Convert models to list if present
            if key == "models":
                auth_data[key] = [model.strip() for model in value.split(",")]
            else:
                auth_data[key] = value
    verbose_proxy_logger.debug(
        "Auth data before creating UserAPIKeyAuth object: keys=%s",
        list(auth_data.keys()),
    )
    user_api_key_auth = UserAPIKeyAuth(**auth_data)
    verbose_proxy_logger.debug(
        "UserAPIKeyAuth object created with keys: %s",
        list(user_api_key_auth.__fields_set__),
    )
    # Create and return UserAPIKeyAuth object
    return user_api_key_auth
