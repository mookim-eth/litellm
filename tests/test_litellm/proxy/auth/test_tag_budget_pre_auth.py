from starlette.requests import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup


def _request(tags: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"x-litellm-tags", tags.encode())],
            "query_string": b"",
        }
    )


def test_client_tags_are_stripped_without_admin_opt_in():
    data = {"metadata": {"tags": ["attacker"]}, "tags": ["attacker-root"]}

    LiteLLMProxyRequestSetup.apply_client_tag_policy_pre_auth(
        request=_request("header-tag"),
        request_data=data,
        user_api_key_dict=UserAPIKeyAuth(metadata={}),
    )

    assert "tags" not in data
    assert "tags" not in data["metadata"]


def test_opted_in_header_tags_are_visible_to_budget_check():
    data = {"metadata": '{"tags":["body-tag"]}'}

    LiteLLMProxyRequestSetup.apply_client_tag_policy_pre_auth(
        request=_request("header-tag"),
        request_data=data,
        user_api_key_dict=UserAPIKeyAuth(
            metadata={"allow_client_tags": True},
        ),
    )

    assert data["metadata"]["tags"] == ["body-tag", "header-tag"]
