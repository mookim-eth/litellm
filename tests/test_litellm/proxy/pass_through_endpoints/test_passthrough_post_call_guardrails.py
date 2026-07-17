import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)

MODULE = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"
COLLECT = "litellm.proxy.pass_through_endpoints.passthrough_guardrails.PassthroughGuardrailHandler.collect_guardrails"


def _response(body):
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
        request=httpx.Request("POST", "https://example.com/generate"),
    )


def _request():
    request = MagicMock()
    request.method = "POST"
    request.query_params = {}
    request.headers = MagicMock()
    request.headers.copy.return_value = {}
    return request


def _auth():
    auth = MagicMock()
    auth.api_key = "hashed-key"
    auth.user_id = "user-1"
    auth.team_id = "team-1"
    auth.org_id = None
    auth.request_route = "/vertex_ai/generate"
    return auth


def _patches(proxy_logging, response):
    client_wrapper = MagicMock()
    client_wrapper.client = AsyncMock()
    endpoint_logging = MagicMock()
    endpoint_logging.pass_through_async_success_handler = AsyncMock()
    stack = ExitStack()
    for patcher in (
        patch(
            f"{MODULE}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            return_value=response,
        ),
        patch(f"{MODULE}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch(f"{MODULE}.pass_through_endpoint_logging", endpoint_logging),
        patch(f"{MODULE}.get_async_httpx_client", return_value=client_wrapper),
        patch(f"{MODULE}._read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{MODULE}._safe_get_request_headers", return_value={}),
    ):
        stack.enter_context(patcher)
    return stack


@pytest.mark.asyncio
@patch(COLLECT, return_value=["required-post-call-guardrail"])
async def test_passthrough_response_runs_configured_post_call_guardrail(_collect):
    body = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(return_value={})
    proxy_logging.post_call_success_hook = AsyncMock(return_value=body)
    with _patches(proxy_logging, _response(body)):
        result = await pass_through_request(
            request=_request(),
            target="https://example.com/generate",
            custom_headers={"Content-Type": "application/json"},
            user_api_key_dict=_auth(),
            stream=False,
        )
    proxy_logging.post_call_success_hook.assert_awaited_once()
    assert result.status_code == 200


@pytest.mark.asyncio
@patch(COLLECT, return_value=["required-post-call-guardrail"])
async def test_passthrough_guardrail_block_returns_content_filter(_collect):
    body = {"dangerous": True}
    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(return_value={})
    proxy_logging.post_call_success_hook = AsyncMock(
        side_effect=ModifyResponseException(
            message="blocked by policy",
            model="provider-model",
            request_data={},
            guardrail_name="required-post-call-guardrail",
        )
    )
    proxy_logging.post_call_failure_hook = AsyncMock()
    with _patches(proxy_logging, _response(body)):
        result = await pass_through_request(
            request=_request(),
            target="https://example.com/generate",
            custom_headers={"Content-Type": "application/json"},
            user_api_key_dict=_auth(),
            stream=False,
        )
    assert result.status_code == 200
    assert json.loads(result.body)["error"]["type"] == "content_filter"
    proxy_logging.post_call_failure_hook.assert_awaited_once()
