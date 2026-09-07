import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from starlette.websockets import WebSocketDisconnect

from litellm.proxy.response_api_endpoints.endpoints import (
    responses_websocket_endpoint,
)


@pytest.fixture
def websocket_context():
    websocket = SimpleNamespace(
        accept=AsyncMock(),
        close=AsyncMock(),
        send_text=AsyncMock(),
        headers={},
        query_params={},
        scope={"headers": []},
        url="ws://localhost/v1/responses",
        receive_text=AsyncMock(),
    )
    with (
        patch(
            "litellm.proxy.response_api_endpoints.endpoints.user_api_key_auth_websocket",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "litellm.proxy.response_api_endpoints.endpoints.verbose_proxy_logger.exception"
        ) as log_exception,
        patch(
            "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as processor,
        patch(
            "litellm.proxy.route_llm_request.route_request", new_callable=AsyncMock
        ) as route_request,
    ):
        yield websocket, log_exception, processor, route_request


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1000, 1001, 1006])
async def test_initial_disconnect_is_not_logged_as_error(websocket_context, code):
    websocket, log_exception, processor, route_request = websocket_context
    websocket.receive_text.side_effect = WebSocketDisconnect(code=code)

    await responses_websocket_endpoint(websocket)

    websocket.accept.assert_awaited_once_with()
    websocket.receive_text.assert_awaited_once_with()
    websocket.close.assert_not_awaited()
    websocket.send_text.assert_not_awaited()
    log_exception.assert_not_called()
    processor.assert_not_called()
    route_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_read_error_is_still_logged(websocket_context):
    websocket, log_exception, processor, route_request = websocket_context
    websocket.receive_text.side_effect = RuntimeError("read failed")

    await responses_websocket_endpoint(websocket)

    log_exception.assert_called_once_with(
        "Responses WebSocket failed to read initial client message"
    )
    websocket.close.assert_awaited_once_with(
        code=1011, reason="Failed to read initial message"
    )
    processor.assert_not_called()
    route_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, code, reason",
    [("{", 1003, "Invalid JSON"), ('{"type":"response.create"}', 1008, "Model required")],
)
async def test_invalid_first_message_retains_error_contract(
    websocket_context, message, code, reason
):
    websocket, log_exception, processor, route_request = websocket_context
    websocket.receive_text.return_value = message

    await responses_websocket_endpoint(websocket)

    websocket.close.assert_awaited_once_with(code=code, reason=reason)
    websocket.send_text.assert_awaited_once()
    error = json.loads(websocket.send_text.call_args.args[0])
    assert error["error"]["type"] == "invalid_request_error"
    log_exception.assert_not_called()
    processor.assert_not_called()
    route_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_source", ["query", "top_level", "response"])
async def test_valid_model_is_routed(websocket_context, model_source):
    websocket, log_exception, processor, route_request = websocket_context
    model = "test-model"
    payload = {"type": "response.create", "model": model}
    if model_source == "query":
        websocket.query_params = {"model": model}
    elif model_source == "response":
        payload = {"type": "response.create", "response": {"model": model}}
    message = json.dumps(payload)
    websocket.receive_text.return_value = message
    processor.return_value.common_processing_pre_call_logic = AsyncMock(
        return_value=({"model": model, "websocket": websocket}, MagicMock())
    )
    upstream = AsyncMock()

    async def run_upstream(**kwargs):
        return upstream()

    route_request.side_effect = run_upstream

    await responses_websocket_endpoint(websocket)

    processor.return_value.common_processing_pre_call_logic.assert_awaited_once()
    route_request.assert_awaited_once()
    data = route_request.call_args.kwargs["data"]
    assert data["model"] == model
    if model_source == "query":
        websocket.receive_text.assert_not_awaited()
        assert "initial_client_message" not in data
    else:
        assert data["initial_client_message"] == message
    upstream.assert_awaited_once()
    websocket.close.assert_not_awaited()
    log_exception.assert_not_called()
