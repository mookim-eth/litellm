from unittest.mock import AsyncMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage


def _chat_completion_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        model="custom-model",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="pong", role="assistant"),
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_sync_responses_bridge_drops_codex_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch(
        "litellm.completion", return_value=_chat_completion_response()
    ) as mock_completion:
        handler.response_api_handler(
            model="custom-model",
            input="ping",
            responses_api_request={},
            custom_llm_provider="custom_openai",
            client_metadata={"session_id": "test-session"},
            allowed_openai_params=["client_metadata", "verbosity"],
        )

    assert "client_metadata" not in mock_completion.call_args.kwargs
    assert mock_completion.call_args.kwargs["allowed_openai_params"] == ["verbosity"]


@pytest.mark.asyncio
async def test_async_responses_bridge_drops_codex_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_chat_completion_response(),
    ) as mock_acompletion:
        await handler.response_api_handler(
            model="custom-model",
            input="ping",
            responses_api_request={},
            custom_llm_provider="custom_openai",
            _is_async=True,
            client_metadata={"session_id": "test-session"},
            allowed_openai_params=["client_metadata", "verbosity"],
        )

    assert "client_metadata" not in mock_acompletion.call_args.kwargs
    assert mock_acompletion.call_args.kwargs["allowed_openai_params"] == ["verbosity"]
