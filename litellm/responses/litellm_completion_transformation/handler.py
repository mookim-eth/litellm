"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse


class LiteLLMCompletionTransformationHandler:
    @staticmethod
    def _requires_reasoning_content(kwargs: Dict[str, Any]) -> bool:
        litellm_metadata = kwargs.get("litellm_metadata") or {}
        model_info = litellm_metadata.get("model_info") or kwargs.get("model_info") or {}
        return bool(
            isinstance(model_info, dict)
            and model_info.get("requires_reasoning_content")
        )

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        # Codex sends client_metadata for its own session telemetry. It is not a
        # Chat Completions parameter, and forwarding it through the Responses
        # fallback bridge makes the OpenAI SDK reject the request before it
        # reaches an OpenAI-compatible provider.
        kwargs.pop("client_metadata", None)
        allowed_openai_params = kwargs.get("allowed_openai_params")
        if isinstance(allowed_openai_params, list):
            kwargs["allowed_openai_params"] = [
                param for param in allowed_openai_params if param != "client_metadata"
            ]

        # Responses clients need structured reasoning items for multi-turn
        # replay. Converting reasoning_content into <think> text drops the
        # provider field that thinking-mode chat APIs require on the next turn.
        kwargs["merge_reasoning_content_in_choices"] = False

        requires_reasoning_content = self._requires_reasoning_content(kwargs)

        litellm_completion_request: dict = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=model,
            input=input,
            responses_api_request=responses_api_request,
            custom_llm_provider=custom_llm_provider,
            stream=stream,
            extra_headers=extra_headers,
            **kwargs,
        )
        if requires_reasoning_content:
            LiteLLMCompletionResponsesConfig.ensure_reasoning_content_on_assistant_tool_calls(
                litellm_completion_request.get("messages") or []
            )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **litellm_completion_request,
            **kwargs,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                chat_completion_response=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )
        if self._requires_reasoning_content(kwargs):
            LiteLLMCompletionResponsesConfig.ensure_reasoning_content_on_assistant_tool_calls(
                litellm_completion_request.get("messages") or []
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(
            **acompletion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                chat_completion_response=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get(
                    "custom_llm_provider"
                ),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )
