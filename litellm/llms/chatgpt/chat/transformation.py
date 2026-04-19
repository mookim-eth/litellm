from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator(api_base=api_base)

    @staticmethod
    def _get_authenticator_for_request(
        api_base: Optional[str],
        litellm_params: Optional[dict],
    ) -> Authenticator:
        auth_file_path: Optional[str] = None
        if litellm_params is not None:
            auth_file_path = litellm_params.get("chatgpt_auth_file_path")
        return Authenticator(auth_file_path=auth_file_path, api_base=api_base)

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = api_base or self.authenticator.get_api_base()
        return dynamic_api_base, None, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        request_api_base = api_base
        if request_api_base is None and litellm_params is not None:
            request_api_base = litellm_params.get("api_base")
        authenticator = self._get_authenticator_for_request(
            request_api_base, litellm_params
        )
        resolved_api_key = api_key
        if not resolved_api_key:
            try:
                resolved_api_key = authenticator.get_access_token()
            except GetAccessTokenError as e:
                raise AuthenticationError(
                    model=model,
                    llm_provider="chatgpt",
                    message=str(e),
                )
        validated_headers = super().validate_environment(
            headers,
            model,
            messages,
            optional_params,
            litellm_params,
            resolved_api_key,
            request_api_base or authenticator.get_api_base(),
        )

        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            resolved_api_key or "", account_id, session_id
        )
        return {**default_headers, **validated_headers}

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    @staticmethod
    def _stringify_instruction_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "\n".join(part for part in text_parts if part)
        return ""

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        transformed_messages = self._transform_messages(messages=messages, model=model)
        instruction_parts: List[str] = []
        non_instruction_messages: List[AllMessageValues] = []

        for message in transformed_messages:
            role = message.get("role")
            if role in ("system", "developer"):
                instruction_text = self._stringify_instruction_content(
                    message.get("content")
                )
                if instruction_text:
                    instruction_parts.append(instruction_text)
            else:
                non_instruction_messages.append(message)

        extra_body = dict(optional_params.get("extra_body") or {})

        if instruction_parts:
            extra_body["instructions"] = "\n\n".join(
                part for part in instruction_parts if part
            )
        else:
            extra_body.setdefault("instructions", "")

        if extra_body:
            optional_params = {**optional_params, "extra_body": extra_body}

        return {"model": model, "messages": non_instruction_messages, **optional_params}

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        optional_params.pop("metadata", None)
        optional_params.setdefault("stream", False)
        return optional_params
