from typing import Any, Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class ZAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Native Responses API configuration for Z.AI."""

    def __init__(self) -> None:
        super().__init__()
        self._next_sequence_number = 0

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ZAI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or get_secret_str("ZAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("ZAI_API_BASE")
            or "https://open.bigmodel.cn/api/v1"
        )
        api_base = api_base.rstrip("/")
        if api_base.endswith("/api/coding/paas/v4"):
            api_base = api_base[: -len("/api/coding/paas/v4")] + "/api/v1"
        return f"{api_base}/responses"

    def transform_streaming_response(
        self, model: str, parsed_chunk: dict, logging_obj: Any
    ):
        """Ensure native Z.AI events have the sequence field Codex expects."""
        parsed_chunk = dict(parsed_chunk)
        upstream_sequence = parsed_chunk.get("sequence_number")
        if not isinstance(upstream_sequence, int):
            parsed_chunk["sequence_number"] = self._next_sequence_number
            self._next_sequence_number += 1
        else:
            self._next_sequence_number = max(
                self._next_sequence_number, upstream_sequence + 1
            )
        return super().transform_streaming_response(
            model=model, parsed_chunk=parsed_chunk, logging_obj=logging_obj
        )
