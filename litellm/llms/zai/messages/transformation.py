"""
ZAI Anthropic Messages transformation config.

ZAI/BigModel exposes an Anthropic-compatible Messages API. Use it directly for
LiteLLM's /v1/messages path instead of bridging Anthropic Messages to Chat
Completions.
"""

from typing import List, Optional, Tuple

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
    DEFAULT_ANTHROPIC_API_VERSION,
)
from litellm.secret_managers.main import get_secret_str

ZAI_ANTHROPIC_API_BASE = "https://open.bigmodel.cn/api/anthropic/v1/messages"


class ZAIMessagesConfig(AnthropicMessagesConfig):
    """ZAI/BigModel Anthropic-compatible Messages configuration."""

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or get_secret_str("ZAI_API_KEY")
            or get_secret_str("BIGMODEL_API_KEY")
            or get_secret_str("ZHIPUAI_API_KEY")
            or litellm.api_key
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        env_api_base = get_secret_str("ZAI_ANTHROPIC_API_BASE")
        if env_api_base:
            return env_api_base

        if api_base:
            # If a user already passed an Anthropic-compatible base/path, honor it.
            if "/anthropic" in api_base:
                return api_base

            # Many ZAI configs use the OpenAI-compatible chat API base. For
            # Anthropic Messages, switch to the matching Anthropic base instead
            # of appending /v1/messages to a chat-completions URL.
            if "open.bigmodel.cn" in api_base or "bigmodel.cn" in api_base:
                return ZAI_ANTHROPIC_API_BASE
            if "api.z.ai" in api_base:
                return "https://api.z.ai/api/anthropic/v1/messages"

        return ZAI_ANTHROPIC_API_BASE

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base)
        if base_url.endswith("/v1/messages"):
            return base_url
        if base_url.endswith("/"):
            return f"{base_url}v1/messages"
        return f"{base_url}/v1/messages"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        if "authorization" not in {k.lower() for k in headers} and "x-api-key" not in {
            k.lower() for k in headers
        }:
            dynamic_api_key = self.get_api_key(api_key=api_key)
            if dynamic_api_key is not None:
                headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "anthropic-version" not in {k.lower() for k in headers}:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in {k.lower() for k in headers}:
            headers["content-type"] = "application/json"

        return headers, api_base
