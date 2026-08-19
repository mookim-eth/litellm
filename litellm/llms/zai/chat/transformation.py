from typing import Any, Dict, List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        # ZAI exposes native Responses at /api/v1/responses, while Chat
        # Completions for these deployments must continue using the Coding API.
        if api_base is not None and api_base.rstrip("/").endswith("/api/v1"):
            api_base = (
                api_base.rstrip("/")[: -len("/api/v1")]
                + "/api/coding/paas/v4"
            )
        dynamic_api_key = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        import litellm

        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def _convert_non_function_tools_to_functions(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        ZAI's OpenAI-compatible chat API only accepts tools with type=function.

        Codex sends Responses API built-in tools (for example local_shell/apply_patch)
        through LiteLLM's Responses->Chat bridge for providers without a native
        Responses API. Convert these built-ins to function tools so ZAI accepts the
        request instead of rejecting it with `tools[n].type:type is illegal`.
        """
        if tools is None:
            return None

        converted_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                converted_tools.append(tool)
                continue

            if tool.get("type") == "function":
                converted_tools.append(tool)
                continue

            tool_name = str(tool.get("name") or tool.get("type") or "tool")
            description = str(
                tool.get("description")
                or f"Run the {tool_name} tool. Arguments depend on the tool."
            )
            parameters = tool.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {
                    "type": "object",
                    "additionalProperties": True,
                }
            elif "type" not in parameters:
                parameters = {**parameters, "type": "object"}

            converted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": parameters,
                        "strict": bool(tool.get("strict", False)),
                    },
                }
            )

        return converted_tools

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        if "tools" in optional_params:
            optional_params["tools"] = self._convert_non_function_tools_to_functions(
                optional_params.get("tools")
            )

        return optional_params
