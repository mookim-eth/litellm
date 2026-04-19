import json
from typing import Any, Dict, List, Optional

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @staticmethod
    def _get_authenticator_for_request(
        api_base: Optional[str],
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> Authenticator:
        auth_file_path: Optional[str] = None
        if litellm_params is not None:
            auth_file_path = litellm_params.get("chatgpt_auth_file_path")
        return Authenticator(auth_file_path=auth_file_path, api_base=api_base)

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    @staticmethod
    def _stringify_instruction_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text_value = block.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "\n".join(part for part in text_parts if part)
        return ""

    def _extract_instructions_from_input(
        self, input_items: Any
    ) -> tuple[Any, Optional[str]]:
        if not isinstance(input_items, list):
            return input_items, None

        instruction_parts: List[str] = []
        filtered_items: List[Any] = []

        for item in input_items:
            if not isinstance(item, dict):
                filtered_items.append(item)
                continue

            role = item.get("role")
            if role in ("system", "developer"):
                instruction_text = self._stringify_instruction_content(
                    item.get("content")
                )
                if instruction_text:
                    instruction_parts.append(instruction_text)
                continue

            filtered_items.append(item)

        instructions = "\n\n".join(part for part in instruction_parts if part) or None
        return filtered_items, instructions

    @staticmethod
    def _get_sse_output_index(
        parsed_chunk: Dict[str, Any], item_id_to_output_index: Dict[str, int]
    ) -> Optional[int]:
        output_index = parsed_chunk.get("output_index")
        if isinstance(output_index, int):
            return output_index

        item_id = parsed_chunk.get("item_id")
        if isinstance(item_id, str):
            return item_id_to_output_index.get(item_id)

        item = parsed_chunk.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str):
                return item_id_to_output_index.get(item_id)

        return None

    @staticmethod
    def _ensure_message_content_part(
        output_item: Dict[str, Any], content_index: int
    ) -> Dict[str, Any]:
        content_list = output_item.setdefault("content", [])
        if not isinstance(content_list, list):
            content_list = []
            output_item["content"] = content_list

        while len(content_list) <= content_index:
            content_list.append({"type": "output_text", "text": ""})

        content_part = content_list[content_index]
        if not isinstance(content_part, dict):
            content_part = {"type": "output_text", "text": str(content_part or "")}
            content_list[content_index] = content_part

        content_part.setdefault("type", "output_text")
        content_part.setdefault("text", "")
        return content_part

    @classmethod
    def _handle_sse_output_item_added(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
    ) -> None:
        item = parsed_chunk.get("item")
        output_index = parsed_chunk.get("output_index")
        if not isinstance(item, dict) or not isinstance(output_index, int):
            return

        reconstructed_item = dict(item)
        if reconstructed_item.get("type") == "message":
            reconstructed_item["content"] = list(reconstructed_item.get("content") or [])

        output_items_by_index[output_index] = reconstructed_item
        item_id = reconstructed_item.get("id")
        if isinstance(item_id, str):
            item_id_to_output_index[item_id] = output_index

    @classmethod
    def _handle_sse_content_part_added(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
    ) -> None:
        output_index = cls._get_sse_output_index(parsed_chunk, item_id_to_output_index)
        if output_index is None:
            return

        output_item = output_items_by_index.get(output_index)
        if not isinstance(output_item, dict) or output_item.get("type") != "message":
            return

        part = parsed_chunk.get("part")
        if not isinstance(part, dict):
            return

        content_index = parsed_chunk.get("content_index", 0)
        if not isinstance(content_index, int):
            content_index = 0

        content_part = cls._ensure_message_content_part(output_item, content_index)
        content_part.update(dict(part))

    @classmethod
    def _handle_sse_output_text_delta(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
        accumulated_text: Dict[tuple[str, int], str],
    ) -> None:
        output_index = cls._get_sse_output_index(parsed_chunk, item_id_to_output_index)
        item_id = parsed_chunk.get("item_id")
        if output_index is None:
            if not isinstance(item_id, str):
                return
            output_index = parsed_chunk.get("output_index")
            if not isinstance(output_index, int):
                return
            item_id_to_output_index[item_id] = output_index

        output_item = output_items_by_index.get(output_index)
        if output_item is None:
            output_item = {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "content": [],
            }
            output_items_by_index[output_index] = output_item

        if output_item.get("type") != "message":
            return

        content_index = parsed_chunk.get("content_index", 0)
        if not isinstance(content_index, int):
            content_index = 0

        delta = parsed_chunk.get("delta", "")
        if not isinstance(delta, str):
            return

        content_part = cls._ensure_message_content_part(output_item, content_index)
        text_key = (
            item_id if isinstance(item_id, str) else f"output-{output_index}",
            content_index,
        )
        accumulated_text[text_key] = accumulated_text.get(text_key, "") + delta
        content_part["text"] = accumulated_text[text_key]

    @classmethod
    def _handle_sse_output_text_annotation_added(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
    ) -> None:
        output_index = cls._get_sse_output_index(parsed_chunk, item_id_to_output_index)
        if output_index is None:
            return

        output_item = output_items_by_index.get(output_index)
        if not isinstance(output_item, dict) or output_item.get("type") != "message":
            return

        content_index = parsed_chunk.get("content_index", 0)
        if not isinstance(content_index, int):
            content_index = 0

        annotation = parsed_chunk.get("annotation")
        if not isinstance(annotation, dict):
            return

        content_part = cls._ensure_message_content_part(output_item, content_index)
        annotations = content_part.setdefault("annotations", [])
        if isinstance(annotations, list):
            annotations.append(dict(annotation))

    @classmethod
    def _handle_sse_function_call_arguments_delta(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
    ) -> None:
        output_index = cls._get_sse_output_index(parsed_chunk, item_id_to_output_index)
        item_id = parsed_chunk.get("item_id")
        if output_index is None:
            if not isinstance(item_id, str):
                return
            output_index = parsed_chunk.get("output_index")
            if not isinstance(output_index, int):
                return
            item_id_to_output_index[item_id] = output_index

        output_item = output_items_by_index.get(output_index)
        if output_item is None:
            output_item = {
                "type": "function_call",
                "id": item_id,
                "call_id": item_id,
                "arguments": "",
            }
            output_items_by_index[output_index] = output_item

        if output_item.get("type") != "function_call":
            return

        delta = parsed_chunk.get("delta", "")
        if isinstance(delta, str):
            output_item["arguments"] = f"{output_item.get('arguments', '')}{delta}"

    @staticmethod
    def _merge_done_item_with_existing_output(
        reconstructed_item: Dict[str, Any], current_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        if current_item.get("content") and not reconstructed_item.get("content"):
            reconstructed_item["content"] = current_item["content"]
        elif current_item.get("arguments") and not reconstructed_item.get("arguments"):
            reconstructed_item["arguments"] = current_item["arguments"]
        return reconstructed_item

    @classmethod
    def _handle_sse_output_item_done(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
    ) -> None:
        item = parsed_chunk.get("item")
        output_index = cls._get_sse_output_index(parsed_chunk, item_id_to_output_index)
        if not isinstance(item, dict) or output_index is None:
            return

        reconstructed_item = dict(item)
        current_item = output_items_by_index.get(output_index, {})
        if isinstance(current_item, dict):
            reconstructed_item = cls._merge_done_item_with_existing_output(
                reconstructed_item, current_item
            )

        output_items_by_index[output_index] = reconstructed_item
        item_id = reconstructed_item.get("id")
        if isinstance(item_id, str):
            item_id_to_output_index[item_id] = output_index

    @classmethod
    def _apply_sse_event_to_output_items(
        cls,
        parsed_chunk: Dict[str, Any],
        output_items_by_index: Dict[int, Dict[str, Any]],
        item_id_to_output_index: Dict[str, int],
        accumulated_text: Dict[tuple[str, int], str],
    ) -> None:
        event_type = parsed_chunk.get("type")
        if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED:
            cls._handle_sse_output_item_added(
                parsed_chunk, output_items_by_index, item_id_to_output_index
            )
        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED:
            cls._handle_sse_content_part_added(
                parsed_chunk, output_items_by_index, item_id_to_output_index
            )
        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            cls._handle_sse_output_text_delta(
                parsed_chunk,
                output_items_by_index,
                item_id_to_output_index,
                accumulated_text,
            )
        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED:
            cls._handle_sse_output_text_annotation_added(
                parsed_chunk, output_items_by_index, item_id_to_output_index
            )
        elif event_type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            cls._handle_sse_function_call_arguments_delta(
                parsed_chunk, output_items_by_index, item_id_to_output_index
            )
        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            cls._handle_sse_output_item_done(
                parsed_chunk, output_items_by_index, item_id_to_output_index
            )

    @classmethod
    def _reconstruct_output_from_sse(cls, body_text: str) -> List[Dict[str, Any]]:
        output_items_by_index: Dict[int, Dict[str, Any]] = {}
        item_id_to_output_index: Dict[str, int] = {}
        accumulated_text: Dict[tuple[str, int], str] = {}

        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue

            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk or stripped_chunk == STREAM_SSE_DONE_STRING:
                continue

            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed_chunk, dict):
                continue

            cls._apply_sse_event_to_output_items(
                parsed_chunk,
                output_items_by_index,
                item_id_to_output_index,
                accumulated_text,
            )

        return [
            output_items_by_index[idx]
            for idx in sorted(output_items_by_index)
            if isinstance(output_items_by_index[idx], dict)
        ]

    @staticmethod
    def _build_completed_sse_response(
        response_payload: Dict[str, Any], reconstructed_output: List[Dict[str, Any]]
    ) -> ResponsesAPIResponse:
        response_payload = dict(response_payload)
        if not response_payload.get("output") and reconstructed_output:
            response_payload["output"] = reconstructed_output

        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(
                response_payload["created_at"]
            )

        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        authenticator = self._get_authenticator_for_request(
            api_base=litellm_params.get("api_base") if litellm_params else None,
            litellm_params=litellm_params,
        )
        try:
            access_token = authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        return {**default_headers, **headers}

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        response_api_optional_request_params.pop("metadata", None)
        input, extracted_instructions = self._extract_instructions_from_input(input)
        request = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        if extracted_instructions:
            request["instructions"] = extracted_instructions
        else:
            request.setdefault("instructions", "")
        request["store"] = False
        request["stream"] = True
        include = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "tools",
            "tool_choice",
            "reasoning",
            "prompt_cache_key",
            "previous_response_id",
            "truncation",
        }

        return {k: v for k, v in request.items() if k in allowed_keys}

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        content_type = (raw_response.headers or {}).get("content-type", "")
        body_text = raw_response.text or ""
        if "text/event-stream" not in content_type.lower():
            trimmed_body = body_text.lstrip()
            if not (
                trimmed_body.startswith("event:")
                or trimmed_body.startswith("data:")
                or "\nevent:" in body_text
                or "\ndata:" in body_text
            ):
                return super().transform_response_api_response(
                    model=model,
                    raw_response=raw_response,
                    logging_obj=logging_obj,
                )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response = None
        error_message = None
        reconstructed_output = self._reconstruct_output_from_sse(body_text)
        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue
            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk:
                continue
            if stripped_chunk == STREAM_SSE_DONE_STRING:
                break
            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_chunk, dict):
                continue
            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    completed_response = self._build_completed_sse_response(
                        response_payload=response_payload,
                        reconstructed_output=reconstructed_output,
                    )
                break
            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_obj = parsed_chunk.get("error") or (
                    parsed_chunk.get("response") or {}
                ).get("error")
                if error_obj is not None:
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get("message") or str(error_obj)
                    else:
                        error_message = str(error_obj)

        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers
        return completed_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        request_api_base = api_base
        if request_api_base is None and litellm_params is not None:
            request_api_base = litellm_params.get("api_base")
        authenticator = self._get_authenticator_for_request(
            api_base=request_api_base, litellm_params=litellm_params
        )
        api_base = request_api_base or authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False
