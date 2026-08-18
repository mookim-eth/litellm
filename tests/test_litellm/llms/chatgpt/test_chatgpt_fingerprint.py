import json

from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.common_utils import (
    apply_chatgpt_client_identity_headers,
    apply_chatgpt_fingerprint_client_metadata,
    apply_chatgpt_fingerprint_headers,
    get_chatgpt_fingerprint_mode,
    get_chatgpt_fingerprint_request_headers,
    resolve_chatgpt_fingerprint_ids,
    resolve_chatgpt_fingerprint_ids_with_fallback,
)
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


VALID_ACCOUNT_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
EXPECTED_ACCOUNT_INSTALLATION_ID = "1cfb3ac2-d7f4-4cc0-91d4-04f9dfdccb1f"


def test_chatgpt_fingerprint_mode_is_explicit_opt_in():
    assert get_chatgpt_fingerprint_mode(GenericLiteLLMParams()) == "off"
    assert (
        get_chatgpt_fingerprint_mode(
            GenericLiteLLMParams(chatgpt_fingerprint_mode="invalid")
        )
        == "off"
    )
    assert (
        get_chatgpt_fingerprint_mode(
            GenericLiteLLMParams(chatgpt_fingerprint_mode="session")
        )
        == "device"
    )
    assert (
        get_chatgpt_fingerprint_mode(
            GenericLiteLLMParams(chatgpt_fingerprint_mode="full")
        )
        == "device"
    )


def test_chatgpt_fingerprint_prefers_account_hash_over_persisted_fallback():
    params = GenericLiteLLMParams(chatgpt_fingerprint_mode="session")
    ids = resolve_chatgpt_fingerprint_ids(
        params,
        account_id=VALID_ACCOUNT_ID,
        persisted_installation_id="11111111-1111-4111-8111-111111111111",
    )
    normalized_ids = resolve_chatgpt_fingerprint_ids(
        params,
        account_id="{AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA}",
    )

    assert ids is not None
    assert normalized_ids is not None
    assert ids.mode == "device"
    assert ids.installation_id == EXPECTED_ACCOUNT_INSTALLATION_ID
    assert normalized_ids.installation_id == EXPECTED_ACCOUNT_INSTALLATION_ID


def test_chatgpt_fingerprint_uses_persisted_fallback_for_invalid_account_id():
    get_persisted_installation_id = MagicMock(
        return_value="11111111-1111-4111-8111-111111111111"
    )

    ids = resolve_chatgpt_fingerprint_ids_with_fallback(
        litellm_params=GenericLiteLLMParams(chatgpt_fingerprint_mode="device"),
        account_id="not-an-account-uuid",
        get_persisted_installation_id=get_persisted_installation_id,
    )

    assert ids is not None
    assert ids.installation_id == "11111111-1111-4111-8111-111111111111"
    get_persisted_installation_id.assert_called_once_with()


def test_chatgpt_fingerprint_uses_persisted_fallback_for_missing_account_id():
    get_persisted_installation_id = MagicMock(
        return_value="12121212-1212-4212-8212-121212121212"
    )

    ids = resolve_chatgpt_fingerprint_ids_with_fallback(
        litellm_params=GenericLiteLLMParams(chatgpt_fingerprint_mode="device"),
        account_id=None,
        get_persisted_installation_id=get_persisted_installation_id,
    )

    assert ids is not None
    assert ids.installation_id == "12121212-1212-4212-8212-121212121212"
    get_persisted_installation_id.assert_called_once_with()


def test_chatgpt_fingerprint_valid_account_does_not_read_persisted_fallback():
    get_persisted_installation_id = MagicMock()

    ids = resolve_chatgpt_fingerprint_ids_with_fallback(
        litellm_params=GenericLiteLLMParams(chatgpt_fingerprint_mode="device"),
        account_id=VALID_ACCOUNT_ID,
        get_persisted_installation_id=get_persisted_installation_id,
    )

    assert ids is not None
    assert ids.installation_id == EXPECTED_ACCOUNT_INSTALLATION_ID
    get_persisted_installation_id.assert_not_called()


def test_chatgpt_fingerprint_invalid_explicit_id_fails_closed():
    get_persisted_installation_id = MagicMock()

    ids = resolve_chatgpt_fingerprint_ids_with_fallback(
        litellm_params=GenericLiteLLMParams(
            chatgpt_fingerprint_mode="device",
            chatgpt_fingerprint_installation_id="invalid-explicit-id",
        ),
        account_id=VALID_ACCOUNT_ID,
        get_persisted_installation_id=get_persisted_installation_id,
    )

    assert ids is None
    get_persisted_installation_id.assert_not_called()


def test_chatgpt_fingerprint_headers_fall_back_to_proxy_request_headers():
    params = {
        "proxy_server_request": {
            "headers": {
                "Session-Id": "proxy-session",
                "Thread-Id": "proxy-thread",
                "X-Client-Request-Id": "proxy-thread",
                "X-Codex-Parent-Thread-Id": "proxy-parent",
                "X-Codex-Turn-State": "proxy-state",
                "X-Codex-Turn-Metadata": '{"session_id":"proxy-session"}',
                "X-Codex-Window-Id": "proxy-window",
            }
        }
    }

    client_headers = get_chatgpt_fingerprint_request_headers(params, {})

    assert client_headers == {
        "session-id": "proxy-session",
        "thread-id": "proxy-thread",
        "x-client-request-id": "proxy-thread",
        "x-codex-parent-thread-id": "proxy-parent",
        "x-codex-turn-state": "proxy-state",
        "x-codex-turn-metadata": '{"session_id":"proxy-session"}',
        "x-codex-window-id": "proxy-window",
    }


def test_chatgpt_client_identity_headers_replace_generated_session_alias():
    headers = {
        "authorization": "Bearer provider-token",
        "session_id": "generated-session",
    }
    client_headers = {
        "session-id": "client-session",
        "thread-id": "client-thread",
        "x-client-request-id": "client-thread",
        "x-codex-parent-thread-id": "client-parent",
        "x-codex-turn-state": "client-state",
        "x-codex-turn-metadata": '{"session_id":"client-session"}',
        "x-codex-window-id": "client-window",
    }

    apply_chatgpt_client_identity_headers(headers, client_headers)

    assert headers["authorization"] == "Bearer provider-token"
    assert "session_id" not in headers
    assert headers["session-id"] == "client-session"
    assert headers["thread-id"] == "client-thread"
    assert headers["x-client-request-id"] == "client-thread"
    assert headers["x-codex-parent-thread-id"] == "client-parent"
    assert headers["x-codex-turn-state"] == "client-state"
    assert headers["x-codex-window-id"] == "client-window"


def test_chatgpt_fingerprint_header_and_body_share_ids():
    params = GenericLiteLLMParams(chatgpt_fingerprint_mode="session")
    ids = resolve_chatgpt_fingerprint_ids(
        params,
        persisted_installation_id="22222222-2222-4222-8222-222222222222",
    )
    assert ids is not None

    headers = {
        "x-codex-installation-id": "old-installation",
        "session_id": "client-session",
        "x-codex-turn-metadata": json.dumps(
            {"session_id": "old", "sandbox": "seatbelt"}
        ),
    }
    request = {
        "model": "gpt-5.6-sol",
        "input": [],
        "client_metadata": {
            "session_id": "old",
            "x-codex-turn-metadata": json.dumps(
                {"session_id": "old", "sandbox": "seatbelt"}
            ),
        },
    }
    apply_chatgpt_fingerprint_headers(headers, ids)
    assert apply_chatgpt_fingerprint_client_metadata(request, ids)

    body_metadata = request["client_metadata"]
    header_metadata = json.loads(headers["x-codex-turn-metadata"])
    body_embedded_metadata = json.loads(body_metadata["x-codex-turn-metadata"])
    assert "x-codex-installation-id" not in headers
    assert headers["session_id"] == "client-session"
    assert body_metadata["session_id"] == "old"
    assert "thread_id" not in body_metadata
    assert "turn_id" not in body_metadata
    assert header_metadata["session_id"] == "old"
    assert header_metadata["installation_id"] == ids.installation_id
    assert body_embedded_metadata["installation_id"] == ids.installation_id
    assert body_embedded_metadata["sandbox"] == "seatbelt"


def test_chatgpt_responses_transform_preserves_and_converges_client_metadata():
    auth = MagicMock()
    auth.get_access_token.return_value = "access-token"
    auth.get_account_id.return_value = VALID_ACCOUNT_ID
    auth.get_or_create_installation_id.return_value = (
        "33333333-3333-4333-8333-333333333333"
    )
    with patch(
        "litellm.llms.chatgpt.responses.transformation.Authenticator",
        return_value=auth,
    ):
        config = ChatGPTResponsesAPIConfig()
        params = GenericLiteLLMParams(
            chatgpt_fingerprint_mode="session",
            proxy_server_request={
                "headers": {
                    "Session-Id": "client-session",
                    "Thread-Id": "client-thread",
                    "X-Client-Request-Id": "client-thread",
                    "X-Codex-Window-Id": "client-window",
                    "X-Codex-Turn-Metadata": json.dumps(
                        {"session_id": "client-session", "sandbox": "seatbelt"}
                    ),
                }
            },
        )
        headers = config.validate_environment(
            headers={
                "x-codex-installation-id": "old-installation",
            },
            model="gpt-5.6-sol",
            litellm_params=params,
        )
        request = config.transform_responses_api_request(
            model="gpt-5.6-sol",
            input="hello",
            response_api_optional_request_params={
                "client_metadata": {
                    "x-codex-turn-metadata": json.dumps({"sandbox": "seatbelt"})
                }
            },
            litellm_params=params,
            headers=headers,
        )

    ids = params["chatgpt_fingerprint_ids"]
    assert ids is not None
    assert ids.mode == "device"
    assert ids.installation_id == EXPECTED_ACCOUNT_INSTALLATION_ID
    auth.get_or_create_installation_id.assert_not_called()
    assert "x-codex-installation-id" not in headers
    assert "session_id" not in headers
    assert headers["session-id"] == "client-session"
    assert headers["thread-id"] == "client-thread"
    assert headers["x-client-request-id"] == "client-thread"
    assert headers["x-codex-window-id"] == "client-window"
    header_metadata = json.loads(headers["x-codex-turn-metadata"])
    assert header_metadata["session_id"] == "client-session"
    assert header_metadata["installation_id"] == ids.installation_id
    assert header_metadata["sandbox"] == "seatbelt"
    assert request["client_metadata"]["x-codex-installation-id"] == (
        ids.installation_id
    )
    assert "session_id" not in request["client_metadata"]
    assert "turn_id" not in request["client_metadata"]
    assert json.loads(request["client_metadata"]["x-codex-turn-metadata"])[
        "sandbox"
    ] == "seatbelt"


def test_explicit_installation_id_does_not_create_sidecar():
    auth = MagicMock()
    auth.get_access_token.return_value = "access-token"
    auth.get_account_id.return_value = VALID_ACCOUNT_ID
    with patch(
        "litellm.llms.chatgpt.responses.transformation.Authenticator",
        return_value=auth,
    ):
        params = GenericLiteLLMParams(
            chatgpt_fingerprint_mode="device",
            chatgpt_fingerprint_installation_id=(
                "55555555-5555-4555-8555-555555555555"
            ),
        )
        ChatGPTResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-5.6-sol",
            litellm_params=params,
        )

    auth.get_or_create_installation_id.assert_not_called()
    assert params["chatgpt_fingerprint_ids"].installation_id == (
        "55555555-5555-4555-8555-555555555555"
    )


def test_chatgpt_chat_transform_preserves_proxy_client_session_graph():
    auth = MagicMock()
    auth.get_access_token.return_value = "access-token"
    auth.get_account_id.return_value = "account-a"
    auth.get_api_base.return_value = "https://chatgpt.com/backend-api/codex"
    auth.get_or_create_installation_id.return_value = (
        "66666666-6666-4666-8666-666666666666"
    )
    with patch(
        "litellm.llms.chatgpt.chat.transformation.Authenticator",
        return_value=auth,
    ):
        params = {
            "chatgpt_fingerprint_mode": "device",
            "proxy_server_request": {
                "headers": {
                    "Session-Id": "client-session",
                    "Thread-Id": "client-thread",
                    "X-Codex-Window-Id": "client-window",
                }
            },
        }
        headers = ChatGPTConfig().validate_environment(
            headers={},
            model="gpt-5.6-sol",
            messages=[],
            optional_params={},
            litellm_params=params,
        )

    assert "session_id" not in headers
    assert headers["session-id"] == "client-session"
    assert headers["thread-id"] == "client-thread"
    assert headers["x-codex-window-id"] == "client-window"
    assert "x-codex-installation-id" not in headers
    auth.get_or_create_installation_id.assert_called_once_with()
    assert params["chatgpt_fingerprint_ids"].installation_id == (
        "66666666-6666-4666-8666-666666666666"
    )


def test_chatgpt_compact_uses_direct_installation_header():
    auth = MagicMock()
    auth.get_access_token.return_value = "access-token"
    auth.get_account_id.return_value = "account-a"
    auth.get_or_create_installation_id.return_value = (
        "44444444-4444-4444-8444-444444444444"
    )
    with patch(
        "litellm.llms.chatgpt.responses.transformation.Authenticator",
        return_value=auth,
    ):
        config = ChatGPTResponsesAPIConfig()
        params = GenericLiteLLMParams(chatgpt_fingerprint_mode="device")
        headers = config.validate_environment(
            headers={},
            model="gpt-5.6-sol",
            litellm_params=params,
        )
        url, _ = config.transform_compact_response_api_request(
            model="gpt-5.6-sol",
            input=[],
            response_api_optional_request_params={},
            api_base="https://chatgpt.com/backend-api/codex/responses",
            litellm_params=params,
            headers=headers,
        )

    assert url.endswith("/responses/compact")
    assert headers["x-codex-installation-id"] == (
        "44444444-4444-4444-8444-444444444444"
    )
