import json
from pathlib import Path

from litellm.llms.chatgpt.authenticator import Authenticator


def test_should_preserve_plan_type_when_refreshing_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "id_token": "old-id",
                "account_id": "account-a",
                "expires_at": 1,
                "plan_type": "plus",
            }
        ),
        encoding="utf-8",
    )
    authenticator = Authenticator(auth_file_path=str(auth_file))

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "id_token": "new-id",
            }

    class _Client:
        def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(
        "litellm.llms.chatgpt.authenticator._get_httpx_client", lambda: _Client()
    )
    monkeypatch.setattr(authenticator, "_get_expires_at", lambda token: 123)
    monkeypatch.setattr(
        authenticator, "_extract_account_id", lambda token: "account-a"
    )

    authenticator._refresh_tokens("old-refresh")

    refreshed_auth = json.loads(auth_file.read_text(encoding="utf-8"))
    assert refreshed_auth["plan_type"] == "plus"
