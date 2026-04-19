#!/usr/bin/env python3
import argparse
import base64
import json
import secrets
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Event
from typing import Any, Dict, Optional

import httpx

CHATGPT_AUTH_BASE = "https://auth.openai.com"
CHATGPT_DEVICE_CODE_URL = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token"
CHATGPT_OAUTH_TOKEN_URL = f"{CHATGPT_AUTH_BASE}/oauth/token"
CHATGPT_OAUTH_AUTHORIZE_URL = f"{CHATGPT_AUTH_BASE}/oauth/authorize"
CHATGPT_DEVICE_VERIFY_URL = f"{CHATGPT_AUTH_BASE}/codex/device"
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_AUTH_DIR = Path.home() / "litellm" / "auth"
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_TIMEOUT_SECONDS = 15 * 60
DEFAULT_CALLBACK_PORT = 1455
OAUTH_SCOPE = (
    "openid profile email offline_access "
    "api.connectors.read api.connectors.invoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ChatGPT/Codex browser OAuth callback flow or device-code flow "
            "and store the auth JSON in ~/litellm/auth/<name>.json."
        )
    )
    parser.add_argument(
        "--name",
        help="Output auth file name without extension, e.g. 50oedmju",
    )
    parser.add_argument(
        "--out",
        help="Explicit output path. Overrides --name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Polling timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--device-code",
        action="store_true",
        help="Use device code login instead of the browser callback flow.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CALLBACK_PORT,
        help=f"Local callback port for browser login. Default: {DEFAULT_CALLBACK_PORT}",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the browser. Print the URL only.",
    )
    return parser.parse_args()


def decode_jwt_claims(token: Optional[str]) -> Dict[str, Any]:
    if not token:
        return {}
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {}


def get_expires_at(access_token: Optional[str]) -> Optional[int]:
    claims = decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return int(exp)
    return None


def extract_account_id(token: Optional[str]) -> Optional[str]:
    claims = decode_jwt_claims(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def resolve_output_path(name: Optional[str], explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    filename = f"{name or secrets.token_urlsafe(6).lower()}.json"
    return DEFAULT_AUTH_DIR / filename


def build_pkce_codes() -> Dict[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("utf-8").rstrip("=")
    challenge_digest = __import__("hashlib").sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(challenge_digest).decode("utf-8").rstrip("=")
    return {
        "code_verifier": verifier,
        "code_challenge": challenge,
    }


def build_authorize_url(redirect_uri: str, pkce: Dict[str, str], state: str) -> str:
    query = {
        "response_type": "code",
        "client_id": CHATGPT_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPE,
        "code_challenge": pkce["code_challenge"],
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": "codex_cli_rs",
    }
    return f"{CHATGPT_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"


def request_device_code(client: httpx.Client) -> Dict[str, str]:
    response = client.post(
        CHATGPT_DEVICE_CODE_URL,
        json={"client_id": CHATGPT_CLIENT_ID},
    )
    response.raise_for_status()
    data = response.json()

    device_auth_id = data.get("device_auth_id")
    user_code = data.get("user_code") or data.get("usercode")
    interval = str(data.get("interval") or DEFAULT_POLL_INTERVAL_SECONDS)
    if not device_auth_id or not user_code:
        raise RuntimeError(f"Device code response missing fields: {data}")

    return {
        "device_auth_id": device_auth_id,
        "user_code": user_code,
        "interval": interval,
    }


def poll_for_authorization_code(
    client: httpx.Client, device_code: Dict[str, str], timeout_seconds: int
) -> Dict[str, str]:
    interval = max(int(device_code.get("interval", DEFAULT_POLL_INTERVAL_SECONDS)), 1)
    started_at = time.time()

    while time.time() - started_at < timeout_seconds:
        response = client.post(
            CHATGPT_DEVICE_TOKEN_URL,
            json={
                "device_auth_id": device_code["device_auth_id"],
                "user_code": device_code["user_code"],
            },
        )

        if response.status_code == 200:
            data = response.json()
            if all(
                key in data
                for key in ("authorization_code", "code_challenge", "code_verifier")
            ):
                return data

        if response.status_code not in (403, 404):
            response.raise_for_status()

        time.sleep(max(interval, DEFAULT_POLL_INTERVAL_SECONDS))

    raise TimeoutError("Timed out waiting for device authorization.")


def exchange_code_for_tokens(
    client: httpx.Client, code_data: Dict[str, str]
) -> Dict[str, str]:
    redirect_uri = f"{CHATGPT_AUTH_BASE}/deviceauth/callback"
    body = (
        "grant_type=authorization_code"
        f"&code={code_data['authorization_code']}"
        f"&redirect_uri={redirect_uri}"
        f"&client_id={CHATGPT_CLIENT_ID}"
        f"&code_verifier={code_data['code_verifier']}"
    )
    response = client.post(
        CHATGPT_OAUTH_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        content=body,
    )
    response.raise_for_status()
    data = response.json()

    if not all(key in data for key in ("access_token", "refresh_token", "id_token")):
        raise RuntimeError(f"Token exchange response missing fields: {data}")

    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "id_token": data["id_token"],
    }


def exchange_authorization_code_for_tokens(
    client: httpx.Client,
    authorization_code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Dict[str, str]:
    body = (
        "grant_type=authorization_code"
        f"&code={urllib.parse.quote(authorization_code, safe='')}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
        f"&client_id={urllib.parse.quote(CHATGPT_CLIENT_ID, safe='')}"
        f"&code_verifier={urllib.parse.quote(code_verifier, safe='')}"
    )
    response = client.post(
        CHATGPT_OAUTH_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        content=body,
    )
    response.raise_for_status()
    data = response.json()
    if not all(key in data for key in ("access_token", "refresh_token", "id_token")):
        raise RuntimeError(f"Token exchange response missing fields: {data}")
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "id_token": data["id_token"],
    }


def build_auth_record(tokens: Dict[str, str]) -> Dict[str, Any]:
    access_token = tokens["access_token"]
    id_token = tokens["id_token"]
    return {
        "access_token": access_token,
        "refresh_token": tokens["refresh_token"],
        "id_token": id_token,
        "expires_at": get_expires_at(access_token),
        "account_id": extract_account_id(id_token) or extract_account_id(access_token),
        "device_code_requested_at": time.time(),
    }


def write_auth_file(output_path: Path, auth_record: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(auth_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_path.chmod(0o600)


def run_browser_login(output_path: Path, timeout_seconds: int, port: int, open_browser: bool) -> None:
    pkce = build_pkce_codes()
    state = secrets.token_urlsafe(24)
    redirect_uri = f"http://localhost:{port}/auth/callback"
    auth_url = build_authorize_url(redirect_uri, pkce, state)
    callback_event = Event()
    callback_data: Dict[str, str] = {}

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
                return

            query = urllib.parse.parse_qs(parsed.query)
            if query.get("state", [""])[0] != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                callback_data["error"] = "state_mismatch"
                callback_event.set()
                return

            if query.get("error", [""])[0]:
                self.send_response(400)
                self.end_headers()
                message = query.get("error_description", [query["error"][0]])[0]
                self.wfile.write(message.encode("utf-8", errors="replace"))
                callback_data["error"] = message
                callback_event.set()
                return

            code = query.get("code", [""])[0]
            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                callback_data["error"] = "missing_authorization_code"
                callback_event.set()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                (
                    "<html><body><h2>Login completed</h2>"
                    "<p>You can close this tab and return to the terminal.</p>"
                    "</body></html>"
                ).encode("utf-8")
            )
            callback_data["code"] = code
            callback_event.set()

    server = ThreadingHTTPServer(("127.0.0.1", port), OAuthCallbackHandler)
    server.timeout = 0.5

    print("ChatGPT / Codex browser login")
    print(f"1. Local callback server: http://localhost:{port}/auth/callback")
    print("2. Open the URL below in your browser and complete login:")
    print(auth_url)
    print(f"3. Auth file will be written to: {output_path}")

    if open_browser:
        webbrowser.open(auth_url)

    started_at = time.time()
    try:
        while not callback_event.is_set():
            server.handle_request()
            if time.time() - started_at > timeout_seconds:
                raise TimeoutError("Timed out waiting for OAuth callback.")
    finally:
        server.server_close()

    if callback_data.get("error"):
        raise RuntimeError(f"OAuth callback failed: {callback_data['error']}")

    authorization_code = callback_data.get("code")
    if not authorization_code:
        raise RuntimeError("OAuth callback did not return an authorization code.")

    with httpx.Client(timeout=30.0) as client:
        tokens = exchange_authorization_code_for_tokens(
            client=client,
            authorization_code=authorization_code,
            redirect_uri=redirect_uri,
            code_verifier=pkce["code_verifier"],
        )

    auth_record = build_auth_record(tokens)
    write_auth_file(output_path, auth_record)


def main() -> int:
    args = parse_args()
    output_path = resolve_output_path(args.name, args.out)

    if args.device_code:
        with httpx.Client(timeout=30.0) as client:
            device_code = request_device_code(client)
            print("ChatGPT / Codex device login")
            print(f"1. Open: {CHATGPT_DEVICE_VERIFY_URL}")
            print(f"2. Enter code: {device_code['user_code']}")
            print("3. Finish the browser login, then wait for this script to continue.")
            print(f"Auth file will be written to: {output_path}")

            code_data = poll_for_authorization_code(client, device_code, args.timeout)
            tokens = exchange_code_for_tokens(client, code_data)

        auth_record = build_auth_record(tokens)
        write_auth_file(output_path, auth_record)
    else:
        run_browser_login(
            output_path=output_path,
            timeout_seconds=args.timeout,
            port=args.port,
            open_browser=not args.no_open,
        )
        auth_record = json.loads(output_path.read_text(encoding="utf-8"))

    print(f"Login succeeded. Auth JSON written to: {output_path}")
    if auth_record.get("account_id"):
        print(f"Derived account_id: {auth_record['account_id']}")
    if auth_record.get("expires_at"):
        print(f"Access token expires_at: {auth_record['expires_at']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
