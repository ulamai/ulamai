from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which
from typing import Any


_OAUTH_CLIENT_ID_RE = re.compile(r"const\s+OAUTH_CLIENT_ID\s*=\s*'([^']+)'")
_OAUTH_CLIENT_SECRET_RE = re.compile(r"const\s+OAUTH_CLIENT_SECRET\s*=\s*'([^']+)'")
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
_REDIRECT_PATH = "/oauth2callback"
_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


@dataclass
class _GeminiOAuthClient:
    client_id: str
    client_secret: str


class _CallbackState:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.payload: dict[str, str] = {}

    def set(self, payload: dict[str, str]) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.payload = payload
            self._event.set()

    def wait(self, timeout_s: float) -> bool:
        return self._event.wait(timeout_s)


def run_gemini_login() -> None:
    if which("gemini") is None:
        raise RuntimeError("gemini CLI not found on PATH")

    if has_gemini_oauth_credentials():
        _cache_google_account_from_saved_creds()
        print("Gemini OAuth credentials already exist; using current login.")
        return

    client: _GeminiOAuthClient | None = None
    discovery_error: Exception | None = None
    try:
        client = _load_oauth_client()
    except Exception as exc:
        discovery_error = exc

    if client is None:
        if _run_gemini_native_login():
            _cache_google_account_from_saved_creds()
            return
        message = (
            "Gemini native login did not produce OAuth credentials "
            f"(expected: {gemini_oauth_creds_path()})."
        )
        if discovery_error is not None:
            raise RuntimeError(f"{discovery_error}\n{message}") from discovery_error
        raise RuntimeError(message)

    state = _random_urlsafe(32)
    code_verifier = _random_urlsafe(64)
    code_challenge = _pkce_s256(code_verifier)
    port = _choose_callback_port()
    redirect_uri = f"http://localhost:{port}{_REDIRECT_PATH}"
    auth_url = _build_auth_url(client.client_id, redirect_uri, state, code_challenge)

    callback_state: _CallbackState | None = _CallbackState()
    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    try:
        server, thread = _start_callback_server(callback_state, port)
    except OSError as exc:
        callback_state = None
        print(f"Could not start local OAuth callback server ({exc}).")
        print("Falling back to manual callback URL/code entry.")
    try:
        print("Starting Gemini OAuth login...")
        print("Open this URL if the browser does not open automatically:")
        print(auth_url)
        if not _browser_suppressed():
            opened = False
            try:
                opened = webbrowser.open(auth_url, new=1, autoraise=True)
            except Exception:
                opened = False
            if not opened:
                print("Could not open the browser automatically.")
        else:
            print("NO_BROWSER is set; open the URL manually.")

        if callback_state is None:
            payload = _prompt_manual_callback(auth_url, state)
        elif not callback_state.wait(timeout_s=180.0):
            payload = _prompt_manual_callback(auth_url, state)
        else:
            payload = callback_state.payload
    finally:
        if server and thread:
            _stop_callback_server(server, thread)

    code = _validate_callback_payload(payload, state)
    tokens = _exchange_code_for_tokens(client, code, code_verifier, redirect_uri)
    _save_gemini_credentials(tokens)
    _cache_google_account_if_available(tokens.get("access_token", ""))
    print("Gemini OAuth login completed.")


def _build_auth_url(client_id: str, redirect_uri: str, state: str, code_challenge: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(_OAUTH_SCOPES),
        "access_type": "offline",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _start_callback_server(
    callback_state: _CallbackState,
    port: int,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = _make_callback_handler(callback_state)
    host = os.environ.get("OAUTH_CALLBACK_HOST", "localhost")
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_callback_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()
    except Exception:
        pass
    thread.join(timeout=1.0)


def _make_callback_handler(callback_state: _CallbackState):
    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != _REDIRECT_PATH:
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(parsed.query)
            payload: dict[str, str] = {}
            for key in ("code", "state", "error", "error_description"):
                values = query.get(key)
                if values:
                    payload[key] = values[0]
            callback_state.set(payload)
            if "error" in payload:
                body = (
                    "<html><body><h1>Gemini OAuth Failed</h1>"
                    "<p>You can return to Ulam and check the terminal for details.</p>"
                    "</body></html>"
                )
            else:
                body = (
                    "<html><body><h1>Gemini OAuth Complete</h1>"
                    "<p>You can close this tab and return to Ulam.</p>"
                    "</body></html>"
                )
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
            return

    return OAuthCallbackHandler


def _prompt_manual_callback(auth_url: str, expected_state: str) -> dict[str, str]:
    print()
    print("Automatic callback did not complete.")
    print("Finish sign-in in your browser, then paste either:")
    print("1) Full callback URL (http://localhost:.../oauth2callback?...), or")
    print("2) The authorization code value.")
    print(f"If needed, reopen this URL:\n{auth_url}")
    raw = input("Callback URL or code: ").strip()
    if not raw:
        raise RuntimeError("No callback URL or code provided.")
    return _parse_manual_callback(raw, expected_state)


def _parse_manual_callback(raw: str, expected_state: str) -> dict[str, str]:
    if "://" in raw:
        parsed = urllib.parse.urlsplit(raw)
        query = urllib.parse.parse_qs(parsed.query)
        payload = {k: v[0] for k, v in query.items() if v}
    elif "=" in raw:
        query = urllib.parse.parse_qs(raw.lstrip("?"))
        payload = {k: v[0] for k, v in query.items() if v}
    else:
        payload = {"code": raw}
    state = payload.get("state")
    if state and state != expected_state:
        raise RuntimeError("OAuth callback state mismatch.")
    return payload


def _validate_callback_payload(payload: dict[str, str], expected_state: str) -> str:
    if not payload:
        raise RuntimeError("Gemini OAuth callback did not return data.")
    if "error" in payload:
        desc = payload.get("error_description", "")
        if desc:
            raise RuntimeError(f"Gemini OAuth error: {payload['error']} ({desc})")
        raise RuntimeError(f"Gemini OAuth error: {payload['error']}")
    state = payload.get("state")
    if state and state != expected_state:
        raise RuntimeError("OAuth callback state mismatch.")
    code = payload.get("code", "").strip()
    if not code:
        raise RuntimeError("Gemini OAuth callback did not include an authorization code.")
    return code


def _exchange_code_for_tokens(
    client: _GeminiOAuthClient,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    data = urllib.parse.urlencode(
        {
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini OAuth token exchange failed: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini OAuth token exchange failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError("Gemini OAuth token exchange returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini OAuth token exchange returned invalid payload.")
    if payload.get("error"):
        desc = payload.get("error_description", "")
        if desc:
            raise RuntimeError(f"Gemini OAuth error: {payload['error']} ({desc})")
        raise RuntimeError(f"Gemini OAuth error: {payload['error']}")
    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Gemini OAuth did not return an access token.")
    expires_in = payload.get("expires_in")
    if "expiry_date" not in payload and isinstance(expires_in, (int, float)):
        payload["expiry_date"] = int(time.time() * 1000) + int(expires_in * 1000)
    payload.setdefault("token_type", "Bearer")
    return payload


def _save_gemini_credentials(tokens: dict[str, Any]) -> None:
    path = gemini_oauth_creds_path()
    existing = _load_existing_creds(path)
    if "refresh_token" not in tokens and isinstance(existing.get("refresh_token"), str):
        tokens["refresh_token"] = existing["refresh_token"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokens, indent=2, ensure_ascii=True), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _load_existing_creds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def gemini_oauth_creds_path() -> Path:
    return Path.home() / ".gemini" / "oauth_creds.json"


def has_gemini_oauth_credentials() -> bool:
    path = gemini_oauth_creds_path()
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    return (isinstance(access_token, str) and bool(access_token.strip())) or (
        isinstance(refresh_token, str) and bool(refresh_token.strip())
    )


def _cache_google_account_if_available(access_token: str) -> None:
    if not access_token:
        return
    req = urllib.request.Request(
        _USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    email = payload.get("email")
    if not isinstance(email, str) or not email.strip():
        return
    _cache_google_account(email.strip())


def _cache_google_account_from_saved_creds() -> None:
    path = gemini_oauth_creds_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        return
    _cache_google_account_if_available(access_token)


def _cache_google_account(email: str) -> None:
    path = Path.home() / ".gemini" / "google_accounts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any]
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        data = existing if isinstance(existing, dict) else {}
    except Exception:
        data = {}
    active = data.get("active")
    old = data.get("old")
    old_accounts = [value for value in old if isinstance(value, str)] if isinstance(old, list) else []
    if isinstance(active, str) and active and active != email and active not in old_accounts:
        old_accounts.append(active)
    old_accounts = [value for value in old_accounts if value != email]
    data["active"] = email
    data["old"] = old_accounts
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_oauth_client() -> _GeminiOAuthClient:
    env_client_id = os.environ.get("ULAM_GEMINI_OAUTH_CLIENT_ID", "").strip()
    env_client_secret = os.environ.get("ULAM_GEMINI_OAUTH_CLIENT_SECRET", "").strip()
    if env_client_id and env_client_secret:
        return _GeminiOAuthClient(client_id=env_client_id, client_secret=env_client_secret)
    oauth_js = _find_gemini_oauth2_js()
    if oauth_js:
        parsed = _parse_oauth_client_from_js(oauth_js)
        if parsed:
            return parsed
    raise RuntimeError(
        "Could not discover Gemini OAuth client credentials. "
        "Set ULAM_GEMINI_OAUTH_CLIENT_ID and ULAM_GEMINI_OAUTH_CLIENT_SECRET, "
        "or set ULAM_GEMINI_OAUTH2_JS to Gemini CLI oauth2.js."
    )


def _run_gemini_native_login() -> bool:
    print(
        "Could not auto-discover Gemini OAuth client credentials; "
        "falling back to Gemini CLI native login."
    )
    env = os.environ.copy()
    env.setdefault("OAUTH_CALLBACK_HOST", "localhost")
    probe_cmd = ["gemini", "-p", "Reply with exactly: ok"]
    try:
        subprocess.run(probe_cmd, check=False, env=env)
    except Exception:
        pass
    if has_gemini_oauth_credentials():
        print("Gemini CLI native login completed.")
        return True
    if os.isatty(0) and os.isatty(1):
        print("Launching interactive Gemini CLI login. Complete sign-in, then exit.")
        try:
            subprocess.run(["gemini"], check=False, env=env)
        except Exception:
            pass
    if has_gemini_oauth_credentials():
        print("Gemini CLI native login completed.")
        return True
    return False


def _find_gemini_oauth2_js() -> Path | None:
    env_path = os.environ.get("ULAM_GEMINI_OAUTH2_JS", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
    exe = which("gemini")
    candidates: list[Path] = []
    if exe:
        resolved = Path(exe).resolve()
        package_root = _guess_gemini_package_root(resolved)
        candidates.extend(
            [
                package_root / "node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                resolved.parent.parent
                / "lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                resolved.parent.parent
                / "node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                resolved.parent.parent / "lib/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
            ]
        )
        for parent in list(resolved.parents)[:10]:
            candidates.extend(
                [
                    parent / "node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                    parent
                    / "node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                    parent
                    / "lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                    parent
                    / "libexec/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js",
                ]
            )
    candidates.extend(
        [
            Path("/opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js"),
            Path("/usr/local/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js"),
        ]
    )
    for cellar in (
        Path("/opt/homebrew/Cellar/gemini-cli"),
        Path("/usr/local/Cellar/gemini-cli"),
    ):
        if cellar.exists():
            candidates.extend(
                cellar.glob(
                    "*/libexec/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/oauth2.js"
                )
            )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _guess_gemini_package_root(resolved: Path) -> Path:
    # Homebrew/npm typically resolve to .../@google/gemini-cli/dist/index.js.
    if resolved.name == "index.js" and resolved.parent.name == "dist":
        return resolved.parent.parent
    return resolved.parent


def _parse_oauth_client_from_js(path: Path) -> _GeminiOAuthClient | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    match_id = _OAUTH_CLIENT_ID_RE.search(text)
    match_secret = _OAUTH_CLIENT_SECRET_RE.search(text)
    if not match_id or not match_secret:
        return None
    return _GeminiOAuthClient(client_id=match_id.group(1), client_secret=match_secret.group(1))


def _pkce_s256(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _random_urlsafe(num_bytes: int) -> str:
    token = os.urandom(num_bytes)
    return base64.urlsafe_b64encode(token).decode("utf-8").rstrip("=")


def _choose_callback_port() -> int:
    raw = os.environ.get("OAUTH_CALLBACK_PORT", "").strip()
    if raw:
        try:
            port = int(raw)
            if 1 <= port <= 65535:
                return port
        except Exception:
            pass
        raise RuntimeError(f"Invalid OAUTH_CALLBACK_PORT value: {raw}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _browser_suppressed() -> bool:
    value = os.environ.get("NO_BROWSER", "")
    return value.lower() in {"1", "true", "yes", "on"}
