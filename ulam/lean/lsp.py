from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from shutil import which
from typing import Any


def lean_lsp_check(
    file_path: Path,
    project_path: Path | None = None,
    timeout_s: float = 60.0,
) -> str | None:
    diagnostics, error = lean_lsp_diagnostics(
        file_path=file_path,
        project_path=project_path,
        timeout_s=timeout_s,
    )
    if error:
        return error
    errors = [item for item in diagnostics if item.get("severity") == "error"]
    if not errors:
        return None
    return _format_diagnostics(errors)


def lean_lsp_diagnostics(
    file_path: Path,
    project_path: Path | None = None,
    timeout_s: float = 60.0,
) -> tuple[list[dict[str, object]], str | None]:
    target = Path(file_path).expanduser().resolve()
    if not target.exists():
        return [], f"Lean file not found: {target}"
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return [], f"Failed to read Lean file: {exc}"

    cmd = _lean_lsp_cmd()
    if cmd is None:
        return [], "Lean LSP server not found. Install Lean and ensure `lean` or `lake` is on PATH."

    project = Path(project_path).expanduser().resolve() if project_path else None
    cwd = str(project) if project else str(target.parent)
    env = os.environ.copy()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return [], f"Failed to start Lean LSP server: {exc}"

    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        _terminate_process(proc)
        return [], "Failed to start Lean LSP server (missing stdio pipes)."

    reader = _MessageReader(proc.stdout)
    reader.start()
    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(proc.stderr, stderr_lines),
        daemon=True,
    )
    stderr_thread.start()

    client = _LSPClient(proc.stdin, reader)
    uri = target.as_uri()
    root_uri = (project or target.parent).as_uri()
    timeout_val = max(2.0, float(timeout_s))

    try:
        client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {"name": "ulam", "version": "0"},
                "rootUri": root_uri,
                "capabilities": {},
                "workspaceFolders": [{"uri": root_uri, "name": (project or target.parent).name}],
            },
            timeout_s=min(15.0, timeout_val),
        )
        client.notify("initialized", {})
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "lean",
                    "version": 1,
                    "text": text,
                }
            },
        )
        rows = client.wait_for_diagnostics(uri=uri, timeout_s=timeout_val)
        normalized = [_normalize_diagnostic(item) for item in rows if isinstance(item, dict)]
        normalized = [item for item in normalized if item is not None]
        return normalized, None
    except Exception as exc:
        stderr_preview = "\n".join(line for line in stderr_lines if line.strip())[-400:]
        if stderr_preview:
            return [], f"{exc}\n{stderr_preview}"
        return [], str(exc) or repr(exc)
    finally:
        try:
            client.request("shutdown", None, timeout_s=2.0)
        except Exception:
            pass
        try:
            client.notify("exit", None)
        except Exception:
            pass
        _terminate_process(proc)


def _lean_lsp_cmd() -> list[str] | None:
    if which("lake") is not None:
        return ["lake", "env", "lean", "--server"]
    if which("lean") is not None:
        return ["lean", "--server"]
    return None


def _read_stderr(stream, sink: list[str]) -> None:
    while True:
        chunk = stream.readline()
        if not chunk:
            return
        try:
            line = chunk.decode("utf-8", errors="replace")
        except Exception:
            continue
        sink.append(line.rstrip("\n"))


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        return


def _format_diagnostics(rows: list[dict[str, object]], max_items: int = 20) -> str:
    lines: list[str] = []
    for item in rows[:max_items]:
        line = int(item.get("line", 1) or 1)
        col = int(item.get("col", 1) or 1)
        msg = str(item.get("message", "") or "Lean error").strip()
        sev = str(item.get("severity", "error") or "error").strip().upper()
        lines.append(f"{line}:{col}: [{sev}] {msg}")
    if len(rows) > max_items:
        lines.append(f"... and {len(rows) - max_items} more diagnostics")
    return "\n".join(lines).strip()


def _normalize_diagnostic(payload: dict[str, Any]) -> dict[str, object] | None:
    rng = payload.get("range")
    if not isinstance(rng, dict):
        return None
    start = rng.get("start")
    end = rng.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    line = int(start.get("line", 0) or 0) + 1
    col = int(start.get("character", 0) or 0) + 1
    end_line = int(end.get("line", 0) or 0) + 1
    end_col = int(end.get("character", 0) or 0) + 1
    message = str(payload.get("message", "") or "").strip()
    severity_code = int(payload.get("severity", 1) or 1)
    severity = {
        1: "error",
        2: "warning",
        3: "info",
        4: "hint",
    }.get(severity_code, "unknown")
    return {
        "line": line,
        "col": col,
        "end_line": end_line,
        "end_col": end_col,
        "severity": severity,
        "message": message,
    }


class _MessageReader:
    def __init__(self, stream) -> None:
        self._stream = stream
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._backlog: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get(self, timeout_s: float) -> dict[str, Any] | None:
        if self._backlog:
            return self._backlog.pop(0)
        try:
            return self._queue.get(timeout=max(0.01, timeout_s))
        except queue.Empty:
            return None

    def push_back_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._backlog.extend(rows)

    def _run(self) -> None:
        while True:
            headers: dict[str, str] = {}
            line = self._stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace")
            if not text.strip():
                continue
            while text.strip():
                if ":" in text:
                    key, value = text.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
                line = self._stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace")
            length_raw = headers.get("content-length", "0")
            try:
                length = int(length_raw)
            except Exception:
                length = 0
            if length <= 0:
                continue
            body = self._stream.read(length)
            if not body:
                return
            try:
                payload = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(payload, dict):
                self._queue.put(payload)


class _LSPClient:
    def __init__(self, stdin, reader: _MessageReader) -> None:
        self._stdin = stdin
        self._reader = reader
        self._next_id = 1

    def notify(self, method: str, params: Any) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout_s: float) -> Any:
        req_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.time() + max(0.5, timeout_s)
        buffered: list[dict[str, Any]] = []
        while time.time() < deadline:
            remaining = max(0.01, deadline - time.time())
            msg = self._reader.get(timeout_s=min(0.25, remaining))
            if msg is None:
                continue
            if msg.get("id") == req_id:
                self._reader.push_back_many(buffered)
                if "error" in msg:
                    _raise_lsp_error(method, msg.get("error"))
                return msg.get("result")
            buffered.append(msg)
        self._reader.push_back_many(buffered)
        raise RuntimeError(f"Lean LSP request timed out: {method}")

    def wait_for_diagnostics(
        self,
        *,
        uri: str,
        timeout_s: float,
        version: int | None = None,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + max(0.5, timeout_s)
        buffered: list[dict[str, Any]] = []
        while time.time() < deadline:
            remaining = max(0.01, deadline - time.time())
            msg = self._reader.get(timeout_s=min(0.25, remaining))
            if msg is None:
                continue
            if msg.get("method") != "textDocument/publishDiagnostics":
                buffered.append(msg)
                continue
            params = msg.get("params", {})
            if not isinstance(params, dict):
                continue
            if str(params.get("uri", "")).strip() != uri:
                buffered.append(msg)
                continue
            if version is not None:
                payload_version = params.get("version")
                if isinstance(payload_version, int) and payload_version != version:
                    buffered.append(msg)
                    continue
            rows = params.get("diagnostics", [])
            self._reader.push_back_many(buffered)
            if isinstance(rows, list):
                return rows
            return []
        self._reader.push_back_many(buffered)
        return []

    def _write(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._stdin.write(header)
        self._stdin.write(body)
        self._stdin.flush()


def _raise_lsp_error(method: str, payload: Any) -> None:
    if isinstance(payload, dict):
        message = str(payload.get("message", "") or "").strip()
        code = payload.get("code")
        if message:
            raise RuntimeError(f"Lean LSP {method} failed ({code}): {message}")
    raise RuntimeError(f"Lean LSP {method} failed.")
