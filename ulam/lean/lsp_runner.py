from __future__ import annotations

import hashlib
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import LeanRunner
from .lsp import (
    _format_diagnostics,
    _lean_lsp_cmd,
    _LSPClient,
    _MessageReader,
    _normalize_diagnostic,
    _read_stderr,
    _terminate_process,
)
from ..types import ProofState, TacticResult


@dataclass(frozen=True)
class _DeclSlice:
    theorem: str
    head: str
    tail: str
    indent: str
    indent_cols: int
    goal_hint: str


@dataclass(frozen=True)
class _ScriptEval:
    ok: bool
    solved: bool
    error: str | None = None
    goal_pretty: str | None = None


class _LeanLspSession:
    def __init__(
        self,
        *,
        file_path: Path,
        project_path: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._file_path = file_path.expanduser().resolve()
        self._project_path = project_path.expanduser().resolve() if project_path else None
        self._timeout_s = max(2.0, float(timeout_s))
        self._uri = self._file_path.as_uri()
        root = self._project_path or self._file_path.parent
        self._root_uri = root.as_uri()
        self._root_name = root.name or "root"

        self._proc: subprocess.Popen | None = None
        self._reader: _MessageReader | None = None
        self._client: _LSPClient | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._opened = False
        self._version = 0

    def open_text(self, text: str, timeout_s: float | None = None) -> list[dict[str, object]]:
        if self._opened:
            return self.update_text(text, timeout_s=timeout_s)
        self._ensure_started()
        if self._client is None:
            raise RuntimeError("Lean LSP client is not initialized.")
        self._version = 1
        self._client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": self._uri,
                    "languageId": "lean",
                    "version": self._version,
                    "text": text,
                }
            },
        )
        self._opened = True
        return self._wait_for_diagnostics(timeout_s=timeout_s)

    def update_text(self, text: str, timeout_s: float | None = None) -> list[dict[str, object]]:
        if not self._opened:
            return self.open_text(text, timeout_s=timeout_s)
        if self._client is None:
            raise RuntimeError("Lean LSP client is not initialized.")
        self._version += 1
        self._client.notify(
            "textDocument/didChange",
            {
                "textDocument": {
                    "uri": self._uri,
                    "version": self._version,
                },
                "contentChanges": [{"text": text}],
            },
        )
        return self._wait_for_diagnostics(timeout_s=timeout_s)

    def plain_goal(
        self,
        *,
        line: int,
        col: int,
        timeout_s: float | None = None,
    ) -> str | None:
        if self._client is None or not self._opened:
            return None
        timeout_val = self._resolve_timeout(timeout_s)
        params = {
            "textDocument": {"uri": self._uri},
            "position": {
                "line": max(0, int(line)),
                "character": max(0, int(col)),
            },
        }
        try:
            payload = self._client.request("$/lean/plainGoal", params, timeout_s=timeout_val)
            text = _coerce_goal_text(payload)
            if text:
                return text
        except Exception:
            pass
        # Fallback for servers that do not expose Lean RPC goal API.
        try:
            hover = self._client.request("textDocument/hover", params, timeout_s=timeout_val)
            return _coerce_hover_text(hover)
        except Exception:
            return None

    def stderr_preview(self, max_chars: int = 400) -> str:
        text = "\n".join(line for line in self._stderr_lines if line.strip())
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.request("shutdown", None, timeout_s=2.0)
            except Exception:
                pass
            try:
                self._client.notify("exit", None)
            except Exception:
                pass
        if self._proc is not None:
            _terminate_process(self._proc)
        self._proc = None
        self._reader = None
        self._client = None
        self._stderr_thread = None
        self._opened = False

    def _resolve_timeout(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return self._timeout_s
        return max(2.0, float(timeout_s))

    def _wait_for_diagnostics(self, timeout_s: float | None = None) -> list[dict[str, object]]:
        if self._client is None:
            raise RuntimeError("Lean LSP client is not initialized.")
        timeout_val = self._resolve_timeout(timeout_s)
        rows = self._client.wait_for_diagnostics(
            uri=self._uri,
            timeout_s=timeout_val,
            version=self._version,
        )
        normalized = [_normalize_diagnostic(item) for item in rows if isinstance(item, dict)]
        return [item for item in normalized if item is not None]

    def _ensure_started(self) -> None:
        if self._client is not None:
            return
        cmd = _lean_lsp_cmd()
        if cmd is None:
            raise RuntimeError(
                "Lean LSP server not found. Install Lean and ensure `lean` or `lake` is on PATH."
            )
        cwd = str(self._project_path or self._file_path.parent)
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
            raise RuntimeError(f"Failed to start Lean LSP server: {exc}") from exc
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            _terminate_process(proc)
            raise RuntimeError("Failed to start Lean LSP server (missing stdio pipes).")
        reader = _MessageReader(proc.stdout)
        reader.start()
        stderr_thread = threading.Thread(
            target=_read_stderr,
            args=(proc.stderr, self._stderr_lines),
            daemon=True,
        )
        stderr_thread.start()
        client = _LSPClient(proc.stdin, reader)
        timeout_val = min(15.0, self._timeout_s)
        client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {"name": "ulam", "version": "0"},
                "rootUri": self._root_uri,
                "capabilities": {},
                "workspaceFolders": [{"uri": self._root_uri, "name": self._root_name}],
            },
            timeout_s=timeout_val,
        )
        client.notify("initialized", {})
        self._proc = proc
        self._reader = reader
        self._client = client
        self._stderr_thread = stderr_thread


class LeanLspRunner(LeanRunner):
    def __init__(
        self,
        *,
        project_path: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._project_path = project_path
        self._timeout_s = max(2.0, float(timeout_s))
        self._session: _LeanLspSession | None = None
        self._decl: _DeclSlice | None = None
        self._state_scripts: dict[str, list[str]] = {}
        self._eval_cache: dict[tuple[str, ...], _ScriptEval] = {}

    def start(self, file_path: Path, theorem: str) -> ProofState:
        path = file_path.expanduser().resolve()
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to read Lean file: {exc}") from exc
        decl = _locate_tactic_declaration(text, theorem)
        if decl is None:
            raise RuntimeError(
                "Lean LSP runner could not locate a `:= by` declaration for the theorem. "
                "Use a declaration with explicit tactic proof body."
            )
        session = _LeanLspSession(
            file_path=path,
            project_path=self._project_path,
            timeout_s=self._timeout_s,
        )
        try:
            guard_text, probe_line, probe_col = _render_candidate(
                decl=decl,
                script=[],
                with_sorry_guard=True,
            )
            diagnostics = session.open_text(guard_text, timeout_s=self._timeout_s)
            errors = _error_diagnostics(diagnostics)
            if errors:
                raise RuntimeError(_format_diagnostics(errors))
            goal_pretty = session.plain_goal(
                line=probe_line,
                col=probe_col,
                timeout_s=self._timeout_s,
            )
            if not goal_pretty:
                goal_pretty = decl.goal_hint or f"goal: {theorem}"
            state = ProofState(key=_script_state_key([]), pretty=goal_pretty)
            self._session = session
            self._decl = decl
            self._state_scripts = {state.key: []}
            self._eval_cache = {tuple(): _ScriptEval(ok=True, solved=False, goal_pretty=state.pretty)}
            return state
        except Exception:
            session.close()
            raise

    def apply(self, state: ProofState, tactic: str, timeout_s: float) -> TacticResult:
        if self._session is None or self._decl is None:
            return TacticResult(
                ok=False,
                new_state=None,
                error="LeanLspRunner is not initialized. Call start() first.",
                is_solved=False,
            )
        base = self._state_scripts.get(state.key)
        if base is None:
            return TacticResult(
                ok=False,
                new_state=None,
                error=f"Unknown state key: {state.key}",
                is_solved=False,
            )
        tactic_text = tactic.strip()
        if not tactic_text:
            return TacticResult(
                ok=False,
                new_state=None,
                error="Empty tactic.",
                is_solved=False,
            )
        script = [*base, tactic_text]
        result = self._evaluate_script(script=script, timeout_s=timeout_s)
        if not result.ok:
            return TacticResult(
                ok=False,
                new_state=None,
                error=result.error or "Lean LSP tactic failed.",
                is_solved=False,
            )
        if result.solved:
            return TacticResult(
                ok=True,
                new_state=None,
                error=None,
                is_solved=True,
            )
        key = _script_state_key(script)
        pretty = result.goal_pretty or f"goals pending after {len(script)} tactic(s)"
        new_state = ProofState(key=key, pretty=pretty)
        self._state_scripts[key] = script
        return TacticResult(
            ok=True,
            new_state=new_state,
            error=None,
            is_solved=False,
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._decl = None
        self._state_scripts.clear()
        self._eval_cache.clear()

    def _evaluate_script(self, *, script: list[str], timeout_s: float) -> _ScriptEval:
        key = tuple(script)
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        if self._session is None or self._decl is None:
            result = _ScriptEval(ok=False, solved=False, error="Lean LSP session is not active.")
            self._eval_cache[key] = result
            return result
        timeout_val = max(self._timeout_s, float(timeout_s))

        guard_text, probe_line, probe_col = _render_candidate(
            decl=self._decl,
            script=script,
            with_sorry_guard=True,
        )
        guard_diagnostics = self._session.update_text(guard_text, timeout_s=timeout_val)
        guard_errors = _error_diagnostics(guard_diagnostics)
        if guard_errors:
            result = _ScriptEval(
                ok=False,
                solved=False,
                error=_format_diagnostics(guard_errors),
            )
            self._eval_cache[key] = result
            return result

        goal_pretty = self._session.plain_goal(
            line=probe_line,
            col=probe_col,
            timeout_s=timeout_val,
        )

        strict_text, _, _ = _render_candidate(
            decl=self._decl,
            script=script,
            with_sorry_guard=False,
        )
        strict_diagnostics = self._session.update_text(strict_text, timeout_s=timeout_val)
        strict_errors = _error_diagnostics(strict_diagnostics)
        if not strict_errors:
            result = _ScriptEval(ok=True, solved=True)
            self._eval_cache[key] = result
            return result

        if _only_unsolved_goal_errors(strict_errors):
            goal = goal_pretty or _goal_from_unsolved_errors(strict_errors)
            result = _ScriptEval(ok=True, solved=False, goal_pretty=goal)
            self._eval_cache[key] = result
            return result

        result = _ScriptEval(
            ok=False,
            solved=False,
            error=_format_diagnostics(strict_errors),
        )
        self._eval_cache[key] = result
        return result


def _locate_tactic_declaration(text: str, theorem: str) -> _DeclSlice | None:
    target = str(theorem or "").strip()
    if not target:
        return None
    pattern = re.compile(
        rf"(?ms)(^[ \t]*(?:theorem|lemma|example)\s+{re.escape(target)}\b.*?:=\s*by[ \t]*\n?)(?P<body>.*?)(?=^[ \t]*(?:theorem|lemma|example)\b|\Z)"
    )
    match = pattern.search(text)
    if match is None:
        return None
    body = match.group("body")
    head = text[: match.start("body")]
    tail = text[match.end("body") :]
    indent = "  "
    for line in body.splitlines():
        stripped = line.lstrip(" \t")
        if not stripped:
            continue
        indent = line[: len(line) - len(stripped)] or "  "
        break
    indent = indent.replace("\t", "  ")
    header = match.group(1)
    goal_hint = _extract_goal_hint(header, target)
    return _DeclSlice(
        theorem=target,
        head=head,
        tail=tail,
        indent=indent,
        indent_cols=len(indent),
        goal_hint=goal_hint,
    )


def _extract_goal_hint(header: str, theorem: str) -> str:
    goal = ""
    match = re.search(r":\s*(.*?)\s*:=\s*by\s*$", header.strip(), flags=re.S)
    if match is not None:
        goal = str(match.group(1) or "").strip()
    if goal:
        return f"⊢ {goal}"
    return f"goal: {theorem}"


def _render_candidate(
    *,
    decl: _DeclSlice,
    script: list[str],
    with_sorry_guard: bool,
) -> tuple[str, int, int]:
    lines: list[str] = []
    for tactic in script:
        for raw in tactic.splitlines():
            line = raw.rstrip()
            if line:
                lines.append(line)
    if with_sorry_guard:
        lines.append("all_goals sorry")
    if not lines:
        lines.append("skip")
    body = "\n".join(f"{decl.indent}{line}" for line in lines)
    head = decl.head.rstrip("\n") + "\n"
    tail = decl.tail.lstrip("\n")
    rendered = head + body
    if tail:
        rendered += "\n" + tail
    probe_line = head.count("\n") + max(0, len(lines) - 1)
    probe_col = decl.indent_cols
    return rendered, probe_line, probe_col


def _script_state_key(script: list[str]) -> str:
    if not script:
        return "lsp:root"
    data = "\n".join(script).encode("utf-8", errors="ignore")
    digest = hashlib.sha1(data).hexdigest()[:16]
    return f"lsp:{len(script)}:{digest}"


def _error_diagnostics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for item in rows:
        severity = str(item.get("severity", "") or "").strip().lower()
        if severity == "error":
            out.append(item)
    return out


def _only_unsolved_goal_errors(rows: list[dict[str, object]]) -> bool:
    if not rows:
        return False
    return all(_is_unsolved_goal_message(str(item.get("message", "") or "")) for item in rows)


def _goal_from_unsolved_errors(rows: list[dict[str, object]]) -> str:
    for item in rows:
        text = str(item.get("message", "") or "").strip()
        if _is_unsolved_goal_message(text):
            return text
    return "unsolved goals"


def _is_unsolved_goal_message(message: str) -> bool:
    text = message.strip().lower()
    if not text:
        return False
    return (
        "unsolved goals" in text
        or "goals remaining" in text
        or "remaining goals" in text
        or "goals are not solved" in text
    )


def _coerce_goal_text(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, dict):
        for key in ("goal", "message", "text", "value"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        goals = payload.get("goals")
        if isinstance(goals, list):
            lines = [str(item).strip() for item in goals if str(item).strip()]
            if lines:
                return "\n\n".join(lines)
    if isinstance(payload, list):
        lines = [str(item).strip() for item in payload if str(item).strip()]
        if lines:
            return "\n\n".join(lines)
    return None


def _coerce_hover_text(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    contents = payload.get("contents")
    if isinstance(contents, str):
        text = contents.strip()
        return text or None
    if isinstance(contents, dict):
        for key in ("value", "contents"):
            value = contents.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(contents, list):
        rows: list[str] = []
        for item in contents:
            if isinstance(item, str) and item.strip():
                rows.append(item.strip())
            elif isinstance(item, dict):
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    rows.append(value.strip())
        if rows:
            return "\n\n".join(rows)
    return None
