from __future__ import annotations

import asyncio
import inspect
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import LeanRunner
from ..types import ProofState, TacticResult

_PANTOGRAPH_BOOTSTRAP_ATTEMPTED = False


@dataclass(frozen=True)
class _LeanDojoConfig:
    project_path: Optional[Path]
    imports: Optional[list[str]]
    timeout_s: Optional[float]


class LeanDojoRunner(LeanRunner):
    def __init__(
        self,
        project_path: Optional[Path] = None,
        imports: Optional[list[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self._Server = _load_pantograph_server()
        if timeout_s is None:
            timeout_s = _default_dojo_timeout()
        self._config = _LeanDojoConfig(project_path=project_path, imports=imports, timeout_s=timeout_s)
        self._server = None
        self._states: dict[str, Any] = {}

    def start(self, file_path: Path, theorem: str) -> ProofState:
        text = file_path.read_text(encoding="utf-8")
        file_imports, body_text = _split_imports(text)
        strip_imports, cleaned_body = _strip_import_lines(body_text)
        merged_imports = file_imports + [imp for imp in strip_imports if imp not in file_imports]
        imports = self._config.imports or merged_imports or None

        if self._server is None:
            project_path = self._config.project_path or _find_project_root(file_path)
            self._server = _create_server(
                self._Server, project_path, imports, timeout_s=self._config.timeout_s
            )

        text_for_dojo = cleaned_body if merged_imports else text
        try:
            target_index = _find_target_sorry_index(text_for_dojo, theorem)
        except RuntimeError as exc:
            found = _list_theorems(text_for_dojo)
            if len(found) == 1:
                print(
                    f"Warning: theorem `{theorem}` not found; using `{found[0]}` from {file_path}."
                )
                target_index = _find_target_sorry_index(text_for_dojo, found[0])
            else:
                found_msg = "none" if not found else ", ".join(found[:10])
                if len(found) > 10:
                    found_msg += ", ..."
                raise RuntimeError(
                    f"{exc} Found theorems: {found_msg}. "
                    "Ensure the file contains the target theorem."
                ) from exc
        units = _load_sorries(self._server, text_for_dojo)
        if target_index >= len(units):
            raise RuntimeError(
                f"Expected at least {target_index + 1} `sorry` goals, found {len(units)}. "
                "Ensure the target theorem contains a `sorry`."
            )
        goal_state = _extract_goal_state(units[target_index])
        if goal_state is None:
            raise RuntimeError("LeanDojo did not return a goal state for the target `sorry`.")
        return self._wrap_state(goal_state)

    def apply(self, state: ProofState, tactic: str, timeout_s: float) -> TacticResult:
        if self._server is None:
            return TacticResult(
                ok=False,
                new_state=None,
                error="LeanDojoRunner not initialized. Call start() first.",
                is_solved=False,
            )
        goal_state = self._states.get(state.key)
        if goal_state is None:
            return TacticResult(
                ok=False,
                new_state=None,
                error="Proof state expired or unknown.",
                is_solved=False,
            )
        _ = timeout_s
        try:
            if _goal_tactic_uses_index(self._server):
                new_state = self._server.goal_tactic(goal_state, 0, tactic)
            else:
                new_state = self._server.goal_tactic(goal_state, tactic)
        except Exception as exc:  # pragma: no cover - depends on external Lean server
            return TacticResult(
                ok=False,
                new_state=None,
                error=str(exc),
                is_solved=False,
            )
        if _is_solved(new_state):
            return TacticResult(ok=True, new_state=None, error=None, is_solved=True)
        return TacticResult(ok=True, new_state=self._wrap_state(new_state), error=None, is_solved=False)

    def close(self) -> None:
        if self._server is None:
            return
        try:
            self._server.close()
        except Exception:
            return

    def _wrap_state(self, goal_state: Any) -> ProofState:
        key = _state_key(goal_state)
        self._states[key] = goal_state
        return ProofState(key=key, pretty=str(goal_state))


def _load_pantograph_server() -> Any:
    try:
        from pantograph import Server  # type: ignore
        return Server
    except ImportError as exc:
        if _auto_bootstrap_pantograph():
            try:
                from pantograph import Server  # type: ignore
                return Server
            except ImportError:
                pass
        raise RuntimeError(_pantograph_missing_message()) from exc


def _pantograph_missing_message() -> str:
    return (
        "Pantograph is not installed in this Python environment. "
        "Run `ulam -lean` to install LeanDojo/Pantograph, or install manually with "
        "`python3 -m pip install --user lean-dojo-v2 "
        "git+https://github.com/stanford-centaur/PyPantograph`."
    )


def _auto_bootstrap_pantograph() -> bool:
    global _PANTOGRAPH_BOOTSTRAP_ATTEMPTED
    if _PANTOGRAPH_BOOTSTRAP_ATTEMPTED:
        return False
    _PANTOGRAPH_BOOTSTRAP_ATTEMPTED = True

    enabled = os.environ.get("ULAM_AUTO_INSTALL_PANTOGRAPH", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False

    print("[lean] Pantograph is missing; attempting one-time install...")
    packages = [
        "lean-dojo-v2",
        "git+https://github.com/stanford-centaur/PyPantograph",
    ]
    ok, flags, output = _pip_install_with_fallback(packages)
    if ok:
        if flags:
            print(f"[lean] Pantograph install completed with flags: {' '.join(flags)}")
        else:
            print("[lean] Pantograph install completed.")
        return True
    short = output[-400:] if output else "install failed"
    print(f"[lean] automatic Pantograph install failed: {short}")
    return False


def _pip_install_with_fallback(packages: list[str]) -> tuple[bool, list[str], str]:
    base = [sys.executable or "python3", "-m", "pip", "install"]
    attempts = _pip_install_attempt_flags()
    last_output = ""
    for flags in attempts:
        cmd = [*base, *flags, *packages]
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=600,
            )
        except Exception as exc:
            last_output = str(exc)
            continue
        if proc.returncode == 0:
            return True, flags, ""
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        last_output = output
        if _is_externally_managed_pip_error(output):
            continue
        return False, flags, output
    return False, attempts[-1], last_output


def _pip_install_attempt_flags() -> list[list[str]]:
    prefers_break = os.environ.get("ULAM_BREAK_SYSTEM_PACKAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    in_venv = (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or bool(os.environ.get("VIRTUAL_ENV"))
    )
    if in_venv:
        if prefers_break:
            return [[], ["--break-system-packages"]]
        return [[], ["--break-system-packages"]]
    default_order = [
        [],
        ["--user"],
        ["--break-system-packages"],
        ["--break-system-packages", "--user"],
    ]
    if not prefers_break:
        return default_order
    return [
        ["--break-system-packages", "--user"],
        ["--break-system-packages"],
        ["--user"],
        [],
    ]


def _is_externally_managed_pip_error(output: str) -> bool:
    text = output.lower()
    return "externally-managed-environment" in text or "externally managed" in text


def _state_key(goal_state: Any) -> str:
    if hasattr(goal_state, "state_id"):
        return str(getattr(goal_state, "state_id"))
    return f"state:{id(goal_state)}"


def _is_solved(goal_state: Any) -> bool:
    goals = getattr(goal_state, "goals", None)
    if goals is None:
        return False
    return len(goals) == 0


def _create_server(
    Server: Any,
    project_path: Path,
    imports: Optional[list[str]],
    timeout_s: Optional[float] = None,
) -> Any:
    imports = _sanitize_imports(imports)
    kwargs: dict[str, Any] = {}
    sig = inspect.signature(Server)
    if "project_path" in sig.parameters:
        kwargs["project_path"] = str(project_path)
    if imports and "imports" in sig.parameters:
        kwargs["imports"] = imports
    if timeout_s is not None and "timeout" in sig.parameters:
        kwargs["timeout"] = float(timeout_s)
    env_updates: dict[str, str | None] = {}
    if _lean_options_invalid(os.environ.get("LEAN_OPTIONS")):
        print("[lean] LEAN_OPTIONS contains invalid entries; ignoring for this run.")
        env_updates["LEAN_OPTIONS"] = ""
    if _lean_options_invalid(os.environ.get("LEAN_CORE_OPTIONS")):
        print("[lean] LEAN_CORE_OPTIONS contains invalid entries; ignoring for this run.")
        env_updates["LEAN_CORE_OPTIONS"] = ""
    quiet_stderr = _pantograph_stderr_quiet()
    if env_updates:
        with _temporary_env(env_updates):
            with _redirect_stderr_to_devnull(quiet_stderr):
                return Server(**kwargs)
    with _redirect_stderr_to_devnull(quiet_stderr):
        return Server(**kwargs)


def _pantograph_stderr_quiet() -> bool:
    raw = os.environ.get("ULAM_PANTOGRAPH_QUIET", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _default_dojo_timeout() -> Optional[float]:
    import os

    env = os.environ.get("ULAM_DOJO_TIMEOUT_S")
    if env:
        try:
            return float(env)
        except Exception:
            pass
    try:
        from ..config import load_config

        cfg = load_config()
        return float(cfg.get("lean", {}).get("dojo_timeout_s", 180))
    except Exception:
        return 180.0


def _goal_tactic_uses_index(server: Any) -> bool:
    try:
        sig = inspect.signature(server.goal_tactic)
    except Exception:
        return False
    params = list(sig.parameters.values())
    if len(params) < 3:
        return False
    second = params[1].name
    third = params[2].name if len(params) > 2 else ""
    if second in {"goal_id", "goalId", "goal", "idx", "index"}:
        return True
    if third == "tactic" and second in {"goal", "goal_id", "idx", "index"}:
        return True
    return False


def _find_project_root(file_path: Path) -> Path:
    for parent in [file_path.parent, *file_path.parents]:
        if (
            (parent / "lakefile.lean").exists()
            or (parent / "lakefile.toml").exists()
            or (parent / "lean-toolchain").exists()
        ):
            return parent
    return file_path.parent


def _find_target_sorry_index(text: str, theorem: str) -> int:
    match = re.search(rf"\b(theorem|lemma|example)\s+{re.escape(theorem)}\b", text)
    if match is None:
        raise RuntimeError(f"Could not find theorem `{theorem}` in file.")

    pattern = re.compile(r"\bsorry\b")
    prefix = text[: match.start()]
    prefix_count = len(pattern.findall(prefix))
    suffix = text[match.start() :]
    suffix_matches = list(pattern.finditer(suffix))
    if not suffix_matches:
        raise RuntimeError(
            f"No `sorry` found in theorem `{theorem}`. "
            "Add a `sorry` placeholder for LeanDojoRunner."
        )
    return prefix_count


def _list_theorems(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\b(?:theorem|lemma|example)\s+([A-Za-z0-9_']+)\b", text):
        names.append(match.group(1))
    return names


def _split_imports(text: str) -> tuple[list[str], str]:
    lines = text.splitlines()
    imports: list[str] = []
    body: list[str] = []
    in_header = True
    in_block_comment = False

    for line in lines:
        stripped = line.strip()
        if in_header:
            if in_block_comment:
                if "-/" in stripped:
                    in_block_comment = False
                continue
            if not stripped:
                continue
            if stripped.startswith("--"):
                continue
            if stripped.startswith("/-"):
                if "-/" not in stripped:
                    in_block_comment = True
                continue
            if stripped.startswith("import "):
                cleaned = _strip_inline_comment(stripped)
                remainder = cleaned[len("import ") :].strip()
                if remainder:
                    imports.extend(part for part in remainder.split() if part)
                continue
            in_header = False
        body.append(line)

    return _sanitize_imports(imports) or [], "\n".join(body).lstrip("\n")


def _strip_import_lines(text: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import "):
            cleaned = _strip_inline_comment(stripped)
            remainder = cleaned[len("import ") :].strip()
            if remainder:
                imports.extend(part for part in remainder.split() if part)
            continue
        body.append(line)
    return _sanitize_imports(imports) or [], "\n".join(body).lstrip("\n")


def _strip_inline_comment(line: str) -> str:
    if "--" in line:
        line = line.split("--", 1)[0]
    if "/-" in line:
        line = line.split("/-", 1)[0]
    return line.rstrip()


def _sanitize_imports(imports: Optional[list[str]]) -> Optional[list[str]]:
    if not imports:
        return imports
    cleaned: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for mod in imports:
        mod = mod.strip()
        if not mod:
            continue
        if not re.match(r"^[A-Za-z0-9_.']+$", mod):
            invalid.append(mod)
            continue
        if mod in seen:
            continue
        seen.add(mod)
        cleaned.append(mod)
    if invalid:
        print(f"[lean] ignoring invalid import tokens: {', '.join(invalid)}")
    return cleaned or None


def _lean_options_invalid(value: str | None) -> bool:
    if not value:
        return False
    tokens = re.split(r"\s+", value.strip())
    if not tokens:
        return False
    return any(tok and "=" not in tok for tok in tokens)


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    original: dict[str, str | None] = {}
    for key, val in updates.items():
        original[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    try:
        yield
    finally:
        for key, old in original.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def _redirect_stderr_to_devnull(enabled: bool):
    if not enabled:
        yield
        return
    saved_fd: int | None = None
    devnull_fd: int | None = None
    try:
        saved_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
    except Exception:
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass
        if saved_fd is not None:
            try:
                os.close(saved_fd)
            except Exception:
                pass
        yield
        return
    try:
        yield
    finally:
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 2)
            except Exception:
                pass
            try:
                os.close(saved_fd)
            except Exception:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass


def _load_sorries(server: Any, text: str) -> list[Any]:
    if hasattr(server, "load_sorry"):
        units = server.load_sorry(text)
        return list(_coerce_units(units))
    if hasattr(server, "load_sorry_async"):
        units = asyncio.run(server.load_sorry_async(text))
        return list(_coerce_units(units))
    raise RuntimeError("Pantograph server does not expose load_sorry/load_sorry_async.")


def _coerce_units(units: Any) -> list[Any]:
    if units is None:
        return []
    if isinstance(units, list):
        return units
    return [units]


def _extract_goal_state(unit: Any) -> Any:
    if unit is None:
        return None
    if hasattr(unit, "goal_state"):
        return getattr(unit, "goal_state")
    return None
