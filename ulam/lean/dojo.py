from __future__ import annotations

import asyncio
import inspect
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import LeanRunner
from ..types import ProofState, TacticResult


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
        try:
            from pantograph import Server  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Pantograph is not installed. Install LeanDojo-v2 and PyPantograph "
                "(e.g., `pip install lean-dojo-v2` and `pip install PyPantograph`)."
            ) from exc

        self._Server = Server
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
    if env_updates:
        with _temporary_env(env_updates):
            return Server(**kwargs)
    return Server(**kwargs)


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
