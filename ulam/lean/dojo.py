from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import LeanRunner
from ..types import ProofState, TacticResult


@dataclass(frozen=True)
class _LeanDojoConfig:
    project_path: Optional[Path]
    imports: Optional[list[str]]


class LeanDojoRunner(LeanRunner):
    def __init__(self, project_path: Optional[Path] = None, imports: Optional[list[str]] = None) -> None:
        try:
            from pantograph import Server  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Pantograph is not installed. Install LeanDojo-v2 and PyPantograph "
                "(e.g., `pip install lean-dojo-v2` and `pip install PyPantograph`)."
            ) from exc

        self._Server = Server
        self._config = _LeanDojoConfig(project_path=project_path, imports=imports)
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
            self._server = _create_server(self._Server, project_path, imports)

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
            new_state = self._server.goal_tactic(goal_state, 0, tactic)
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


def _create_server(Server: Any, project_path: Path, imports: Optional[list[str]]) -> Any:
    kwargs: dict[str, Any] = {}
    sig = inspect.signature(Server)
    if "project_path" in sig.parameters:
        kwargs["project_path"] = str(project_path)
    if imports and "imports" in sig.parameters:
        kwargs["imports"] = imports
    return Server(**kwargs)


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
                remainder = stripped[len("import ") :].strip()
                if remainder:
                    imports.extend(remainder.split())
                continue
            in_header = False
        body.append(line)

    return imports, "\n".join(body).lstrip("\n")


def _strip_import_lines(text: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import "):
            remainder = stripped[len("import ") :].strip()
            if remainder:
                imports.extend(remainder.split())
            continue
        body.append(line)
    return imports, "\n".join(body).lstrip("\n")


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
