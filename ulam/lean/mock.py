from __future__ import annotations

from pathlib import Path

from .base import LeanRunner
from ..types import ProofState, TacticResult


class MockLeanRunner(LeanRunner):
    def __init__(self) -> None:
        self._file_path: Path | None = None
        self._theorem: str | None = None

    def start(self, file_path: Path, theorem: str) -> ProofState:
        self._file_path = file_path
        self._theorem = theorem
        key = f"mock:{theorem}"
        pretty = f"goal: {theorem}"
        return ProofState(key=key, pretty=pretty)

    def apply(self, state: ProofState, tactic: str, timeout_s: float) -> TacticResult:
        _ = timeout_s
        tactic_norm = " ".join(tactic.strip().split())
        if tactic_norm in {"sorry", "admit"}:
            return TacticResult(
                ok=False,
                new_state=None,
                error="MockLeanRunner rejects 'sorry'/'admit'.",
                is_solved=False,
            )

        if self._theorem and "irrational_sqrt_two" in self._theorem:
            if tactic_norm in {
                "simpa using irrational_sqrt_two",
                "exact irrational_sqrt_two",
                "exact?",
            }:
                return TacticResult(ok=True, new_state=None, error=None, is_solved=True)

        return TacticResult(
            ok=False,
            new_state=None,
            error="MockLeanRunner cannot verify this tactic. Configure a real Lean backend.",
            is_solved=False,
        )

    def close(self) -> None:
        return None
