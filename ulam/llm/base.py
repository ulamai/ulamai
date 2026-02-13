from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..types import ProofState


class LLMClient(ABC):
    @abstractmethod
    def propose(
        self,
        state: ProofState,
        retrieved: Iterable[str],
        k: int,
        instruction: str | None = None,
        context: Iterable[str] | None = None,
        mode: str = "tactic",
    ) -> list[str]:
        raise NotImplementedError

    def repair(
        self,
        state: ProofState,
        retrieved: Iterable[str],
        failed_tactic: str,
        error: str,
        k: int,
        instruction: str | None = None,
        context: Iterable[str] | None = None,
        mode: str = "tactic",
    ) -> list[str]:
        extra = (
            "Previous tactic failed.\n"
            f"Tactic: {failed_tactic}\n"
            f"Lean error: {error}\n"
        )
        merged = _merge_instruction(instruction, extra)
        return self.propose(state, retrieved, k, instruction=merged, context=context, mode=mode)


def _merge_instruction(instruction: str | None, extra: str) -> str:
    base = instruction.strip() if instruction else ""
    if base:
        return base + "\n\n" + extra
    return extra
