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
    ) -> list[str]:
        _ = failed_tactic
        _ = error
        return self.propose(state, retrieved, k, instruction=instruction, context=context)
