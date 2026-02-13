from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import ProofState, TacticResult


class LeanRunner(ABC):
    @abstractmethod
    def start(self, file_path: Path, theorem: str) -> ProofState:
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: ProofState, tactic: str, timeout_s: float) -> TacticResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
