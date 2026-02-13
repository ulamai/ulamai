from __future__ import annotations

from typing import Iterable

from .base import LLMClient
from ..types import ProofState


class MockLLMClient(LLMClient):
    def propose(
        self,
        state: ProofState,
        retrieved: Iterable[str],
        k: int,
        instruction: str | None = None,
        context: Iterable[str] | None = None,
    ) -> list[str]:
        _ = instruction
        _ = context
        suggestions: list[str] = []
        retrieved_text = " ".join(retrieved).lower()
        if "irrational_sqrt_two" in state.pretty or "irrational_sqrt_two" in retrieved_text:
            suggestions.append("simpa using irrational_sqrt_two")
        suggestions.extend([
            "exact?",
            "simp",
            "aesop",
            "linarith",
            "ring",
        ])
        return suggestions[:k]
