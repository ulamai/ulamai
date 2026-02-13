from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..types import ProofState


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, state: ProofState, k: int) -> list[str]:
        raise NotImplementedError


class NullRetriever(Retriever):
    def retrieve(self, state: ProofState, k: int) -> list[str]:
        _ = state
        _ = k
        return []


class SimpleRetriever(Retriever):
    def __init__(self, premises: Iterable[str]) -> None:
        self._premises = [p.strip() for p in premises if p.strip()]

    def retrieve(self, state: ProofState, k: int) -> list[str]:
        if not self._premises or k <= 0:
            return []
        haystack = _tokenize(state.pretty)
        scored = []
        for premise in self._premises:
            score = len(haystack.intersection(_tokenize(premise)))
            if score:
                scored.append((score, premise))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [premise for _, premise in scored[:k]]


def _tokenize(text: str) -> set[str]:
    return {token.strip().lower() for token in text.split() if token.strip()}
