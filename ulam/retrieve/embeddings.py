from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

from ..types import ProofState
from .base import Retriever


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        return _extract_embeddings(raw)


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        premises: Iterable[str],
        embedder: EmbeddingClient,
        cache_path: Optional[Path] = None,
        batch_size: int = 16,
    ) -> None:
        self._premises = [p.strip() for p in premises if p.strip()]
        self._embedder = embedder
        self._cache_path = cache_path
        self._batch_size = max(1, batch_size)
        self._embeddings, self._norms = self._load_or_compute_embeddings()

    def retrieve(self, state: ProofState, k: int) -> list[str]:
        if not self._premises or k <= 0:
            return []
        query_vec = self._embedder.embed([state.pretty])[0]
        qnorm = _norm(query_vec)
        scored: list[tuple[float, str]] = []
        for premise, vec, vnorm in zip(self._premises, self._embeddings, self._norms):
            denom = vnorm * qnorm
            if denom == 0.0:
                score = 0.0
            else:
                score = _dot(vec, query_vec) / denom
            scored.append((score, premise))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [premise for _, premise in scored[:k]]

    def _load_or_compute_embeddings(self) -> tuple[list[list[float]], list[float]]:
        if not self._premises:
            return [], []
        cache_key = _hash_premises(self._premises)
        if self._cache_path and self._cache_path.exists():
            cached = _load_cache(self._cache_path)
            if cached and cached.get("hash") == cache_key:
                embeddings = cached.get("embeddings", [])
                if len(embeddings) == len(self._premises):
                    return embeddings, [_norm(vec) for vec in embeddings]

        embeddings = []
        for i in range(0, len(self._premises), self._batch_size):
            batch = self._premises[i : i + self._batch_size]
            embeddings.extend(self._embedder.embed(batch))

        if self._cache_path:
            _write_cache(self._cache_path, cache_key, embeddings)

        return embeddings, [_norm(vec) for vec in embeddings]


def _extract_embeddings(raw: str) -> list[list[float]]:
    data = json.loads(raw)
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError("Embedding response missing data")
    rows = sorted(rows, key=lambda item: item.get("index", 0))
    embeddings = []
    for row in rows:
        embedding = row.get("embedding")
        if embedding is None:
            raise RuntimeError("Embedding row missing embedding")
        embeddings.append(embedding)
    return embeddings


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vec: list[float]) -> float:
    return math.sqrt(_dot(vec, vec))


def _hash_premises(premises: list[str]) -> str:
    joined = "\n".join(premises).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _load_cache(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, cache_key: str, embeddings: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hash": cache_key,
        "embeddings": embeddings,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
