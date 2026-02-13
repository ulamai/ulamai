from __future__ import annotations

import json
import urllib.request
from typing import Iterable

from .base import LLMClient
from .prompt import build_prompt, parse_tactics
from ..types import ProofState


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def propose(
        self,
        state: ProofState,
        retrieved: Iterable[str],
        k: int,
        instruction: str | None = None,
        context: Iterable[str] | None = None,
        mode: str = "tactic",
    ) -> list[str]:
        system, user = build_prompt(
            state, retrieved, k, instruction=instruction, context=context, mode=mode
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        content = _extract_content(raw)
        return parse_tactics(content, k)


def _extract_content(raw: str) -> str:
    data = json.loads(raw)
    message = data.get("message")
    if not message or "content" not in message:
        raise RuntimeError("Ollama response missing content")
    return message["content"]
