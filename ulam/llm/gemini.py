from __future__ import annotations

import json
import urllib.request
from typing import Iterable

from .base import LLMClient
from .prompt import build_prompt, parse_tactics
from ..types import ProofState


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        model: str = "gemini-3-pro-preview",
        temperature: float = 0.2,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("Gemini API key is required.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
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
            "temperature": self._temperature,
            "max_tokens": 256,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _chat_endpoint(self._base_url),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        content = _extract_content(raw)
        return parse_tactics(content, k)


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/openai") or base.endswith("/openai/v1") or base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_content(raw: str) -> str:
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Gemini response missing choices")
    msg = choices[0]
    if "message" in msg and "content" in msg["message"]:
        return msg["message"]["content"]
    if "text" in msg:
        return msg["text"]
    raise RuntimeError("Gemini response missing content")
