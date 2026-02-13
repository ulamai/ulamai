from __future__ import annotations

import json
import urllib.request
from typing import Iterable

from .base import LLMClient
from .prompt import build_prompt, parse_tactics
from ..types import ProofState


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.2,
        timeout_s: float = 60.0,
        api_version: str = "2023-06-01",
    ) -> None:
        if not api_key:
            raise RuntimeError("Anthropic API key or setup-token is required.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._api_version = api_version

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
            "max_tokens": 256,
            "temperature": self._temperature,
            "system": system,
            "messages": [
                {"role": "user", "content": user},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self._api_version,
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        content = _extract_content(raw)
        return parse_tactics(content, k)


def _extract_content(raw: str) -> str:
    data = json.loads(raw)
    content = data.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        if parts:
            return "\n".join(parts)
    if "text" in data and isinstance(data["text"], str):
        return data["text"]
    raise RuntimeError("Anthropic response missing content")
