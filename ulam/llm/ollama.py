from __future__ import annotations

import json
import urllib.error
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
        endpoints: list[str] = []
        base_url = self._base_url
        if base_url.endswith("/api"):
            base_url = base_url[: -len("/api")]
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
            endpoints.append(f"{base_url}/v1/chat/completions")
        endpoints.append(f"{base_url}/api/chat")
        endpoints.append(f"{base_url}/v1/chat/completions")
        seen = set()
        endpoints = [url for url in endpoints if not (url in seen or seen.add(url))]
        last_error: Exception | None = None
        for url in endpoints:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                content = _extract_content(raw)
                return parse_tactics(content, k)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (404, 405):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Ollama response missing content")


def _extract_content(raw: str) -> str:
    data = json.loads(raw)
    message = data.get("message")
    if isinstance(message, dict) and "content" in message:
        return message["content"]
    choices = data.get("choices") or []
    if choices:
        choice = choices[0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    if "response" in data and isinstance(data["response"], str):
        return data["response"]
    raise RuntimeError("Ollama response missing content")
