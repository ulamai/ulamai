from __future__ import annotations

from typing import Iterable

from .base import LLMClient
from .prompt import build_prompt, parse_tactics
from .cli_utils import codex_exec
from ..types import ProofState


class CodexCLIClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        timeout_s: float | None = None,
        heartbeat_s: float | None = None,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._heartbeat_s = heartbeat_s
        _ensure_cmd("codex")

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
        text = codex_exec(
            system,
            user,
            model=self._model,
            timeout_s=self._timeout_s,
            heartbeat_s=self._heartbeat_s,
        )
        return parse_tactics(text, k)


def _ensure_cmd(cmd: str) -> None:
    from shutil import which

    if which(cmd) is None:
        raise RuntimeError(f"{cmd} CLI not found on PATH")
