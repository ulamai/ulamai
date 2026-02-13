from __future__ import annotations

from typing import Iterable

from .base import LLMClient
from .prompt import build_prompt, parse_tactics
from .cli_utils import codex_exec
from ..types import ProofState


class CodexCLIClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        self._model = model
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
        text = codex_exec(system, user, model=self._model)
        return parse_tactics(text, k)


def _ensure_cmd(cmd: str) -> None:
    from shutil import which

    if which(cmd) is None:
        raise RuntimeError(f"{cmd} CLI not found on PATH")
