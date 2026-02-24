from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class ProofState:
    key: str
    pretty: str


@dataclass(frozen=True)
class TacticResult:
    ok: bool
    new_state: Optional[ProofState]
    error: Optional[str]
    is_solved: bool


@dataclass(frozen=True)
class ProofStep:
    state_key: str
    state_pretty: str
    state_hash: Optional[str]
    tactic: str
    ok: bool
    error: Optional[str]
    new_state_key: Optional[str]
    new_state_hash: Optional[str]
    solved: bool
    cached: bool = False
    elapsed_ms: Optional[int] = None
    error_kind: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    file_path: Path
    theorem: str
    max_steps: int
    beam_width: int
    suggestions_per_state: int
    timeout_s: float
    repair_attempts: int
    seed: int
    trace_path: Optional[Path]
    retriever_k: int = 8
    autop: bool = True
    instruction: Optional[str] = None
    context: Optional[list[str]] = None
    verbose: bool = False
    on_progress: Optional[Callable[[list[str]], None]] = None
