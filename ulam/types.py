from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    tactic: str
    ok: bool
    error: Optional[str]
    new_state_key: Optional[str]
    solved: bool
    cached: bool = False


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
    autop: bool = True
    instruction: Optional[str] = None
    context: Optional[list[str]] = None
    verbose: bool = False
