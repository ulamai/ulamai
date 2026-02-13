from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FormalizationSegment:
    kind: str
    title: str
    body: str


@dataclass(frozen=True)
class FormalizationConfig:
    tex_path: Path
    output_path: Path
    context_files: list[Path]
    max_rounds: int
    max_repairs: int
    max_equivalence_repairs: int
    max_proof_rounds: int
    proof_max_steps: int
    proof_beam: int
    proof_k: int
    proof_timeout_s: float
    proof_repair: int
    lean_project: Optional[Path]
    lean_imports: list[str]
    verbose: bool
    artifact_dir: Optional[Path] = None
    equivalence_checks: bool = True


@dataclass(frozen=True)
class FormalizationResult:
    output_path: Path
    rounds: int
    typecheck_ok: bool
    solved: int
    remaining_sorries: int
    error: Optional[str]
    artifact_dir: Optional[Path] = None
