from __future__ import annotations

import re
from typing import Iterable

from ..types import ProofState


def build_prompt(
    state: ProofState,
    retrieved: Iterable[str],
    k: int,
    instruction: str | None = None,
    context: Iterable[str] | None = None,
    mode: str = "tactic",
) -> tuple[str, str]:
    retrieved_text = "\n".join(f"- {item}" for item in retrieved)
    if not retrieved_text:
        retrieved_text = "- (none)"
    instruction_text = instruction.strip() if instruction else ""
    context_text = "\n\n".join(str(item) for item in (context or []) if str(item).strip())
    system = (
        "You are a Lean 4 theorem prover. "
        "Only output tactic lines. No explanations."
    )
    user = (
        "Current proof state:\n"
        f"{state.pretty}\n\n"
        "Retrieved premises:\n"
        f"{retrieved_text}\n\n"
    )
    if instruction_text:
        user += "User instruction:\n" + instruction_text + "\n\n"
    if context_text:
        user += "Context files:\n" + context_text + "\n\n"
    if mode == "script":
        user += (
            "You are writing a short Lean tactic script to execute IN ORDER.\n"
            "- Each line must be a complete Lean tactic command.\n"
            "- Later lines may depend on names introduced earlier.\n"
            "- Avoid multi-line blocks, `case` blocks, or bullet syntax.\n"
            "- If you use `by`, complete it on the same line (e.g., `by simp`).\n"
            "- Do not include commentary, numbering, or code fences.\n"
            f"Return up to {k} line(s), one tactic per line."
        )
    else:
        user += (
            "Suggest the next single-line tactic.\n"
            "- Avoid multi-line blocks, `case` blocks, or bullet syntax.\n"
            "- If you use `by`, complete it on the same line (e.g., `by simp`).\n"
            f"Return exactly {k} line(s), one tactic per line."
        )
    return system, user


def parse_tactics(text: str, k: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tactics = []
    for line in lines:
        if line.startswith("```"):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        cleaned = line.strip()
        if re.search(r":=\s*by\s*$", cleaned):
            continue
        if cleaned.endswith("=>"):
            continue
        if cleaned in {"case", "cases", "calc"}:
            continue
        if cleaned.startswith("case ") and cleaned.endswith("=>"):
            continue
        if cleaned.startswith("| ") and cleaned.endswith("=>"):
            continue
        tactics.append(cleaned)
        if len(tactics) >= k:
            break
    return tactics
