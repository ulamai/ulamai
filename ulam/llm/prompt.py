from __future__ import annotations

from typing import Iterable

from ..types import ProofState


def build_prompt(
    state: ProofState,
    retrieved: Iterable[str],
    k: int,
    instruction: str | None = None,
    context: Iterable[str] | None = None,
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
    user += (
        "Suggest the next tactic or small lemma.\n"
        f"Return exactly {k} line(s), one tactic per line."
    )
    return system, user


def parse_tactics(text: str, k: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tactics = []
    for line in lines:
        if line.startswith("```"):
            continue
        tactics.append(line)
        if len(tactics) >= k:
            break
    return tactics
