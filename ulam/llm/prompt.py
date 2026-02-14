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
        for candidate in _split_tactic_line(cleaned):
            if re.search(r":=\s*by\s*$", candidate):
                continue
            if candidate.endswith("=>"):
                continue
            if candidate in {"case", "cases", "calc"}:
                continue
            if candidate.startswith("case ") and candidate.endswith("=>"):
                continue
            if candidate.startswith("| ") and candidate.endswith("=>"):
                continue
            tactics.append(candidate)
            if len(tactics) >= k:
                break
        if len(tactics) >= k:
            break
    return tactics


def _split_tactic_line(text: str) -> list[str]:
    if "," not in text:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        if ch == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            candidate = "".join(buf).strip()
            if candidate:
                parts.append(candidate)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts
