from __future__ import annotations

import hashlib
import re


def canonical_state_text(pretty: str) -> str:
    lines: list[str] = []
    for raw in pretty.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"0x[0-9a-fA-F]+", "0x#", line)
        # Lean goal states often include unstable generated identifiers.
        line = re.sub(r"\?m(?:_[0-9]+|[0-9]+)", "?m#", line)
        line = re.sub(r"\bmvarId!?[0-9]+\b", "mvarId#", line)
        line = re.sub(r"\bfvarId!?[0-9]+\b", "fvarId#", line)
        line = re.sub(r"\b(_uniq|_proof|_inst)\.[0-9]+\b", r"\1.#", line)
        lines.append(line)
    return "\n".join(lines)


def state_hash(pretty: str) -> str:
    canonical = canonical_state_text(pretty)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
