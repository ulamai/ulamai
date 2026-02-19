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
        lines.append(line)
    return "\n".join(lines)


def state_hash(pretty: str) -> str:
    canonical = canonical_state_text(pretty)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
