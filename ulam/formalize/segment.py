from __future__ import annotations

import re
from typing import Iterable

from .types import FormalizationSegment


_ENV_KIND = {
    "definition": "definition",
    "lemma": "lemma",
    "theorem": "theorem",
    "proposition": "proposition",
    "corollary": "corollary",
    "example": "example",
    "proof": "proof",
}


def segment_tex(tex: str) -> list[FormalizationSegment]:
    segments: list[FormalizationSegment] = []
    if not tex.strip():
        return segments

    env_pattern = re.compile(
        r"\\begin\{(?P<env>definition|lemma|theorem|proposition|corollary|example|proof)\}(?P<body>.*?)\\end\{\1\}",
        re.DOTALL | re.IGNORECASE,
    )

    last_end = 0
    for match in env_pattern.finditer(tex):
        if match.start() > last_end:
            prefix = tex[last_end:match.start()].strip()
            if prefix:
                segments.append(FormalizationSegment("text", "", prefix))
        env = match.group("env").lower()
        body = match.group("body").strip()
        title = _extract_title(body)
        segments.append(FormalizationSegment(_ENV_KIND.get(env, env), title, body))
        last_end = match.end()

    tail = tex[last_end:].strip()
    if tail:
        segments.append(FormalizationSegment("text", "", tail))
    return segments


def attach_proofs(segments: list[FormalizationSegment]) -> list[FormalizationSegment]:
    if not segments:
        return segments
    merged: list[FormalizationSegment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.kind in {"lemma", "theorem", "proposition", "corollary", "example"}:
            body = seg.body
            if i + 1 < len(segments) and segments[i + 1].kind == "proof":
                proof_body = segments[i + 1].body.strip()
                if proof_body:
                    body = body.rstrip() + "\n\nProof.\n" + proof_body
                i += 1
            merged.append(
                FormalizationSegment(seg.kind, seg.title, body)
            )
        else:
            merged.append(seg)
        i += 1
    return merged


def _extract_title(body: str) -> str:
    title_match = re.search(r"\\label\{([^}]+)\}", body)
    if title_match:
        return title_match.group(1)
    name_match = re.search(r"\\textbf\{([^}]+)\}", body)
    if name_match:
        return name_match.group(1)
    return ""


def collect_segment_hints(segments: Iterable[FormalizationSegment], kind: str) -> list[str]:
    hints = []
    for seg in segments:
        if seg.kind != kind:
            continue
        chunk = seg.body.strip()
        if chunk:
            hints.append(chunk)
    return hints
