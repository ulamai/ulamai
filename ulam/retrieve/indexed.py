from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_DECL_RE = re.compile(
    r"^\s*(theorem|lemma|example|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b",
    re.M,
)

_SKIP_PARTS = {
    ".git",
    ".lake",
    ".elan",
    "__pycache__",
    "build",
    "dist",
    "target",
}


@dataclass(frozen=True)
class PremiseRecord:
    name: str
    statement: str
    premise: str
    kind: str
    path: str
    line: int
    scope: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "statement": self.statement,
                "premise": self.premise,
                "kind": self.kind,
                "path": self.path,
                "line": self.line,
                "scope": self.scope,
            },
            ensure_ascii=True,
        )


def build_premise_index(project_path: Path, output_path: Path, scope: str = "local") -> dict[str, int]:
    project = project_path.expanduser().resolve()
    output = output_path.expanduser()
    if not output.is_absolute():
        output = project / output
    output.parent.mkdir(parents=True, exist_ok=True)

    if scope not in {"local", "mathlib", "both"}:
        raise RuntimeError(f"unknown scope: {scope}")

    local_files = list(_iter_local_lean_files(project)) if scope in {"local", "both"} else []
    mathlib_files = list(_iter_mathlib_lean_files(project)) if scope in {"mathlib", "both"} else []

    records: list[PremiseRecord] = []
    for path in local_files:
        records.extend(_extract_records(path, project, "local"))
    for path in mathlib_files:
        records.extend(_extract_records(path, project, "mathlib"))

    unique: dict[str, PremiseRecord] = {}
    for record in records:
        key = f"{record.name}|{record.statement}|{record.path}|{record.line}"
        unique[key] = record
    ordered = sorted(unique.values(), key=lambda item: (item.scope, item.path, item.line))

    with output.open("w", encoding="utf-8") as fh:
        for record in ordered:
            fh.write(record.to_json() + "\n")

    stats = {
        "records": len(ordered),
        "local_files": len(local_files),
        "mathlib_files": len(mathlib_files),
    }
    return stats


def load_index_premises(index_path: Path) -> list[str]:
    records = _read_index(index_path)
    premises: list[str] = []
    seen: set[str] = set()
    for row in records:
        premise = str(row.get("premise", "")).strip()
        if not premise or premise in seen:
            continue
        seen.add(premise)
        premises.append(premise)
    return premises


def load_index_stats(index_path: Path) -> dict[str, int]:
    records = _read_index(index_path)
    local = 0
    mathlib = 0
    for row in records:
        scope = str(row.get("scope", "")).strip().lower()
        if scope == "local":
            local += 1
        elif scope == "mathlib":
            mathlib += 1
    return {
        "records": len(records),
        "local_records": local,
        "mathlib_records": mathlib,
    }


def _read_index(index_path: Path) -> list[dict]:
    path = index_path.expanduser()
    if not path.exists():
        raise RuntimeError(f"Index file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _iter_local_lean_files(project: Path) -> Iterable[Path]:
    for path in project.rglob("*.lean"):
        if _skip_path(path, project):
            continue
        yield path


def _iter_mathlib_lean_files(project: Path) -> Iterable[Path]:
    root = project / ".lake" / "packages" / "mathlib" / "Mathlib"
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*.lean"):
        if path.is_file():
            files.append(path)
    return files


def _skip_path(path: Path, project: Path) -> bool:
    try:
        rel = path.relative_to(project)
    except Exception:
        rel = path
    parts = {part for part in rel.parts}
    if parts.intersection(_SKIP_PARTS):
        return True
    return False


def _extract_records(path: Path, project: Path, scope: str) -> list[PremiseRecord]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    matches = list(_DECL_RE.finditer(text))
    if not matches:
        return []

    rel = _rel_path(path, project)
    out: list[PremiseRecord] = []
    for idx, match in enumerate(matches):
        kind = match.group(1)
        name = match.group(2)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        header = _header_text(block)
        statement = _extract_statement(header, kind, name)
        premise = f"{name}: {statement}" if statement else name
        line = text.count("\n", 0, start) + 1
        out.append(
            PremiseRecord(
                name=name,
                statement=statement,
                premise=premise,
                kind=kind,
                path=rel,
                line=line,
                scope=scope,
            )
        )
    return out


def _rel_path(path: Path, project: Path) -> str:
    try:
        return str(path.relative_to(project))
    except Exception:
        return str(path)


def _header_text(block: str) -> str:
    for marker in (":= by", ":=", "where"):
        pos = block.find(marker)
        if pos >= 0:
            return block[:pos]
    return block


def _extract_statement(header: str, kind: str, name: str) -> str:
    flat = " ".join(header.split())
    prefix = f"{kind} {name}"
    if flat.startswith(prefix):
        flat = flat[len(prefix) :].strip()
    colon = _find_top_level_colon(flat)
    if colon >= 0:
        stmt = flat[colon + 1 :].strip()
    else:
        stmt = flat.strip()
    return stmt


def _find_top_level_colon(text: str) -> int:
    paren = 0
    brace = 0
    bracket = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == ":" and idx + 1 < len(text) and text[idx + 1] == "=":
            continue
        elif ch == ":" and paren == 0 and brace == 0 and bracket == 0:
            return idx
    return -1
