from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .engine import FormalizationEngine, _normalize_lean_output
from .segment import segment_tex, attach_proofs
from .types import FormalizationConfig


FORMAL_KINDS = {"definition", "lemma", "theorem", "proposition", "corollary", "example"}
SECTION_RE = re.compile(r"(\\section\\*?\\{[^}]*\\})", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"(\\subsection\\*?\\{[^}]*\\})", re.IGNORECASE)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def should_segment(text: str, threshold_words: int = 1000) -> bool:
    return count_words(text) > threshold_words


def run_segmented_formalize(
    config: FormalizationConfig,
    llm,
    artifact_dir: Path | None = None,
    max_words: int = 1000,
) -> Path:
    tex = config.tex_path.read_text(encoding="utf-8", errors="ignore")

    if artifact_dir is None:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path("runs") / f"formalize_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "segment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _build_manifest(config, tex, artifact_dir, max_words=max_words)
        _write_manifest(manifest_path, manifest)

    segments_dir = Path(manifest["segments_dir"])
    segments_dir.mkdir(parents=True, exist_ok=True)
    so_far_path = segments_dir / "combined_so_far.lean"

    for entry in manifest["segments"]:
        if entry.get("done") and Path(entry["lean_path"]).exists():
            continue

        seg_tex = Path(entry["tex_path"])
        seg_lean = Path(entry["lean_path"])
        context_files = list(config.context_files)
        if so_far_path.exists():
            context_files.append(so_far_path)

        seg_cfg = FormalizationConfig(
            tex_path=seg_tex,
            output_path=seg_lean,
            context_files=context_files,
            max_rounds=config.max_rounds,
            max_repairs=config.max_repairs,
            max_equivalence_repairs=config.max_equivalence_repairs,
            max_proof_rounds=config.max_proof_rounds,
            proof_max_steps=config.proof_max_steps,
            proof_beam=config.proof_beam,
            proof_k=config.proof_k,
            proof_timeout_s=config.proof_timeout_s,
            proof_repair=config.proof_repair,
            dojo_timeout_s=config.dojo_timeout_s,
            lemma_max=config.lemma_max,
            lemma_depth=config.lemma_depth,
            allow_axioms=config.allow_axioms,
            lean_project=config.lean_project,
            lean_imports=config.lean_imports,
            verbose=config.verbose,
            proof_backend=config.proof_backend,
            lean_backend=config.lean_backend,
            resume_path=seg_lean if seg_lean.exists() else None,
            artifact_dir=segments_dir / f"segment_{entry['index']:03d}",
            equivalence_checks=config.equivalence_checks,
            llm_check=config.llm_check,
            llm_check_timing=config.llm_check_timing,
            llm_check_repairs=config.llm_check_repairs,
        )
        engine = FormalizationEngine(seg_cfg, llm)
        result = engine.run()
        entry["done"] = True
        entry["typecheck_ok"] = result.typecheck_ok
        _write_manifest(manifest_path, manifest)
        _append_to_so_far(so_far_path, seg_lean)

    merged = _merge_segments([Path(entry["lean_path"]) for entry in manifest["segments"]])
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(merged, encoding="utf-8")
    return config.output_path


def _build_manifest(
    config: FormalizationConfig,
    tex: str,
    artifact_dir: Path,
    max_words: int = 1000,
) -> dict[str, Any]:
    segments_dir = artifact_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    pieces = _build_pieces(tex, max_words=max_words)

    entries: list[dict[str, Any]] = []
    for idx, piece in enumerate(pieces, start=1):
        tex_path = segments_dir / f"segment_{idx:03d}.tex"
        lean_path = segments_dir / f"segment_{idx:03d}.lean"
        tex_path.write_text(piece["tex"], encoding="utf-8")
        entries.append(
            {
                "index": idx,
                "kind": piece["kind"],
                "title": piece["title"],
                "tex_path": str(tex_path),
                "lean_path": str(lean_path),
                "done": False,
            }
        )

    return {
        "segmented": True,
        "segmentation": "section/subsection/lemma",
        "max_words": max_words,
        "tex_path": str(config.tex_path),
        "output_path": str(config.output_path),
        "segments_dir": str(segments_dir),
        "segments": entries,
        "context_files": [str(p) for p in config.context_files],
    }


def _build_pieces(tex: str, max_words: int = 1000) -> list[dict[str, str]]:
    sections = _split_by_heading(tex, SECTION_RE)
    if not sections:
        sections = [{"kind": "section", "title": "", "tex": tex}]

    pieces: list[dict[str, str]] = []
    for sec in sections:
        if count_words(sec["tex"]) <= max_words:
            pieces.append(sec)
            continue
        subsections = _split_by_heading(sec["tex"], SUBSECTION_RE)
        subsections = _merge_prefix_into_next(subsections)
        if len(subsections) > 1:
            for sub in subsections:
                if count_words(sub["tex"]) <= max_words:
                    pieces.append(sub)
                else:
                    pieces.extend(_split_by_lemmas(sub["tex"], max_words))
        else:
            pieces.extend(_split_by_lemmas(sec["tex"], max_words))
    return _merge_prefix_into_next(pieces)


def _split_by_heading(text: str, pattern: re.Pattern) -> list[dict[str, str]]:
    parts = pattern.split(text)
    if len(parts) <= 1:
        return []
    pieces: list[dict[str, str]] = []
    prefix = parts[0].strip()
    if prefix:
        pieces.append({"kind": "text", "title": "", "tex": prefix})
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (heading + "\n" + body).strip()
        title = _extract_heading_title(heading)
        pieces.append({"kind": "section", "title": title, "tex": chunk})
    return pieces


def _extract_heading_title(heading: str) -> str:
    m = re.search(r"\\(?:sub)?section\\*?\\{([^}]*)\\}", heading, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _split_by_lemmas(tex: str, max_words: int) -> list[dict[str, str]]:
    segs = attach_proofs(segment_tex(tex))
    chunks: list[dict[str, str]] = []
    current: list[str] = []
    for seg in segs:
        piece = seg.body.strip()
        if not piece:
            continue
        tentative = "\n\n".join(current + [piece]) if current else piece
        if current and count_words(tentative) > max_words:
            chunks.append({"kind": "chunk", "title": "", "tex": "\n\n".join(current)})
            current = [piece]
        else:
            current.append(piece)
    if current:
        chunks.append({"kind": "chunk", "title": "", "tex": "\n\n".join(current)})
    return chunks


def _merge_prefix_into_next(pieces: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(pieces) < 2:
        return pieces
    if pieces[0]["kind"] != "text":
        return pieces
    merged = pieces[1:]
    merged[0]["tex"] = pieces[0]["tex"].rstrip() + "\n\n" + merged[0]["tex"].lstrip()
    return merged


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _append_to_so_far(so_far_path: Path, lean_path: Path) -> None:
    if not lean_path.exists():
        return
    text = lean_path.read_text(encoding="utf-8", errors="ignore")
    body, imports = _split_imports(text)
    if not so_far_path.exists():
        so_far_path.write_text(_normalize_lean_output("\n".join(imports) + "\n\n" + body), encoding="utf-8")
        return
    current = so_far_path.read_text(encoding="utf-8", errors="ignore")
    current_body, current_imports = _split_imports(current)
    merged_imports = _merge_import_lines(current_imports + imports)
    combined = "\n".join(merged_imports) + "\n\n" + current_body.rstrip() + "\n\n" + body
    so_far_path.write_text(_normalize_lean_output(combined), encoding="utf-8")


def _merge_segments(paths: list[Path]) -> str:
    bodies: list[str] = []
    imports: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        body, imp = _split_imports(text)
        if body.strip():
            bodies.append(body.strip())
        imports.extend(imp)
    merged_imports = _merge_import_lines(imports)
    combined = "\n".join(merged_imports) + "\n\n" + "\n\n".join(bodies)
    return _normalize_lean_output(combined)


def _split_imports(text: str) -> tuple[str, list[str]]:
    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("import "):
            import_lines.append(line.strip())
        else:
            body_lines.append(line)
    return "\n".join(body_lines).strip(), import_lines


def _merge_import_lines(lines: list[str]) -> list[str]:
    modules: list[str] = []
    for line in lines:
        if not line.startswith("import "):
            continue
        remainder = line[len("import ") :].strip()
        if remainder:
            modules.extend(part.strip() for part in remainder.split() if part.strip())
    if not modules or "Mathlib" not in modules:
        modules.insert(0, "Mathlib")
    seen: set[str] = set()
    deduped = []
    for mod in modules:
        if mod in seen:
            continue
        seen.add(mod)
        deduped.append(mod)
    return ["import " + " ".join(deduped)]
