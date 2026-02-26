from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable


class FormalizationLLM:
    def __init__(self, provider: str, config: dict) -> None:
        self._provider = provider
        self._config = config

    def statement(self, text: str, context: str) -> str:
        prompt = _build_statement_prompt(text, context)
        return self._call(prompt)

    def draft(self, tex: str, context: str, hints: list[str]) -> str:
        prompt = _build_draft_prompt(tex, context, hints)
        return self._call(prompt)

    def repair(self, lean_code: str, error: str, context: str) -> str:
        prompt = _build_repair_prompt(lean_code, error, context)
        return self._call(prompt)

    def improve(self, lean_code: str, failures: list[str], context: str) -> str:
        prompt = _build_improve_prompt(lean_code, failures, context)
        return self._call(prompt)

    def prove(
        self,
        lean_code: str,
        name: str,
        instruction: str,
        tex_snippet: str,
        context: str,
        error: str | None = None,
    ) -> str:
        prompt = _build_proof_prompt(lean_code, name, instruction, tex_snippet, context, error)
        return self._call(prompt)

    def equivalence_check(self, tex_statement: str, lean_statement: str) -> dict:
        prompt = _build_equivalence_prompt(tex_statement, lean_statement)
        raw = self._call(prompt)
        return _parse_equivalence(raw)

    def repair_statement(self, lean_code: str, name: str, tex_statement: str, context: str) -> str:
        prompt = _build_statement_repair_prompt(lean_code, name, tex_statement, context)
        return self._call(prompt)

    def plan_lemmas(
        self,
        theorem_name: str,
        theorem_statement: str,
        original_statement: str,
        context: str,
    ) -> str:
        prompt = _build_lemma_plan_prompt(
            theorem_name, theorem_statement, original_statement, context
        )
        return self._call(prompt)

    def expand_lemmas(
        self,
        lemma_name: str,
        lemma_statement: str,
        last_goal: str,
        failures: list[str],
        successes: list[str],
        context: str,
    ) -> str:
        prompt = _build_lemma_expand_prompt(
            lemma_name, lemma_statement, last_goal, failures, successes, context
        )
        return self._call(prompt)

    def summarize_attempt(
        self,
        theorem_name: str,
        theorem_statement: str,
        last_goal: str,
        failures: list[str],
        successes: list[str],
        context: str,
    ) -> str:
        prompt = _build_summary_prompt(
            theorem_name, theorem_statement, last_goal, failures, successes, context
        )
        return self._call(prompt)

    def semantic_check(
        self,
        tex: str,
        lean_code: str,
        deterministic_issues: list[dict],
        context: str,
        stage: str = "end",
    ) -> dict:
        prompt = _build_semantic_check_prompt(
            tex, lean_code, deterministic_issues, context, stage=stage
        )
        raw = self._call(prompt)
        return _parse_semantic_check(raw)

    def semantic_repair(
        self,
        lean_code: str,
        tex: str,
        deterministic_issues: list[dict],
        audit: dict,
        context: str,
    ) -> str:
        prompt = _build_semantic_repair_prompt(
            lean_code,
            tex,
            deterministic_issues,
            audit,
            context,
        )
        return self._call(prompt)

    def _call(self, prompt: str) -> str:
        if self._provider == "openai":
            return _call_openai(self._config, prompt)
        if self._provider == "ollama":
            return _call_ollama(self._config, prompt)
        if self._provider == "anthropic":
            return _call_anthropic(self._config, prompt)
        if self._provider == "codex_cli":
            return _call_codex_cli(self._config, prompt)
        if self._provider == "claude_cli":
            return _call_claude_cli(self._config, prompt)
        if self._provider == "gemini":
            return _call_gemini(self._config, prompt)
        if self._provider == "gemini_cli":
            return _call_gemini_cli(self._config, prompt)
        return ""


def _build_draft_prompt(tex: str, context: str, hints: list[str]) -> str:
    hint_block = "\n\n".join(hints)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Task: formalize the following LaTeX into Lean 4.\n"
        "- Add missing definitions.\n"
        "- Use `sorry` for incomplete proofs.\n"
        "- Keep theorem names stable and human-readable.\n\n"
        "LaTeX:\n"
        f"{tex}\n\n"
    )
    if hint_block:
        prompt += "Hints from theorem/proof text:\n" + hint_block + "\n\n"
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code."
    return prompt


def _build_statement_prompt(text: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant.\n"
        "Convert the following informal statement into a Lean 4 proposition.\n"
        "- Return ONLY the Lean proposition (type), without `theorem`/`lemma` keywords.\n"
        "- Do not include any proofs or code fences.\n"
        "- If the statement is already Lean, return it unchanged.\n\n"
        "Statement:\n"
        f"{text}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY the Lean proposition."
    return prompt


def _build_repair_prompt(lean_code: str, error: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "The following Lean file fails to typecheck. Fix it.\n"
        "- Preserve existing structure where possible.\n"
        "- If needed, add imports or small helper lemmas.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
        "Lean error:\n"
        f"{error}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY corrected Lean code."
    return prompt


def _build_improve_prompt(lean_code: str, failures: list[str], context: str) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Some theorems remain unsolved. Improve the Lean file by:\n"
        "- Adding missing intermediate lemmas.\n"
        "- Tweaking statements to match intended informal meaning.\n"
        "- Preserving already solved proofs.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
        "Unsolved items:\n"
        f"{failure_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY updated Lean code."
    return prompt


def _build_proof_prompt(
    lean_code: str,
    name: str,
    instruction: str,
    tex_snippet: str,
    context: str,
    error: str | None,
) -> str:
    prompt = (
        "You are a Lean 4 assistant. Output Lean code only.\n"
        f"Task: complete the proof of `{name}` in the Lean file below.\n"
        "- Preserve other declarations and imports.\n"
        "- You may add helper lemmas before the target declaration if needed.\n"
        "- Do NOT use `sorry` or `admit` in the proof of the target.\n\n"
        "Lean file:\n"
        f"{lean_code}\n\n"
    )
    if instruction:
        prompt += "User guidance:\n" + instruction.strip() + "\n\n"
    if tex_snippet:
        snippet = tex_snippet.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600] + "\n-- (truncated)"
        prompt += f"Informal proof snippet for {name}:\n{snippet}\n\n"
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    if error:
        trimmed = error.strip()
        if len(trimmed) > 2000:
            trimmed = trimmed[:2000] + "\n-- (truncated)"
        prompt += "Lean error:\n" + trimmed + "\n\n"
    prompt += "Return ONLY the updated Lean file."
    return prompt


def _build_equivalence_prompt(tex_statement: str, lean_statement: str) -> str:
    prompt = (
        "You compare informal math statements to Lean statements.\n"
        "Return a JSON object with fields: match (yes|no|unknown), reason.\n"
        "Only return JSON.\n\n"
        "Informal statement:\n"
        f"{tex_statement}\n\n"
        "Lean statement:\n"
        f"{lean_statement}\n"
    )
    return prompt


def _build_statement_repair_prompt(lean_code: str, name: str, tex_statement: str, context: str) -> str:
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Update the statement of the declaration below to match the informal statement.\n"
        "- Edit ONLY the named declaration.\n"
        "- Do NOT change any unrelated declaration (statement or proof).\n"
        "- If the target proof no longer matches, `by sorry` is allowed only for the target declaration.\n"
        "- Keep theorem/lemma names unchanged.\n\n"
        f"Declaration name: {name}\n\n"
        "Informal statement:\n"
        f"{tex_statement}\n\n"
        "Current Lean file:\n"
        f"{lean_code}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY the updated Lean file."
    return prompt


def _build_lemma_plan_prompt(
    theorem_name: str,
    theorem_statement: str,
    original_statement: str,
    context: str,
) -> str:
    prompt = (
        "You are a Lean 4 assistant. Produce a lemma-first proof plan as Lean code.\n"
        "- Output Lean code only.\n"
        "- Introduce helper definitions/lemmas if needed.\n"
        "- Use `by sorry` for proofs.\n"
        f"- The final theorem MUST be named `{theorem_name}`.\n"
        f"- The final theorem statement MUST be: {theorem_statement}\n\n"
        "Original informal statement:\n"
        f"{original_statement}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code."
    return prompt


def _build_lemma_expand_prompt(
    lemma_name: str,
    lemma_statement: str,
    last_goal: str,
    failures: list[str],
    successes: list[str],
    context: str,
) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures[:12])
    success_block = "\n".join(f"- {item}" for item in successes[:12])
    prompt = (
        "You are a Lean 4 assistant. Suggest new helper lemmas/defs to prove a stuck lemma.\n"
        "- Output Lean code only.\n"
        "- Do NOT restate the target lemma.\n"
        "- Use `by sorry` for proofs.\n\n"
        f"Stuck lemma name: {lemma_name}\n"
        f"Stuck lemma statement:\n{lemma_statement}\n\n"
        f"Last goal:\n{last_goal}\n\n"
        "Recent failed tactics/errors:\n"
        f"{failure_block}\n\n"
        "Recent successful tactics:\n"
        f"{success_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return ONLY Lean code for new helper declarations."
    return prompt


def _build_summary_prompt(
    theorem_name: str,
    theorem_statement: str,
    last_goal: str,
    failures: list[str],
    successes: list[str],
    context: str,
) -> str:
    failure_block = "\n".join(f"- {item}" for item in failures[:12])
    success_block = "\n".join(f"- {item}" for item in successes[:12])
    prompt = (
        "You are a Lean 4 assistant. Summarize a failed proof attempt and propose next steps.\n"
        "- Keep it short and actionable.\n"
        "- Prefer Lean tactics/lemmas likely to work.\n"
        "- Do not output code fences.\n\n"
        f"Theorem name: {theorem_name}\n"
        f"Theorem statement:\n{theorem_statement}\n\n"
        f"Last goal:\n{last_goal}\n\n"
        "Recent failed tactics/errors:\n"
        f"{failure_block}\n\n"
        "Recent successful tactics:\n"
        f"{success_block}\n\n"
    )
    if context:
        prompt += "Context files:\n" + context + "\n\n"
    prompt += "Return a brief summary and 3-7 next-step suggestions."
    return prompt


def _truncate_block(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n-- (truncated)"


def _build_semantic_check_prompt(
    tex: str,
    lean_code: str,
    deterministic_issues: list[dict],
    context: str,
    stage: str = "end",
) -> str:
    issues = deterministic_issues[:32]
    issue_json = json.dumps(issues, ensure_ascii=True, indent=2)
    prompt = (
        "You are auditing semantic integrity of a Lean formalization.\n"
        "Goal: detect whether the Lean file preserves the intended theorem(s) from LaTeX,\n"
        "or if it was made trivially true by weakening definitions/statements.\n"
        "Focus on cheats such as:\n"
        "- introducing `axiom`/`constant` (including `private axiom` / `private constant`) to bypass proofs,\n"
        "- redefining hard concepts to `True`/`False` or constants,\n"
        "- replacing counting/probability objects by trivial identities/constants,\n"
        "- changing theorem statements away from intended meaning.\n\n"
        f"Audit stage: {stage}\n\n"
        "Deterministic findings (pre-check):\n"
        f"{issue_json}\n\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        '  "verdict": "pass|fail|unknown",\n'
        '  "summary": "short text",\n'
        '  "issues": [{"kind":"...", "severity":"low|medium|high", "evidence":"..."}],\n'
        '  "should_repair": true|false\n'
        "}\n\n"
        "LaTeX source:\n"
        f"{_truncate_block(tex, 18000)}\n\n"
        "Lean file:\n"
        f"{_truncate_block(lean_code, 26000)}\n\n"
    )
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 6000) + "\n\n"
    return prompt


def _build_semantic_repair_prompt(
    lean_code: str,
    tex: str,
    deterministic_issues: list[dict],
    audit: dict,
    context: str,
) -> str:
    audit_json = json.dumps(audit, ensure_ascii=True, indent=2)
    deterministic_json = json.dumps(deterministic_issues[:32], ensure_ascii=True, indent=2)
    prompt = (
        "You are a Lean 4 formalization assistant. Output Lean code only.\n"
        "Repair semantic integrity issues in this Lean file.\n"
        "Constraints:\n"
        "- Preserve intended meaning of the original LaTeX theorem(s).\n"
        "- Do NOT trivialize definitions (e.g., Prop := True/False, constant stubs).\n"
        "- Keep theorem names stable.\n"
        "- Avoid introducing `axiom` or `constant` unless mathematically unavoidable.\n"
        "- Keep already-correct parts when possible.\n\n"
        "Semantic audit result:\n"
        f"{audit_json}\n\n"
        "Deterministic findings:\n"
        f"{deterministic_json}\n\n"
        "LaTeX source:\n"
        f"{_truncate_block(tex, 14000)}\n\n"
        "Current Lean file:\n"
        f"{_truncate_block(lean_code, 26000)}\n\n"
    )
    if context:
        prompt += "Context files:\n" + _truncate_block(context, 6000) + "\n\n"
    prompt += "Return ONLY the corrected Lean file."
    return prompt


def _call_openai(config: dict, prompt: str) -> str:
    openai = config.get("openai", {})
    api_key = openai.get("api_key", "") or os.environ.get("ULAM_OPENAI_API_KEY", "")
    if not api_key:
        return ""
    base_url = openai.get("base_url", "https://api.openai.com").rstrip("/")
    model = openai.get("model", "gpt-4.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        raw = resp.read().decode("utf-8")
    return _extract_openai(raw)


def _call_ollama(config: dict, prompt: str) -> str:
    ollama = config.get("ollama", {})
    base_url = ollama.get("base_url", "http://localhost:11434").rstrip("/")
    model = ollama.get("model", "llama3.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    endpoints: list[str] = []
    if base_url.endswith("/api"):
        base_url = base_url[: -len("/api")]
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
        endpoints.append(f"{base_url}/v1/chat/completions")
    endpoints.append(f"{base_url}/api/chat")
    endpoints.append(f"{base_url}/v1/chat/completions")
    seen = set()
    endpoints = [url for url in endpoints if not (url in seen or seen.add(url))]
    last_error: Exception | None = None
    for url in endpoints:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                raw = resp.read().decode("utf-8")
            return _extract_ollama(raw)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (404, 405):
                continue
            raise
    if last_error:
        raise last_error
    return ""


def _call_anthropic(config: dict, prompt: str) -> str:
    anthropic = config.get("anthropic", {})
    api_key = anthropic.get("api_key", "") or anthropic.get("setup_token", "")
    api_key = api_key or os.environ.get("ULAM_ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    base_url = anthropic.get("base_url", "https://api.anthropic.com").rstrip("/")
    model = anthropic.get("model", "claude-3-5-sonnet-20240620")
    payload = {
        "model": model,
        "max_tokens": 3000,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        raw = resp.read().decode("utf-8")
    return _extract_anthropic(raw)


def _call_gemini(config: dict, prompt: str) -> str:
    gemini = config.get("gemini", {})
    api_key = (
        gemini.get("api_key", "")
        or os.environ.get("ULAM_GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    if not api_key:
        return ""
    base_url = gemini.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    model = gemini.get("model", "gemini-3.1-pro-preview")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Lean 4 formalization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _gemini_chat_endpoint(base_url),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        raw = resp.read().decode("utf-8")
    return _extract_openai(raw)


def _extract_openai(raw: str) -> str:
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0]
    if "message" in msg and "content" in msg["message"]:
        return msg["message"]["content"]
    if "text" in msg:
        return msg["text"]
    return ""


def _extract_ollama(raw: str) -> str:
    data = json.loads(raw)
    message = data.get("message")
    if isinstance(message, dict) and "content" in message:
        return message["content"]
    choices = data.get("choices") or []
    if choices:
        choice = choices[0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    if "response" in data and isinstance(data["response"], str):
        return data["response"]
    return ""


def _extract_anthropic(raw: str) -> str:
    data = json.loads(raw)
    content = data.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return data.get("text", "") if isinstance(data.get("text", ""), str) else ""


def _call_codex_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import codex_exec

    openai = config.get("openai", {})
    llm_cfg = config.get("llm", {})
    model = openai.get("codex_model") or openai.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return codex_exec(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _call_claude_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import claude_print

    anthropic = config.get("anthropic", {})
    llm_cfg = config.get("llm", {})
    model = anthropic.get("claude_model") or anthropic.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return claude_print(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _call_gemini_cli(config: dict, prompt: str) -> str:
    from ..llm.cli_utils import gemini_exec

    gemini = config.get("gemini", {})
    llm_cfg = config.get("llm", {})
    model = gemini.get("cli_model") or gemini.get("model") or None
    system = "You are a Lean 4 formalization assistant. Output Lean code only."
    timeout_s = float(llm_cfg.get("timeout_s", 0))
    heartbeat_s = float(llm_cfg.get("heartbeat_s", 60))
    return gemini_exec(
        system,
        prompt,
        model=model,
        timeout_s=timeout_s,
        heartbeat_s=heartbeat_s,
    )


def _gemini_chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/openai") or base.endswith("/openai/v1") or base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _parse_equivalence(raw: str) -> dict:
    if not raw.strip():
        return {"match": "unknown", "reason": "empty response"}
    text = raw.strip()
    try:
        return json.loads(_extract_json(text))
    except Exception:
        match = "unknown"
        reason = "unparsed response"
        lowered = text.lower()
        if "match" in lowered and "yes" in lowered:
            match = "yes"
        elif "match" in lowered and "no" in lowered:
            match = "no"
        return {"match": match, "reason": reason, "raw": text[:400]}


def _parse_semantic_check(raw: str) -> dict:
    if not raw.strip():
        return {
            "verdict": "unknown",
            "summary": "empty response",
            "issues": [],
            "should_repair": True,
        }
    text = raw.strip()
    try:
        payload = json.loads(_extract_json(text))
    except Exception:
        lowered = text.lower()
        verdict = "unknown"
        if "verdict" in lowered and "pass" in lowered:
            verdict = "pass"
        elif "verdict" in lowered and "fail" in lowered:
            verdict = "fail"
        return {
            "verdict": verdict,
            "summary": "unparsed response",
            "issues": [],
            "should_repair": verdict != "pass",
            "raw": text[:600],
        }
    if not isinstance(payload, dict):
        return {
            "verdict": "unknown",
            "summary": "invalid payload",
            "issues": [],
            "should_repair": True,
        }
    verdict = str(payload.get("verdict", "unknown")).strip().lower()
    if verdict not in {"pass", "fail", "unknown"}:
        verdict = "unknown"
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    summary = str(payload.get("summary", "")).strip()
    should_repair = payload.get("should_repair", verdict != "pass")
    if isinstance(should_repair, str):
        should_repair = should_repair.strip().lower() in {"1", "true", "yes", "y"}
    else:
        should_repair = bool(should_repair)
    return {
        "verdict": verdict,
        "summary": summary or ("ok" if verdict == "pass" else "issues found"),
        "issues": issues[:32],
        "should_repair": should_repair,
    }


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
