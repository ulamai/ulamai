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
        "- Preserve existing proofs where possible.\n"
        "- If the proof no longer matches, replace with `by sorry`.\n"
        "- Do not change unrelated declarations.\n\n"
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


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
