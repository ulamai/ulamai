from __future__ import annotations

import heapq
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMClient
from ..retrieve import Retriever
from ..state import state_hash
from ..trace import TraceLogger
from ..types import ProofState, ProofStep, RunConfig, TacticResult


@dataclass(order=True)
class _Node:
    score: int
    state: ProofState = field(compare=False)
    proof: list[str] = field(compare=False, default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    proof: list[str]
    steps: int
    error: Optional[str]


def best_first_search(
    runner,
    llm: LLMClient,
    retriever: Retriever,
    trace: TraceLogger,
    config: RunConfig,
    mode: str = "tactic",
) -> SearchResult:
    initial_state = runner.start(config.file_path, config.theorem)
    _log(config, f"[init] state={initial_state.key}")
    if config.instruction:
        _log(config, f"[instruction] {config.instruction}")
    frontier: list[_Node] = []
    initial_score = _node_score(0, initial_state)
    heapq.heappush(frontier, _Node(score=initial_score, state=initial_state, proof=[]))
    best_seen: dict[str, int] = {initial_state.key: initial_score}
    step_cache: dict[tuple[str, str], TacticResult] = {}
    steps = 0
    best_progress: list[str] = []

    while frontier and steps < config.max_steps:
        node = heapq.heappop(frontier)
        _log(config, f"[state] {node.state.key} {_summarize_state(node.state.pretty)}")
        retrieved = retriever.retrieve(node.state, k=config.retriever_k)
        if retrieved:
            _log(config, f"[retriever] {len(retrieved)} premises")
        _log(config, f"[llm] requesting {config.suggestions_per_state} suggestions")
        suggestions = llm.propose(
            node.state,
            retrieved,
            config.suggestions_per_state,
            instruction=config.instruction,
            context=config.context,
            mode=mode,
        )
        if config.autop:
            suggestions = _merge_suggestions(suggestions, _autop_tactics())
        if suggestions:
            _log(config, f"[llm] suggestions: {', '.join(suggestions)}")
        for tactic in suggestions:
            if steps >= config.max_steps:
                break
            cache_key = (node.state.key, tactic)
            cached = cache_key in step_cache
            if cached:
                result = step_cache[cache_key]
                elapsed_ms = 0
            else:
                start = time.perf_counter()
                result = runner.apply(node.state, tactic, config.timeout_s)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                step_cache[cache_key] = result
                steps += 1
            trace.log_step(
                _step_from_result(
                    node.state,
                    tactic,
                    result,
                    cached=cached,
                    elapsed_ms=elapsed_ms,
                )
            )
            _log(config, _format_result(tactic, result, cached))
            if result.ok and result.is_solved:
                final_proof = node.proof + [tactic]
                _emit_progress(config, final_proof)
                return SearchResult(True, final_proof, steps, None)
            if result.ok and result.new_state is not None:
                new_proof = node.proof + [tactic]
                if len(new_proof) > len(best_progress):
                    best_progress = new_proof
                    _emit_progress(config, new_proof)
                score = _node_score(len(new_proof), result.new_state)
                prev = best_seen.get(result.new_state.key)
                if prev is None or score < prev:
                    best_seen[result.new_state.key] = score
                    heapq.heappush(frontier, _Node(score=score, state=result.new_state, proof=new_proof))
                    _cap_frontier(frontier, config.beam_width)
                continue

            if not result.ok and config.repair_attempts > 0:
                if steps >= config.max_steps:
                    break
                _log(config, f"[repair] tactic failed: {tactic}")
                repaired = _attempt_repair(
                    runner=runner,
                    llm=llm,
                    retriever=retriever,
                    trace=trace,
                    step_cache=step_cache,
                    state=node.state,
                    failed_tactic=tactic,
                    error=result.error or "unknown error",
                    config=config,
                    mode=mode,
                    step_budget=max(0, config.max_steps - steps),
                )
                steps += repaired.steps_used
                if repaired.solved:
                    final_proof = node.proof + [repaired.tactic]
                    _emit_progress(config, final_proof)
                    return SearchResult(True, final_proof, steps, None)
                if repaired.new_state is not None:
                    new_proof = node.proof + [repaired.tactic]
                    if len(new_proof) > len(best_progress):
                        best_progress = new_proof
                        _emit_progress(config, new_proof)
                    score = _node_score(len(new_proof), repaired.new_state)
                    prev = best_seen.get(repaired.new_state.key)
                    if prev is None or score < prev:
                        best_seen[repaired.new_state.key] = score
                        heapq.heappush(frontier, _Node(score=score, state=repaired.new_state, proof=new_proof))
                        _cap_frontier(frontier, config.beam_width)

    return SearchResult(False, [], steps, "Search exhausted or max steps reached")


@dataclass(frozen=True)
class _RepairResult:
    solved: bool
    tactic: str
    new_state: Optional[ProofState]
    steps_used: int = 0


def _attempt_repair(
    runner,
    llm: LLMClient,
    retriever: Retriever,
    trace: TraceLogger,
    step_cache: dict[tuple[str, str], TacticResult],
    state: ProofState,
    failed_tactic: str,
    error: str,
    config: RunConfig,
    mode: str = "tactic",
    step_budget: int = 0,
) -> _RepairResult:
    retrieved = retriever.retrieve(state, k=config.retriever_k)
    suggestions = llm.repair(
        state,
        retrieved,
        failed_tactic,
        error,
        config.repair_attempts,
        instruction=config.instruction,
        context=config.context,
        mode=mode,
    )
    steps_used = 0
    for tactic in suggestions:
        cache_key = (state.key, tactic)
        cached = cache_key in step_cache
        if cached:
            result = step_cache[cache_key]
            elapsed_ms = 0
        else:
            if steps_used >= step_budget:
                continue
            start = time.perf_counter()
            result = runner.apply(state, tactic, config.timeout_s)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            step_cache[cache_key] = result
            steps_used += 1
        trace.log_step(
            _step_from_result(
                state,
                tactic,
                result,
                cached=cached,
                elapsed_ms=elapsed_ms,
            )
        )
        _log(config, _format_result(tactic, result, cached))
        if result.ok and result.is_solved:
            return _RepairResult(True, tactic, None, steps_used=steps_used)
        if result.ok and result.new_state is not None:
            return _RepairResult(False, tactic, result.new_state, steps_used=steps_used)
    return _RepairResult(False, failed_tactic, None, steps_used=steps_used)


def scripted_search(
    runner,
    llm: LLMClient,
    retriever: Retriever,
    trace: TraceLogger,
    config: RunConfig,
) -> SearchResult:
    state = runner.start(config.file_path, config.theorem)
    _log(config, f"[init] state={state.key}")
    if config.instruction:
        _log(config, f"[instruction] {config.instruction}")

    proof: list[str] = []
    steps = 0
    last_error: str | None = None
    last_tactic: str | None = None
    repair_budget = config.repair_attempts

    while steps < config.max_steps:
        _log(config, f"[state] {state.key} {_summarize_state(state.pretty)}")
        retrieved = retriever.retrieve(state, k=config.retriever_k)
        if retrieved:
            _log(config, f"[retriever] {len(retrieved)} premises")
        if last_error and repair_budget > 0:
            _log(config, "[repair] requesting script")
            suggestions = llm.repair(
                state,
                retrieved,
                last_tactic or "",
                last_error,
                config.suggestions_per_state,
                instruction=config.instruction,
                context=config.context,
                mode="script",
            )
            repair_budget -= 1
        else:
            _log(config, f"[llm] requesting script ({config.suggestions_per_state} lines)")
            suggestions = llm.propose(
                state,
                retrieved,
                config.suggestions_per_state,
                instruction=config.instruction,
                context=config.context,
                mode="script",
            )
            repair_budget = config.repair_attempts

        if suggestions:
            _log(config, f"[llm] suggestions: {', '.join(suggestions)}")
        else:
            return SearchResult(False, proof, steps, "LLM returned no tactics")

        progressed = False
        for tactic in suggestions:
            if steps >= config.max_steps:
                break
            start = time.perf_counter()
            result = runner.apply(state, tactic, config.timeout_s)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            steps += 1
            trace.log_step(
                _step_from_result(
                    state,
                    tactic,
                    result,
                    cached=False,
                    elapsed_ms=elapsed_ms,
                )
            )
            _log(config, _format_result(tactic, result, cached=False))
            if result.ok and result.is_solved:
                proof.append(tactic)
                _emit_progress(config, proof)
                return SearchResult(True, proof, steps, None)
            if result.ok and result.new_state is not None:
                proof.append(tactic)
                _emit_progress(config, proof)
                state = result.new_state
                last_error = None
                last_tactic = None
                progressed = True
                continue

            last_error = result.error or "unknown error"
            last_tactic = tactic
            break

        if not progressed and config.autop:
            _log(config, "[autop] trying fallback tactics")
            for tactic in _autop_tactics():
                if steps >= config.max_steps:
                    break
                start = time.perf_counter()
                result = runner.apply(state, tactic, config.timeout_s)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                steps += 1
                trace.log_step(
                    _step_from_result(
                        state,
                        tactic,
                        result,
                        cached=False,
                        elapsed_ms=elapsed_ms,
                    )
                )
                _log(config, _format_result(tactic, result, cached=False))
                if result.ok and result.is_solved:
                    proof.append(tactic)
                    _emit_progress(config, proof)
                    return SearchResult(True, proof, steps, None)
                if result.ok and result.new_state is not None:
                    proof.append(tactic)
                    _emit_progress(config, proof)
                    state = result.new_state
                    last_error = None
                    last_tactic = None
                    progressed = True
                    break
                last_error = result.error or "unknown error"
                last_tactic = tactic

        if not progressed and last_error and repair_budget <= 0:
            return SearchResult(False, proof, steps, last_error)

    return SearchResult(False, proof, steps, "Search exhausted or max steps reached")


def _step_from_result(
    state: ProofState,
    tactic: str,
    result: TacticResult,
    cached: bool = False,
    elapsed_ms: int | None = None,
) -> ProofStep:
    return ProofStep(
        state_key=state.key,
        state_pretty=state.pretty,
        state_hash=state_hash(state.pretty),
        tactic=tactic,
        ok=result.ok,
        error=result.error,
        new_state_key=result.new_state.key if result.new_state else None,
        new_state_hash=state_hash(result.new_state.pretty) if result.new_state else None,
        solved=result.is_solved,
        cached=cached,
        elapsed_ms=elapsed_ms,
        error_kind=_error_kind(result.error),
    )


def _cap_frontier(frontier: list[_Node], beam_width: int) -> None:
    if beam_width <= 0:
        return
    if len(frontier) <= beam_width:
        return
    trimmed = heapq.nsmallest(beam_width, frontier)
    frontier.clear()
    frontier.extend(trimmed)
    heapq.heapify(frontier)


def _log(config: RunConfig, message: str) -> None:
    if config.verbose:
        print(message)


def _emit_progress(config: RunConfig, proof: list[str]) -> None:
    callback = config.on_progress
    if callback is None:
        return
    try:
        callback(list(proof))
    except Exception:
        return


def _summarize_state(pretty: str, max_chars: int = 200) -> str:
    text = " ".join(line.strip() for line in pretty.splitlines() if line.strip())
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _format_result(tactic: str, result: TacticResult, cached: bool) -> str:
    prefix = "[cached]" if cached else "[exec]"
    if result.ok and result.is_solved:
        return f"{prefix} {tactic} -> solved"
    if result.ok and result.new_state is not None:
        return f"{prefix} {tactic} -> ok (state={result.new_state.key})"
    err = result.error or "unknown error"
    if len(err) > 160:
        err = err[:160] + "..."
    return f"{prefix} {tactic} -> error: {err}"


def _autop_tactics() -> list[str]:
    return [
        "aesop",
        "simp",
        "ring_nf",
        "linarith",
        "nlinarith",
    ]


def _merge_suggestions(suggestions: list[str], extras: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in suggestions + extras:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _node_score(proof_len: int, state: ProofState) -> int:
    # Prefer shorter proofs, then states that look simpler.
    return proof_len * 1000 + _state_complexity(state.pretty)


def _state_complexity(pretty: str) -> int:
    lines = [line.strip() for line in pretty.splitlines() if line.strip()]
    if not lines:
        return 0
    goals = sum(1 for line in lines if "⊢" in line)
    hyps = sum(1 for line in lines if ":" in line and "⊢" not in line)
    text = " ".join(lines)
    char_penalty = min(400, len(text) // 8)
    return goals * 120 + hyps * 8 + char_penalty


def _error_kind(error: str | None) -> str | None:
    if not error:
        return None
    text = error.lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "unknown identifier" in text:
        return "unknown_identifier"
    if "type mismatch" in text:
        return "type_mismatch"
    if "rewrite failed" in text or "did not find an occurrence" in text:
        return "rewrite_failed"
    if "unsolved goals" in text:
        return "unsolved_goals"
    if re.search(r"unexpected token|parse error", text):
        return "parse_error"
    return "other"
