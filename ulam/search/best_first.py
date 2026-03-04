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
    stats: dict[str, int] = field(default_factory=dict)


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
    best_seen_hash: dict[str, int] = {state_hash(initial_state.pretty): initial_score}
    step_cache: dict[tuple[str, str], TacticResult] = {}
    tactic_head_attempts: dict[tuple[str, str], int] = {}
    repair_error_attempts: dict[tuple[str, str], int] = {}
    plan_cache: dict[str, list[str]] = {}
    stagnant_expansions = 0
    planner_stats = _init_planner_stats()
    steps = 0
    best_progress: list[str] = []
    search_instruction = _search_instruction(config.instruction, config.theorem)

    while frontier and steps < config.max_steps:
        node = heapq.heappop(frontier)
        made_progress = False
        _log(config, f"[state] {node.state.key} {_summarize_state(node.state.pretty)}")
        round_instruction = search_instruction
        if stagnant_expansions >= _STAGNATION_REPLAN_THRESHOLD:
            _log(config, "[planner] stagnation detected; requesting replan-focused suggestions")
            planner_stats["planner_replan_triggers"] += 1
            round_instruction = _with_replan_hint(
                base_instruction=search_instruction,
                best_progress=best_progress,
                current_state=node.state.pretty,
            )
        retrieved = retriever.retrieve(node.state, k=config.retriever_k)
        if retrieved:
            _log(config, f"[retriever] {len(retrieved)} premises")
        cached_tactics = _cached_plan_tactics(plan_cache, node.state.pretty)
        if cached_tactics:
            _log(config, f"[planner] cached tactics: {', '.join(cached_tactics[:4])}")
            planner_stats["planner_cache_hit_states"] += 1
            planner_stats["planner_cached_tactic_candidates"] += len(cached_tactics)
        cached_tactic_set = set(cached_tactics)
        request_k = max(
            1,
            int(
                config.generation_budget_per_state
                if config.generation_budget_per_state > 0
                else config.suggestions_per_state
            ),
        )
        _log(config, f"[llm] requesting {request_k} suggestions")
        suggestions = llm.propose(
            node.state,
            retrieved,
            request_k,
            instruction=round_instruction,
            context=config.context,
            mode=mode,
        )
        if cached_tactics:
            suggestions = _merge_suggestions(cached_tactics, suggestions)
        if config.autop:
            suggestions = _merge_suggestions(suggestions, _autop_tactics())
        suggestions = _sanitize_suggestions(suggestions, theorem=config.theorem)
        suggestions = _rank_and_limit_suggestions(
            suggestions,
            cached_tactics=cached_tactics,
            execution_budget_per_state=config.execution_budget_per_state,
            verification_level=config.verification_level,
        )
        if suggestions:
            _log(config, f"[llm] suggestions: {', '.join(suggestions)}")
        for tactic in suggestions:
            if steps >= config.max_steps:
                break
            if tactic in cached_tactic_set:
                planner_stats["planner_cached_tactic_tries"] += 1
            if not _consume_state_tactic_budget(tactic_head_attempts, node.state.key, tactic):
                _log(config, f"[policy] skipped over-budget tactic head: {tactic}")
                continue
            cache_key = (node.state.key, tactic)
            cached = cache_key in step_cache
            if cached:
                result = step_cache[cache_key]
                elapsed_ms = 0
            else:
                start = time.perf_counter()
                _log(config, f"[try] {tactic}")
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
                return _with_planner_stats(
                    SearchResult(True, final_proof, steps, None),
                    planner_stats,
                    plan_cache,
                )
            if result.ok and result.new_state is not None:
                made_progress = True
                if _remember_plan_tactic(plan_cache, node.state.pretty, tactic, result.new_state.pretty):
                    planner_stats["planner_remembered_tactics"] += 1
                new_proof = node.proof + [tactic]
                if len(new_proof) > len(best_progress):
                    best_progress = new_proof
                    _emit_progress(config, new_proof)
                score = _node_score(len(new_proof), result.new_state)
                prev = best_seen.get(result.new_state.key)
                state_h = state_hash(result.new_state.pretty)
                prev_h = best_seen_hash.get(state_h)
                if (prev is None or score < prev) and (prev_h is None or score < prev_h):
                    best_seen[result.new_state.key] = score
                    best_seen_hash[state_h] = score
                    heapq.heappush(frontier, _Node(score=score, state=result.new_state, proof=new_proof))
                    _cap_frontier(frontier, config.beam_width)
                continue

            if not result.ok and config.repair_attempts > 0:
                if steps >= config.max_steps:
                    break
                error_kind = _error_kind(result.error) or "other"
                if not _allow_repair_for_error_kind(
                    repair_error_attempts, node.state.key, error_kind
                ):
                    _log(config, f"[repair] skipped repeated error kind: {error_kind}")
                    continue
                _log(config, f"[repair] tactic failed: {tactic}")
                repaired = _attempt_repair(
                    runner=runner,
                    llm=llm,
                    retriever=retriever,
                    trace=trace,
                    step_cache=step_cache,
                    tactic_head_attempts=tactic_head_attempts,
                    state=node.state,
                    theorem=config.theorem,
                    failed_tactic=tactic,
                    error=result.error or "unknown error",
                    config=config,
                    mode=mode,
                    instruction=search_instruction,
                    step_budget=max(0, config.max_steps - steps),
                )
                steps += repaired.steps_used
                if repaired.solved:
                    final_proof = node.proof + [repaired.tactic]
                    _emit_progress(config, final_proof)
                    return _with_planner_stats(
                        SearchResult(True, final_proof, steps, None),
                        planner_stats,
                        plan_cache,
                    )
                if repaired.new_state is not None:
                    made_progress = True
                    if _remember_plan_tactic(
                        plan_cache,
                        node.state.pretty,
                        repaired.tactic,
                        repaired.new_state.pretty,
                    ):
                        planner_stats["planner_remembered_tactics"] += 1
                    new_proof = node.proof + [repaired.tactic]
                    if len(new_proof) > len(best_progress):
                        best_progress = new_proof
                        _emit_progress(config, new_proof)
                    score = _node_score(len(new_proof), repaired.new_state)
                    prev = best_seen.get(repaired.new_state.key)
                    state_h = state_hash(repaired.new_state.pretty)
                    prev_h = best_seen_hash.get(state_h)
                    if (prev is None or score < prev) and (prev_h is None or score < prev_h):
                        best_seen[repaired.new_state.key] = score
                        best_seen_hash[state_h] = score
                        heapq.heappush(frontier, _Node(score=score, state=repaired.new_state, proof=new_proof))
                        _cap_frontier(frontier, config.beam_width)

        if made_progress:
            stagnant_expansions = 0
        else:
            stagnant_expansions += 1

    return _with_planner_stats(
        SearchResult(False, [], steps, "Search exhausted or max steps reached"),
        planner_stats,
        plan_cache,
    )


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
    tactic_head_attempts: dict[tuple[str, str], int],
    state: ProofState,
    theorem: str,
    failed_tactic: str,
    error: str,
    config: RunConfig,
    mode: str = "tactic",
    instruction: str | None = None,
    step_budget: int = 0,
) -> _RepairResult:
    retrieved = retriever.retrieve(state, k=config.retriever_k)
    suggestions = llm.repair(
        state,
        retrieved,
        failed_tactic,
        error,
        config.repair_attempts,
        instruction=instruction,
        context=config.context,
        mode=mode,
    )
    suggestions = _sanitize_suggestions(
        suggestions,
        theorem=theorem,
        reject={failed_tactic},
    )
    suggestions = _rank_and_limit_suggestions(
        suggestions,
        cached_tactics=[],
        execution_budget_per_state=config.execution_budget_per_state,
        verification_level=config.verification_level,
    )
    steps_used = 0
    for tactic in suggestions:
        if not _consume_state_tactic_budget(tactic_head_attempts, state.key, tactic):
            _log(config, f"[policy] skipped over-budget tactic head: {tactic}")
            continue
        cache_key = (state.key, tactic)
        cached = cache_key in step_cache
        if cached:
            result = step_cache[cache_key]
            elapsed_ms = 0
        else:
            if steps_used >= step_budget:
                continue
            start = time.perf_counter()
            _log(config, f"[repair-try] {tactic}")
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
    search_instruction = _search_instruction(config.instruction, config.theorem)
    tactic_head_attempts: dict[tuple[str, str], int] = {}
    repair_error_attempts: dict[tuple[str, str], int] = {}
    plan_cache: dict[str, list[str]] = {}
    stagnant_turns = 0
    planner_stats = _init_planner_stats()

    while steps < config.max_steps:
        _log(config, f"[state] {state.key} {_summarize_state(state.pretty)}")
        retrieved = retriever.retrieve(state, k=config.retriever_k)
        if retrieved:
            _log(config, f"[retriever] {len(retrieved)} premises")
        round_instruction = search_instruction
        if stagnant_turns >= _STAGNATION_REPLAN_THRESHOLD:
            _log(config, "[planner] stagnation detected; requesting replan-focused script")
            planner_stats["planner_replan_triggers"] += 1
            round_instruction = _with_replan_hint(
                base_instruction=search_instruction,
                best_progress=proof,
                current_state=state.pretty,
            )
        cached_tactics = _cached_plan_tactics(plan_cache, state.pretty)
        if cached_tactics:
            _log(config, f"[planner] cached script heads: {', '.join(cached_tactics[:4])}")
            planner_stats["planner_cache_hit_states"] += 1
            planner_stats["planner_cached_tactic_candidates"] += len(cached_tactics)
        cached_tactic_set = set(cached_tactics)
        request_k = max(
            1,
            int(
                config.generation_budget_per_state
                if config.generation_budget_per_state > 0
                else config.suggestions_per_state
            ),
        )
        if last_error and repair_budget > 0:
            error_kind = _error_kind(last_error) or "other"
            if _allow_repair_for_error_kind(repair_error_attempts, state.key, error_kind):
                _log(config, "[repair] requesting script")
                suggestions = llm.repair(
                    state,
                    retrieved,
                    last_tactic or "",
                    last_error,
                    request_k,
                    instruction=round_instruction,
                    context=config.context,
                    mode="script",
                )
                repair_budget -= 1
            else:
                _log(config, f"[repair] skipped repeated error kind: {error_kind}")
                suggestions = []
        else:
            _log(config, f"[llm] requesting script ({request_k} lines)")
            suggestions = llm.propose(
                state,
                retrieved,
                request_k,
                instruction=round_instruction,
                context=config.context,
                mode="script",
            )
            repair_budget = config.repair_attempts
        if cached_tactics:
            suggestions = _merge_suggestions(cached_tactics, suggestions)

        suggestions = _sanitize_suggestions(
            suggestions,
            theorem=config.theorem,
            reject={last_tactic} if last_tactic else None,
        )
        suggestions = _rank_and_limit_suggestions(
            suggestions,
            cached_tactics=cached_tactics,
            execution_budget_per_state=config.execution_budget_per_state,
            verification_level=config.verification_level,
        )
        if suggestions:
            _log(config, f"[llm] suggestions: {', '.join(suggestions)}")
        elif not (last_error and repair_budget <= 0):
            # No usable script lines from this pass; continue to fallback / next loop.
            if config.autop:
                _log(config, "[policy] no valid script suggestions after filtering")
            else:
                return _with_planner_stats(
                    SearchResult(False, proof, steps, "LLM returned no tactics"),
                    planner_stats,
                    plan_cache,
                )
        else:
            return _with_planner_stats(
                SearchResult(False, proof, steps, "LLM returned no tactics"),
                planner_stats,
                plan_cache,
            )

        progressed = False
        for tactic in suggestions:
            if steps >= config.max_steps:
                break
            if tactic in cached_tactic_set:
                planner_stats["planner_cached_tactic_tries"] += 1
            if not _consume_state_tactic_budget(tactic_head_attempts, state.key, tactic):
                _log(config, f"[policy] skipped over-budget tactic head: {tactic}")
                continue
            start = time.perf_counter()
            _log(config, f"[try] {tactic}")
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
                return _with_planner_stats(
                    SearchResult(True, proof, steps, None),
                    planner_stats,
                    plan_cache,
                )
            if result.ok and result.new_state is not None:
                proof.append(tactic)
                _emit_progress(config, proof)
                if _remember_plan_tactic(plan_cache, state.pretty, tactic, result.new_state.pretty):
                    planner_stats["planner_remembered_tactics"] += 1
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
                if not _consume_state_tactic_budget(tactic_head_attempts, state.key, tactic):
                    _log(config, f"[policy] skipped over-budget tactic head: {tactic}")
                    continue
                start = time.perf_counter()
                _log(config, f"[autop-try] {tactic}")
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
                    return _with_planner_stats(
                        SearchResult(True, proof, steps, None),
                        planner_stats,
                        plan_cache,
                    )
                if result.ok and result.new_state is not None:
                    proof.append(tactic)
                    _emit_progress(config, proof)
                    if _remember_plan_tactic(plan_cache, state.pretty, tactic, result.new_state.pretty):
                        planner_stats["planner_remembered_tactics"] += 1
                    state = result.new_state
                    last_error = None
                    last_tactic = None
                    progressed = True
                    break
                last_error = result.error or "unknown error"
                last_tactic = tactic

        if progressed:
            stagnant_turns = 0
        else:
            stagnant_turns += 1

        if not progressed and last_error and repair_budget <= 0:
            return _with_planner_stats(
                SearchResult(False, proof, steps, last_error),
                planner_stats,
                plan_cache,
            )

    return _with_planner_stats(
        SearchResult(False, proof, steps, "Search exhausted or max steps reached"),
        planner_stats,
        plan_cache,
    )


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
        "simp",
        "ring_nf",
        "linarith",
        "nlinarith",
        "aesop",
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


_TACTIC_HEAD_RANK_PRIORITIES: dict[str, int] = {
    "rfl": 0,
    "exact": 0,
    "simpa": 0,
    "assumption": 0,
    "constructor": 1,
    "intro": 1,
    "apply": 1,
    "refine": 1,
    "cases": 2,
    "rcases": 2,
    "rw": 2,
    "simp": 3,
    "ring_nf": 3,
    "norm_num": 3,
    "linarith": 4,
    "nlinarith": 5,
    "aesop": 6,
}

_STRICT_VERIFY_BLOCK_FRAGMENTS = (
    "all_goals",
    "repeat",
    "<;>",
    "first |",
)


def _rank_and_limit_suggestions(
    suggestions: list[str],
    *,
    cached_tactics: list[str],
    execution_budget_per_state: int,
    verification_level: str,
) -> list[str]:
    verify = (verification_level or "light").strip().lower()
    verified = [
        tactic
        for tactic in suggestions
        if _verify_tactic_for_level(tactic, verify)
    ]
    cached_set = set(cached_tactics)
    ranked = sorted(verified, key=lambda tactic: _tactic_rank_key(tactic, cached_set))
    if execution_budget_per_state > 0:
        ranked = ranked[:execution_budget_per_state]
    return ranked


def _verify_tactic_for_level(tactic: str, level: str) -> bool:
    verify = (level or "light").strip().lower()
    if verify in {"", "none", "light"}:
        return True
    lowered = tactic.lower()
    if ";" in lowered:
        return False
    for fragment in _STRICT_VERIFY_BLOCK_FRAGMENTS:
        if fragment in lowered:
            return False
    return True


def _tactic_rank_key(tactic: str, cached_tactics: set[str]) -> tuple[int, int, int, int, str]:
    head = _tactic_head(tactic)
    priority = _TACTIC_HEAD_RANK_PRIORITIES.get(head, 7)
    cached_rank = 0 if tactic in cached_tactics else 1
    token_count = len(tactic.split())
    return (cached_rank, priority, token_count, len(tactic), tactic)


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
    # Prefer states with fewer goals/hypotheses and fewer unresolved metavariables.
    meta_vars = len(re.findall(r"\?m(?:_[0-9]+|[0-9]+)", text))
    existential_count = text.count("∃")
    char_penalty = min(500, len(text) // 8)
    return (
        goals * 140
        + hyps * 10
        + meta_vars * 30
        + existential_count * 12
        + char_penalty
    )


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


_EXPENSIVE_TACTIC_STATE_LIMITS: dict[str, int] = {
    "aesop": 1,
    "nlinarith": 2,
    "linarith": 3,
}

_REPAIR_ERROR_KIND_LIMITS: dict[str, int] = {
    "unknown_identifier": 1,
    "parse_error": 1,
    "timeout": 1,
    "rewrite_failed": 2,
    "unsolved_goals": 2,
    "type_mismatch": 3,
    "other": 2,
}

_BLOCKED_TACTIC_HEADS: set[str] = {
    "admit",
    "exact?",
    "apply?",
    "library_search",
    "simp?",
    "aesop?",
}

_STAGNATION_REPLAN_THRESHOLD = 3
_MAX_PLAN_CACHE_ENTRIES = 2000
_MAX_PLAN_TACTICS_PER_STATE = 6


def _search_instruction(instruction: str | None, theorem: str) -> str:
    theorem = theorem.strip()
    extra = (
        "Do not reference the theorem currently being proved by its name"
        f" (`{theorem}`) in tactics. It is not available as a premise."
    )
    base = instruction.strip() if instruction else ""
    if base:
        return base + "\n\n" + extra
    return extra


def _init_planner_stats() -> dict[str, int]:
    return {
        "planner_cache_hit_states": 0,
        "planner_cached_tactic_candidates": 0,
        "planner_cached_tactic_tries": 0,
        "planner_replan_triggers": 0,
        "planner_remembered_tactics": 0,
    }


def _with_planner_stats(
    result: SearchResult,
    stats: dict[str, int],
    plan_cache: dict[str, list[str]],
) -> SearchResult:
    merged = dict(stats)
    merged["planner_cache_entries"] = len(plan_cache)
    return SearchResult(
        solved=result.solved,
        proof=result.proof,
        steps=result.steps,
        error=result.error,
        stats=merged,
    )


def _with_replan_hint(
    *,
    base_instruction: str,
    best_progress: list[str],
    current_state: str,
) -> str:
    hints = [
        "Replan from the current goal state.",
        "If the previous tactic family stalls, switch to a different tactic family.",
        f"Current goal summary: {_summarize_state(current_state, max_chars=260)}",
    ]
    if best_progress:
        recent = ", ".join(best_progress[-6:])
        hints.append(f"Recent progress tactics: {recent}")
    hints.append("Prefer short, composable tactics over one-shot expensive tactics.")
    return base_instruction.rstrip() + "\n\n" + "\n".join(hints)


def _cached_plan_tactics(plan_cache: dict[str, list[str]], state_pretty: str) -> list[str]:
    return list(plan_cache.get(state_hash(state_pretty), []))


def _remember_plan_tactic(
    plan_cache: dict[str, list[str]],
    state_pretty: str,
    tactic: str,
    new_state_pretty: str,
) -> bool:
    head = _tactic_head(tactic)
    if not head:
        return False
    parent_complexity = _state_complexity(state_pretty)
    child_complexity = _state_complexity(new_state_pretty)
    # Keep tactics that usually maintain or simplify the state.
    if child_complexity > parent_complexity + 20:
        return False
    key = state_hash(state_pretty)
    cached = list(plan_cache.get(key, []))
    if tactic in cached:
        return False
    if len(cached) >= _MAX_PLAN_TACTICS_PER_STATE:
        return False
    cached.append(tactic)
    plan_cache[key] = cached
    # Bound memory growth in long runs.
    while len(plan_cache) > _MAX_PLAN_CACHE_ENTRIES:
        oldest = next(iter(plan_cache))
        if oldest == key:
            break
        plan_cache.pop(oldest, None)
    return True


def _sanitize_suggestions(
    suggestions: list[str],
    theorem: str,
    reject: set[str] | None = None,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    blocked = {item.strip() for item in (reject or set()) if item.strip()}
    theorem_name = theorem.strip()
    for raw in suggestions:
        tactic = (raw or "").strip()
        if not tactic:
            continue
        if _is_disallowed_tactic(tactic):
            continue
        if tactic in seen or tactic in blocked:
            continue
        if theorem_name and _contains_lean_identifier(tactic, theorem_name):
            continue
        seen.add(tactic)
        cleaned.append(tactic)
    return cleaned


def _contains_lean_identifier(text: str, name: str) -> bool:
    if not name:
        return False
    start = 0
    while True:
        idx = text.find(name, start)
        if idx < 0:
            return False
        left = text[idx - 1] if idx > 0 else ""
        right_idx = idx + len(name)
        right = text[right_idx] if right_idx < len(text) else ""
        if not _is_ident_char(left) and not _is_ident_char(right):
            return True
        start = idx + 1


def _is_ident_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in {"_", "'"})


def _consume_state_tactic_budget(
    counter: dict[tuple[str, str], int],
    state_key: str,
    tactic: str,
) -> bool:
    head = _tactic_head(tactic)
    if not head:
        return True
    limit = _EXPENSIVE_TACTIC_STATE_LIMITS.get(head)
    if limit is None:
        return True
    key = (state_key, head)
    used = counter.get(key, 0)
    if used >= limit:
        return False
    counter[key] = used + 1
    return True


def _tactic_head(tactic: str) -> str:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_']*)", tactic or "")
    if not match:
        return ""
    return match.group(1).lower()


def _is_disallowed_tactic(tactic: str) -> bool:
    stripped = tactic.strip()
    lower = stripped.lower()
    if not stripped:
        return True
    if lower in {"sorry", "by sorry", "admit", "by admit"}:
        return True
    if lower.startswith("set_option "):
        return True
    head = _tactic_head(stripped)
    if head in _BLOCKED_TACTIC_HEADS:
        return True
    if "exact?" in lower or "apply?" in lower or "simp?" in lower:
        return True
    return False


def _allow_repair_for_error_kind(
    counter: dict[tuple[str, str], int],
    state_key: str,
    error_kind: str,
) -> bool:
    kind = (error_kind or "other").strip().lower() or "other"
    limit = _REPAIR_ERROR_KIND_LIMITS.get(kind, _REPAIR_ERROR_KIND_LIMITS["other"])
    key = (state_key, kind)
    used = counter.get(key, 0)
    if used >= limit:
        return False
    counter[key] = used + 1
    return True
