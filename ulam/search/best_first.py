from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMClient
from ..retrieve import Retriever
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
) -> SearchResult:
    initial_state = runner.start(config.file_path, config.theorem)
    _log(config, f"[init] state={initial_state.key}")
    if config.instruction:
        _log(config, f"[instruction] {config.instruction}")
    frontier: list[_Node] = []
    heapq.heappush(frontier, _Node(score=0, state=initial_state, proof=[]))
    best_seen: dict[str, int] = {initial_state.key: 0}
    step_cache: dict[tuple[str, str], TacticResult] = {}
    steps = 0

    while frontier and steps < config.max_steps:
        node = heapq.heappop(frontier)
        _log(config, f"[state] {node.state.key} {_summarize_state(node.state.pretty)}")
        retrieved = retriever.retrieve(node.state, k=8)
        if retrieved:
            _log(config, f"[retriever] {len(retrieved)} premises")
        _log(config, f"[llm] requesting {config.suggestions_per_state} suggestions")
        suggestions = llm.propose(
            node.state,
            retrieved,
            config.suggestions_per_state,
            instruction=config.instruction,
            context=config.context,
        )
        if suggestions:
            _log(config, f"[llm] suggestions: {', '.join(suggestions)}")
        for tactic in suggestions:
            if steps >= config.max_steps:
                break
            cache_key = (node.state.key, tactic)
            cached = cache_key in step_cache
            if cached:
                result = step_cache[cache_key]
            else:
                result = runner.apply(node.state, tactic, config.timeout_s)
                step_cache[cache_key] = result
                steps += 1
            trace.log_step(_step_from_result(node.state, tactic, result, cached=cached))
            _log(config, _format_result(tactic, result, cached))
            if result.ok and result.is_solved:
                return SearchResult(True, node.proof + [tactic], steps, None)
            if result.ok and result.new_state is not None:
                new_proof = node.proof + [tactic]
                score = len(new_proof)
                prev = best_seen.get(result.new_state.key)
                if prev is None or score < prev:
                    best_seen[result.new_state.key] = score
                    heapq.heappush(frontier, _Node(score=score, state=result.new_state, proof=new_proof))
                    _cap_frontier(frontier, config.beam_width)
                continue

            if not result.ok and config.repair_attempts > 0:
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
                )
                if repaired.solved:
                    return SearchResult(True, node.proof + [repaired.tactic], steps, None)
                if repaired.new_state is not None:
                    new_proof = node.proof + [repaired.tactic]
                    score = len(new_proof)
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
) -> _RepairResult:
    retrieved = retriever.retrieve(state, k=8)
    suggestions = llm.repair(
        state,
        retrieved,
        failed_tactic,
        error,
        config.repair_attempts,
        instruction=config.instruction,
        context=config.context,
    )
    for tactic in suggestions:
        cache_key = (state.key, tactic)
        cached = cache_key in step_cache
        if cached:
            result = step_cache[cache_key]
        else:
            result = runner.apply(state, tactic, config.timeout_s)
            step_cache[cache_key] = result
        trace.log_step(_step_from_result(state, tactic, result, cached=cached))
        _log(config, _format_result(tactic, result, cached))
        if result.ok and result.is_solved:
            return _RepairResult(True, tactic, None)
        if result.ok and result.new_state is not None:
            return _RepairResult(False, tactic, result.new_state)
    return _RepairResult(False, failed_tactic, None)


def _step_from_result(
    state: ProofState, tactic: str, result: TacticResult, cached: bool = False
) -> ProofStep:
    return ProofStep(
        state_key=state.key,
        state_pretty=state.pretty,
        tactic=tactic,
        ok=result.ok,
        error=result.error,
        new_state_key=result.new_state.key if result.new_state else None,
        solved=result.is_solved,
        cached=cached,
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
