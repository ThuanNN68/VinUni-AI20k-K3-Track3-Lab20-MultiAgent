"""Benchmark — measures latency, cost, quality, citation coverage, and failure rate."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _compute_citation_coverage(state: ResearchState) -> float | None:
    """Fraction of retrieved sources explicitly cited in the final answer."""
    if not state.sources or not state.final_answer:
        return None
    cited = sum(
        1 for i in range(len(state.sources)) if f"[{i + 1}]" in state.final_answer
    )
    return cited / len(state.sources)


def _compute_cost(state: ResearchState) -> float | None:
    """Sum cost across all agent results."""
    costs = [r.metadata.get("cost_usd") for r in state.agent_results]
    if not any(c is not None for c in costs):
        return None
    return sum(c or 0.0 for c in costs)


def _quality_heuristic(state: ResearchState) -> float | None:
    """Simple heuristic quality score 0-10:
    - 2 pts: answer exists and is >200 chars
    - 3 pts: has citations
    - 3 pts: has analysis notes (multi-agent only)
    - 2 pts: route_history shows full pipeline
    """
    if not state.final_answer:
        return 0.0
    score = 0.0
    if len(state.final_answer) > 200:
        score += 2.0
    cov = _compute_citation_coverage(state)
    if cov and cov > 0:
        score += 3.0
    if state.analysis_notes:
        score += 3.0
    expected_agents = {AgentName.RESEARCHER, AgentName.ANALYST, AgentName.WRITER}
    if expected_agents.issubset(set(state.route_history)):
        score += 2.0
    return min(score, 10.0)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and failure rate."""
    failed = False
    state: ResearchState | None = None

    started = perf_counter()
    try:
        state = runner(query)
    except Exception:
        logger.exception("[Benchmark] Runner '%s' raised an exception", run_name)
        failed = True
    latency = perf_counter() - started

    if state is None:
        # Dummy state for failure case
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_compute_cost(state),
        quality_score=_quality_heuristic(state) if not failed else 0.0,
        citation_coverage=_compute_citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes=f"errors={state.errors}" if state.errors else "",
    )
    return state, metrics
