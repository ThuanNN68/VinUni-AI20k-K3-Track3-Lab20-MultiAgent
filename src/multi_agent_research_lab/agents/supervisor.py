"""Supervisor agent — decides which worker runs next.

Routing policy
--------------
The supervisor inspects the current ``ResearchState`` and returns a route
string.  Routing is deterministic (no LLM call needed) to keep cost and
latency low for control logic:

  1. If ``sources`` is empty → go to **researcher**
  2. If ``research_notes`` is None → go to **researcher** (search happened
     but LLM synthesis not yet done — shouldn't normally occur)
  3. If ``analysis_notes`` is None → go to **analyst**
  4. If ``final_answer`` is None → go to **writer**
  5. Otherwise → **done**

Hard stops:
- ``iteration >= max_iterations`` → **done** (prevents infinite loops)
- ``errors`` is non-empty AND iteration > 2 → **done** (fail-fast after retries)
"""

from __future__ import annotations

import logging

from langfuse import observe  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import update_span

logger = logging.getLogger(__name__)

DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = AgentName.SUPERVISOR

    def __init__(self) -> None:
        self._max_iterations = get_settings().max_iterations

    @observe(name="supervisor-agent")
    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route and return state."""
        update_span(
            name="supervisor-agent",
            input={
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
                "errors": len(state.errors),
            },
        )

        next_route = self._decide(state)
        state.record_route(next_route)

        update_span(output={"next_route": next_route})
        logger.info(
            "[Supervisor] iteration=%d → %s", state.iteration, next_route
        )
        return state

    def _decide(self, state: ResearchState) -> str:
        # Hard stops
        if state.iteration >= self._max_iterations:
            logger.warning(
                "[Supervisor] Max iterations (%d) reached — stopping", self._max_iterations
            )
            return DONE
        if state.errors and state.iteration > 2:
            logger.warning("[Supervisor] Errors detected — stopping early")
            return DONE

        # Normal routing
        if not state.sources:
            return AgentName.RESEARCHER
        if state.research_notes is None:
            return AgentName.RESEARCHER
        if state.analysis_notes is None:
            return AgentName.ANALYST
        if state.final_answer is None:
            return AgentName.WRITER
        return DONE
