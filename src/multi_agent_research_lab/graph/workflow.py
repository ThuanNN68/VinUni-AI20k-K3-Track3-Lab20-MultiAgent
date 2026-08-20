"""LangGraph multi-agent workflow.

Graph structure
---------------
  supervisor ──► researcher ──► supervisor
            ──► analyst    ──► supervisor
            ──► writer     ──► supervisor
            ──► done

All node functions wrap the corresponding agent's ``run()`` method.
The conditional edge reads ``state.route_history[-1]`` set by the supervisor.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State adapter — LangGraph works with dict-like objects; we use a thin shim
# ---------------------------------------------------------------------------


def _state_to_dict(state: ResearchState) -> dict:
    return state.model_dump()


def _dict_to_state(d: dict) -> ResearchState:
    return ResearchState.model_validate(d)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Agents are instantiated once and reused across calls (stateless by design).
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    # ------------------------------------------------------------------
    # Node functions — each receives a plain dict from LangGraph and
    # returns the updated dict.  We convert to ResearchState and back.
    # ------------------------------------------------------------------

    def _node_supervisor(self, data: dict) -> dict:
        state = _dict_to_state(data)
        state = self._supervisor.run(state)
        return _state_to_dict(state)

    def _node_researcher(self, data: dict) -> dict:
        state = _dict_to_state(data)
        try:
            state = self._researcher.run(state)
        except Exception as exc:
            logger.exception("[Workflow] Researcher failed: %s", exc)
            state.errors.append(str(exc))
        return _state_to_dict(state)

    def _node_analyst(self, data: dict) -> dict:
        state = _dict_to_state(data)
        try:
            state = self._analyst.run(state)
        except Exception as exc:
            logger.exception("[Workflow] Analyst failed: %s", exc)
            state.errors.append(str(exc))
        return _state_to_dict(state)

    def _node_writer(self, data: dict) -> dict:
        state = _dict_to_state(data)
        try:
            state = self._writer.run(state)
        except Exception as exc:
            logger.exception("[Workflow] Writer failed: %s", exc)
            state.errors.append(str(exc))
        return _state_to_dict(state)

    # ------------------------------------------------------------------
    # Conditional routing — reads the latest route from route_history
    # ------------------------------------------------------------------

    @staticmethod
    def _route(
        data: dict,
    ) -> Literal["researcher", "analyst", "writer", "__end__"]:
        history: list[str] = data.get("route_history", [])
        last = history[-1] if history else DONE
        if last == AgentName.RESEARCHER:
            return "researcher"
        if last == AgentName.ANALYST:
            return "analyst"
        if last == AgentName.WRITER:
            return "writer"
        return END  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------

    def build(self) -> object:
        """Compile and return the LangGraph CompiledGraph."""
        builder: StateGraph = StateGraph(dict)

        # Nodes
        builder.add_node("supervisor", self._node_supervisor)
        builder.add_node("researcher", self._node_researcher)
        builder.add_node("analyst", self._node_analyst)
        builder.add_node("writer", self._node_writer)

        # Entry point
        builder.set_entry_point("supervisor")

        # Workers always hand control back to supervisor
        for worker in ("researcher", "analyst", "writer"):
            builder.add_edge(worker, "supervisor")

        # Supervisor decides next step
        builder.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        return builder.compile()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final ResearchState."""
        graph = self.build()
        initial = _state_to_dict(state)
        final_dict = graph.invoke(initial)
        return _dict_to_state(final_dict)
