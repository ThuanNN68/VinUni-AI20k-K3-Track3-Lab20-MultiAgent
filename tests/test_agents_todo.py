"""Unit tests for the SupervisorAgent routing policy.

Replaces the skeleton guard test (which expected StudentTodoError)
now that the agent is fully implemented.
"""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _make_state(**kwargs) -> ResearchState:
    base = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    for k, v in kwargs.items():
        object.__setattr__(base, k, v)
    return base


@pytest.fixture()
def fresh_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


@pytest.fixture()
def supervisor() -> SupervisorAgent:
    return SupervisorAgent()


class TestSupervisorRouting:
    """Verify the deterministic routing policy for all state transitions."""

    def test_routes_to_researcher_when_no_sources(
        self, supervisor: SupervisorAgent, fresh_state: ResearchState
    ) -> None:
        """Empty sources → researcher."""
        result = supervisor.run(fresh_state)
        assert result.route_history[-1] == AgentName.RESEARCHER

    def test_routes_to_analyst_after_research(self, supervisor: SupervisorAgent) -> None:
        """Sources + research_notes present → analyst."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.sources = [SourceDocument(title="T", url="http://x.com", snippet="s")]
        state.research_notes = "some notes"
        result = supervisor.run(state)
        assert result.route_history[-1] == AgentName.ANALYST

    def test_routes_to_writer_after_analysis(self, supervisor: SupervisorAgent) -> None:
        """Sources + research_notes + analysis_notes → writer."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.sources = [SourceDocument(title="T", url="http://x.com", snippet="s")]
        state.research_notes = "some notes"
        state.analysis_notes = "some analysis"
        result = supervisor.run(state)
        assert result.route_history[-1] == AgentName.WRITER

    def test_routes_to_done_when_complete(self, supervisor: SupervisorAgent) -> None:
        """All fields set → done."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.sources = [SourceDocument(title="T", url="http://x.com", snippet="s")]
        state.research_notes = "notes"
        state.analysis_notes = "analysis"
        state.final_answer = "answer"
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"

    def test_stops_at_max_iterations(self, supervisor: SupervisorAgent) -> None:
        """Hard stop when iteration >= max_iterations (default 6)."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        # Force the iteration counter to be at the limit
        state.iteration = supervisor._max_iterations
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"

    def test_increments_iteration(
        self, supervisor: SupervisorAgent, fresh_state: ResearchState
    ) -> None:
        """Each call to run() increments iteration via record_route()."""
        assert fresh_state.iteration == 0
        result = supervisor.run(fresh_state)
        assert result.iteration == 1

    def test_route_history_is_appended(
        self, supervisor: SupervisorAgent, fresh_state: ResearchState
    ) -> None:
        """route_history grows with each supervisor call."""
        result = supervisor.run(fresh_state)
        assert len(result.route_history) == 1
        supervisor.run(result)
        assert len(result.route_history) == 2

    def test_stops_early_with_errors_after_iteration_2(
        self, supervisor: SupervisorAgent
    ) -> None:
        """With errors and iteration > 2, supervisor stops immediately."""
        state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
        state.errors = ["something went wrong"]
        state.iteration = 3  # already past 2
        result = supervisor.run(state)
        assert result.route_history[-1] == "done"
