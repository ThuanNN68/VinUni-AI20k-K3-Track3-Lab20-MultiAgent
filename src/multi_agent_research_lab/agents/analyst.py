"""Analyst agent — turns research notes into structured insights."""

from __future__ import annotations

import logging

from langfuse import observe  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import update_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a critical analyst reviewing research notes.
Your job is to produce structured analysis of the gathered information.

Structure your output as:
## Key Claims
(Bullet list of the 3-5 most important claims)

## Evidence Strength
(For each claim: rate evidence as Strong / Moderate / Weak and explain why)

## Conflicting Viewpoints
(Any disagreements or tensions between sources)

## Gaps & Limitations
(What is unknown or not covered by the sources)

Be concise — target 200-300 words total. Do not repeat the research notes verbatim."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights using an LLM."""

    name = AgentName.ANALYST

    def __init__(self) -> None:
        self._llm = LLMClient()

    @observe(name="analyst-agent", as_type="agent")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""
        update_span(
            name="analyst-agent",
            input={"research_notes_chars": len(state.research_notes or "")},
        )

        user_prompt = (
            f"Research Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
            f"Audience: {state.request.audience}"
        )

        resp = self._llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            name="analyst-analysis",
        )
        state.analysis_notes = resp.content

        result = AgentResult(
            agent=AgentName.ANALYST,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
        state.agent_results.append(result)

        update_span(
            output={"analysis_chars": len(resp.content)},
            metadata=result.metadata,
        )

        logger.info("[Analyst] Done — %d chars of analysis", len(resp.content))
        return state
