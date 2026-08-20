"""Critic agent — validates final answer, checks citations, and flags potential hallucinations."""

from __future__ import annotations

import logging

from langfuse import observe  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import update_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """You are a rigorous factual reviewer and safety critic.
Review the proposed final answer against the retrieved sources and analysis notes.

Your task:
1. Verify citation coverage: Are all major claims backed by cited sources [N]?
2. Fact-check for unsupported claims or potential hallucinations.
3. Provide a brief verdict (APPROVED / NEEDS_REVISION) and a list of feedback notes.

Format:
Verdict: [APPROVED or NEEDS_REVISION]
Citation Coverage: [Good / Fair / Poor]
Critique Notes:
- (Points on factual accuracy, citation correctness, tone)
"""


class CriticAgent(BaseAgent):
    """Fact-checking, citation validation, and hallucination inspection agent."""

    name = AgentName.CRITIC

    def __init__(self) -> None:
        self._llm = LLMClient()

    @observe(name="critic-agent", as_type="agent")
    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append evaluation findings to agent results."""
        update_span(
            name="critic-agent",
            input={
                "has_final_answer": state.final_answer is not None,
                "sources_count": len(state.sources),
            },
        )

        sources_text = "\n".join(
            f"[{i + 1}] {s.title}: {s.snippet}" for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or '(none)'}\n\n"
            f"Proposed Final Answer:\n{state.final_answer or '(none)'}"
        )

        resp = self._llm.complete(
            system_prompt=_CRITIC_PROMPT,
            user_prompt=user_prompt,
            name="critic-review",
        )

        result = AgentResult(
            agent=AgentName.CRITIC,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
        state.agent_results.append(result)

        update_span(
            output={"critic_review_length": len(resp.content)},
            metadata=result.metadata,
        )

        logger.info("[Critic] Review completed (%d chars)", len(resp.content))
        return state
