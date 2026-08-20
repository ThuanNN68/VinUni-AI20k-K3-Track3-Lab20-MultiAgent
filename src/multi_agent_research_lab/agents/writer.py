"""Writer agent — synthesises research + analysis into a final answer with citations."""

from __future__ import annotations

import logging

from langfuse import observe  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import update_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a skilled technical writer producing a well-cited research summary.

Instructions:
1. Write a clear, structured answer to the user's query targeting the specified audience.
2. Every factual claim must cite at least one source using inline notation [N] where N is the
   source number from the provided list.
3. Organise with a brief introduction, 2-4 key-finding paragraphs, and a conclusion.
4. Highlight any important caveats from the analysis.
5. Do not hallucinate — only use facts from the research notes and analysis.
6. End with a **References** section listing all cited sources.
Target length: 300-500 words."""


class WriterAgent(BaseAgent):
    """Produces a final answer with inline citations from research and analysis notes."""

    name = AgentName.WRITER

    def __init__(self) -> None:
        self._llm = LLMClient()

    @observe(name="writer-agent", as_type="agent")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer`` with a cited response."""
        update_span(
            name="writer-agent",
            input={
                "query": state.request.query,
                "has_analysis": state.analysis_notes is not None,
                "sources": len(state.sources),
            },
        )

        # Format source list for the prompt
        sources_text = "\n".join(
            f"[{i + 1}] {s.title}" + (f" — {s.url}" if s.url else "")
            for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
            f"Analysis:\n{state.analysis_notes or '(none)'}\n\n"
            f"Sources:\n{sources_text}"
        )

        resp = self._llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            name="writer-synthesis",
        )
        state.final_answer = resp.content

        # Compute citation coverage: how many sources [N] are referenced
        cited = {
            i + 1 for i in range(len(state.sources)) if f"[{i + 1}]" in resp.content
        }
        citation_coverage = len(cited) / len(state.sources) if state.sources else 0.0

        result = AgentResult(
            agent=AgentName.WRITER,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
                "citation_coverage": citation_coverage,
                "cited_sources": sorted(cited),
            },
        )
        state.agent_results.append(result)

        update_span(
            output={"answer_chars": len(resp.content), "citation_coverage": citation_coverage},
            metadata=result.metadata,
        )

        logger.info(
            "[Writer] Done — %d chars, %.0f%% citation coverage",
            len(resp.content),
            citation_coverage * 100,
        )
        return state
