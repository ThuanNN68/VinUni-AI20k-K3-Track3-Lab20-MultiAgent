"""Researcher agent — gathers sources and writes research notes."""

from __future__ import annotations

import logging

from langfuse import observe  # type: ignore[import-untyped]

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import update_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a meticulous research assistant.
Given a set of retrieved source snippets, write concise, factual research notes.
- Summarise the key findings from each source.
- Note any contradictions or gaps between sources.
- Keep each point short (1-2 sentences).
- Do NOT add information beyond what the sources contain.
Format: bullet list, ~150-250 words total."""


class ResearcherAgent(BaseAgent):
    """Collects sources via search and synthesises research notes with an LLM."""

    name = AgentName.RESEARCHER

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    @observe(name="researcher-agent", as_type="agent")
    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""
        query = state.request.query
        max_sources = state.request.max_sources

        update_span(
            name="researcher-agent",
            input={"query": query, "max_sources": max_sources},
        )

        # 1. Search
        logger.info("[Researcher] Searching for: %s", query)
        sources = self._search.search(query, max_results=max_sources)
        state.sources = sources

        # 2. Synthesise notes from snippets
        snippets = "\n\n".join(
            f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
        )
        user_prompt = f"Query: {query}\n\nSources:\n{snippets}"
        resp = self._llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            name="researcher-synthesis",
        )
        state.research_notes = resp.content

        # 3. Record result
        result = AgentResult(
            agent=AgentName.RESEARCHER,
            content=resp.content,
            metadata={
                "sources_found": len(sources),
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
        state.agent_results.append(result)

        update_span(
            output={"research_notes_chars": len(resp.content), "sources": len(sources)},
            metadata=result.metadata,
        )

        logger.info(
            "[Researcher] Done — %d sources, %d chars of notes",
            len(sources),
            len(resp.content),
        )
        return state
