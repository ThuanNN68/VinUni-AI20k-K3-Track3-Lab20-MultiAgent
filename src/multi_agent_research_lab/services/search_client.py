"""Search client backed by Tavily with a deterministic mock fallback.

Falls back to the mock when no TAVILY_API_KEY is configured so the lab
can run end-to-end without an internet subscription.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data — used when Tavily is not configured
# ---------------------------------------------------------------------------

_MOCK_SOURCES: list[SourceDocument] = [
    SourceDocument(
        title="GraphRAG: Unlocking LLM discovery on narrative private data",
        url="https://arxiv.org/abs/2404.16130",
        snippet=(
            "GraphRAG is a structured approach to Retrieval-Augmented Generation that uses an LLM "
            "to build a knowledge graph from private text corpora, enabling global and local "
            "summarisation queries that plain vector RAG cannot answer."
        ),
    ),
    SourceDocument(
        title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        url="https://arxiv.org/abs/2404.16130",
        snippet=(
            "We show that GraphRAG significantly outperforms naive RAG on community-level "
            "comprehensiveness and diversity for global sensemaking tasks across two large corpora."
        ),
    ),
    SourceDocument(
        title="Microsoft GraphRAG — GitHub",
        url="https://github.com/microsoft/graphrag",
        snippet=(
            "Open-source Python implementation of GraphRAG. Supports both global and local search, "
            "community detection via Leiden algorithm, and structured entity extraction."
        ),
    ),
    SourceDocument(
        title="HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs",
        url="https://arxiv.org/abs/2405.14831",
        snippet=(
            "HippoRAG uses a knowledge-graph-like associative memory structure inspired by the "
            "human hippocampal indexing theory to improve multi-hop reasoning in RAG."
        ),
    ),
    SourceDocument(
        title="Benchmarking RAG Approaches on Knowledge-Intensive QA",
        url="https://arxiv.org/abs/2409.05571",
        snippet=(
            "Systematic comparison of vector RAG, GraphRAG, and hybrid approaches across three "
            "public benchmarks. GraphRAG excels at global questions but incurs ~3× token cost."
        ),
    ),
]


class SearchClient:
    """Provider-agnostic search client with Tavily backend and mock fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.tavily_api_key
        self._client = None
        if self._api_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]

                self._client = TavilyClient(api_key=self._api_key)
                logger.info("SearchClient: using Tavily backend")
            except ImportError:
                logger.warning("tavily-python not installed; falling back to mock sources")
        else:
            logger.info("SearchClient: no TAVILY_API_KEY — using mock sources")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to *query*."""
        if self._client is not None:
            return self._search_tavily(query, max_results)
        return _MOCK_SOURCES[:max_results]

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            response = self._client.search(  # type: ignore[union-attr]
                query=query,
                max_results=max_results,
                include_answer=False,
            )
            results: list[SourceDocument] = []
            for r in response.get("results", []):
                results.append(
                    SourceDocument(
                        title=r.get("title", "Untitled"),
                        url=r.get("url"),
                        snippet=r.get("content", ""),
                        metadata={"score": r.get("score", 0.0)},
                    )
                )
            logger.info("SearchClient: retrieved %d results from Tavily", len(results))
            return results
        except Exception:
            logger.exception("Tavily search failed; falling back to mock sources")
            return _MOCK_SOURCES[:max_results]
