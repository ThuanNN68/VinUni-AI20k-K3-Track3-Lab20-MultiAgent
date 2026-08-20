"""Search client backed by Tavily with an intelligent offline research corpus retriever.

Supports:
1. Online search via Tavily API (when configured).
2. Offline search across the 30-topic research corpus in `data/corpus/topics/`.
3. Deterministic mock fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_DEFAULT_CORPUS_DIR = Path("data/corpus/topics")


class SearchClient:
    """Provider-agnostic search client with Tavily and 30-Topic Offline Corpus backends."""

    def __init__(self, corpus_dir: Path | str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.tavily_api_key
        self._corpus_dir = Path(corpus_dir) if corpus_dir else _DEFAULT_CORPUS_DIR
        self._client = None
        if self._api_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]

                self._client = TavilyClient(api_key=self._api_key)
                logger.info("SearchClient: using Tavily backend")
            except ImportError:
                logger.warning("tavily-python not installed; using offline corpus")
        else:
            logger.info("SearchClient: no TAVILY_API_KEY — using offline corpus")

    def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        prefer_offline: bool = False,
    ) -> list[SourceDocument]:
        """Search for documents relevant to *query*."""
        if prefer_offline or self._client is None:
            corpus_results = self.search_offline_corpus(query, max_results=max_results)
            if corpus_results:
                return corpus_results

        if self._client is not None:
            return self._search_tavily(query, max_results)

        return self.search_offline_corpus(query, max_results=max_results)

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
                        metadata={"score": r.get("score", 0.0), "source_type": "web_tavily"},
                    )
                )
            logger.info("SearchClient: retrieved %d results from Tavily", len(results))
            return results
        except Exception:
            logger.exception("Tavily search failed; falling back to offline corpus")
            return self.search_offline_corpus(query, max_results=max_results)

    def search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search within the 30-topic offline research corpus."""
        if not self._corpus_dir.exists():
            logger.warning("Corpus directory %s not found", self._corpus_dir)
            return []

        query_terms = set(query.lower().split())
        scored_docs: list[tuple[float, SourceDocument]] = []

        for json_file in self._corpus_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                topic_info = data.get("topic", {})
                topic_name = topic_info.get("name", "")
                topic_tags = topic_info.get("tags", [])
                kb = data.get("knowledge_base", {})

                # Check topic relevance
                topic_score = sum(
                    2.0 for term in query_terms
                    if term in topic_name.lower() or any(term in tag.lower() for tag in topic_tags)
                )

                # Extract knowledge articles
                for art in kb.get("knowledge_articles", []):
                    title = art.get("title", "")
                    content = art.get("content", "")
                    art_id = art.get("article_id", "A")
                    overlap = sum(
                        1.0 for term in query_terms
                        if term in title.lower() or term in content.lower()
                    )
                    score = topic_score + overlap
                    if score > 0 or not query_terms:
                        snippet = content[:600] + ("..." if len(content) > 600 else "")
                        scored_docs.append((
                            score,
                            SourceDocument(
                                title=f"[{art_id}] {title} ({topic_name})",
                                url=f"offline://corpus/{json_file.stem}#{art_id}",
                                snippet=snippet,
                                metadata={"source_id": art_id, "type": "knowledge_article"},
                            ),
                        ))

                # Extract embedded source documents
                for doc in kb.get("source_documents", []):
                    title = doc.get("title", "")
                    content = doc.get("full_text", doc.get("summary", ""))
                    doc_id = doc.get("source_id", "S")
                    overlap = sum(
                        1.0 for term in query_terms
                        if term in title.lower() or term in content.lower()
                    )
                    score = topic_score + overlap
                    if score > 0:
                        snippet = content[:600] + ("..." if len(content) > 600 else "")
                        scored_docs.append((
                            score,
                            SourceDocument(
                                title=f"[{doc_id}] {title}",
                                url=doc.get("url", f"offline://corpus/{json_file.stem}#{doc_id}"),
                                snippet=snippet,
                                metadata={"source_id": doc_id, "type": "source_document"},
                            ),
                        ))
            except Exception as e:
                logger.debug("Failed to parse corpus file %s: %e", json_file, e)

        # Sort by relevance score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored_docs[:max_results]]
        logger.info(
            "SearchClient: retrieved %d documents from offline corpus (%s)",
            len(results),
            self._corpus_dir,
        )
        return results

    def list_topics(self) -> list[dict[str, str]]:
        """List all 30 topics available in the offline corpus."""
        if not self._corpus_dir.exists():
            return []
        topics = []
        for json_file in sorted(self._corpus_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                t = data.get("topic", {})
                meta = data.get("benchmark_metadata", {})
                topics.append({
                    "id": meta.get("topic_id", json_file.stem),
                    "file": json_file.name,
                    "name": t.get("name", json_file.stem),
                    "question": t.get("research_question", ""),
                })
            except Exception:
                continue
        return topics
