"""Langfuse tracing module — compatible with Langfuse v4.

Langfuse v4 uses OpenTelemetry under the hood. Key API:
- ``from langfuse import observe``          — decorator for spans/traces
- ``langfuse.get_client().update_current_span(...)``  — enrich current span
- ``langfuse.get_client().update_current_generation(...)`` — enrich LLM gen
- ``langfuse.get_client().set_current_trace_io(...)`` — set trace I/O
- ``from langfuse.openai import openai``   — drop-in OpenAI wrapper

Integration strategy
--------------------
* **OpenAI calls** — auto-traced via ``langfuse.openai`` drop-in imported
  in ``services/llm_client.py``.
* **Agent / workflow spans** — each agent ``run()`` is decorated with
  ``@observe(as_type="agent")``; supervisor decisions are typed as ``agent``.
* **Top-level trace** — CLI commands are wrapped with ``@observe`` so all
  nested spans belong to one trace per request.
* **Flush on exit** — ``atexit`` registered to prevent losing buffered events.

Best practices
--------------
* Single client initialised lazily via ``get_langfuse_client()``.
* Keys read from ``Settings`` (never from raw env vars in agent code).
* ``capture_input=False`` on agents to avoid leaking full state dicts.
"""

from __future__ import annotations

import atexit
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse, get_client, observe  # noqa: F401 (re-exported)

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning("langfuse not installed — tracing is a no-op")


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """Return the singleton Langfuse client (or None when disabled)."""
    if not _LANGFUSE_AVAILABLE:
        return None

    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse keys not configured — tracing disabled")
        return None

    # Langfuse v4 reads LANGFUSE_* env vars for its OTel OTLP exporter.
    # We inject them here so the exporter is authenticated even when the
    # .env file was loaded by pydantic-settings but not exported into os.environ.
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    # v4: configure via env vars or constructor kwargs
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    atexit.register(client.flush)
    logger.info("Langfuse v4 tracing enabled → %s", settings.langfuse_host)
    return client



def init_trace(
    name: str,
    input: dict[str, Any] | None = None,  # noqa: A002
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Set trace-level metadata from inside an @observe-decorated function."""
    if not _LANGFUSE_AVAILABLE:
        return
    client = get_langfuse_client()
    if client is None:
        return
    kwargs: dict[str, Any] = {"name": name}
    if input is not None:
        kwargs["input"] = input
    client.set_current_trace_io(**{k: v for k, v in kwargs.items() if k in ("input",)})


def update_span(
    name: str | None = None,
    input: Any = None,  # noqa: A002
    output: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Enrich the current @observe span with extra context."""
    if not _LANGFUSE_AVAILABLE:
        return
    client = get_langfuse_client()
    if client is None:
        return
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if metadata is not None:
        kwargs["metadata"] = metadata
    if kwargs:
        client.update_current_span(**kwargs)


def flush() -> None:
    """Manually flush buffered events — important for short-lived CLI processes."""
    if not _LANGFUSE_AVAILABLE:
        return
    client = get_langfuse_client()
    if client:
        client.flush()
