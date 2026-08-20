"""LLM client abstraction backed by OpenAI via the Langfuse v4 drop-in wrapper.

The Langfuse OpenAI wrapper (``langfuse.openai``) is a transparent replacement
for ``openai`` that automatically creates a *generation* observation for every
chat completion, capturing model name, token usage, latency, and estimated cost.

In Langfuse v4 the wrapper works via OpenTelemetry instrumentation — no extra
kwargs are needed; everything is captured automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# Langfuse-instrumented OpenAI drop-in — auto-traces every chat completion
from langfuse.openai import openai  # type: ignore[import-untyped]

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# gpt-4o-mini pricing (USD / 1 000 tokens, as of Aug 2025)
_COST_PER_1K_INPUT = 0.000150
_COST_PER_1K_OUTPUT = 0.000600


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI.

    Uses the Langfuse OpenAI integration so every chat completion is
    automatically traced as a *generation* observation — no manual spans needed.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        name: str | None = None,  # kept for interface compatibility; logged only
    ) -> LLMResponse:
        """Return a model completion with full Langfuse tracing."""
        if name:
            logger.debug("LLMClient.complete called as '%s'", name)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except Exception:
            logger.exception("LLMClient.complete failed (model=%s)", self._model)
            raise

        message = resp.choices[0].message
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None

        cost: float | None = None
        if in_tok is not None and out_tok is not None:
            cost = (in_tok / 1000) * _COST_PER_1K_INPUT + (out_tok / 1000) * _COST_PER_1K_OUTPUT

        return LLMResponse(
            content=message.content or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
