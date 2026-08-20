"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Pre-load .env into os.environ BEFORE any langfuse import.
# Langfuse v4 reads LANGFUSE_* env vars at import time for the OTel exporter.
# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv(override=False)  # does NOT overwrite already-set env vars

import time  # noqa: E402
from typing import Annotated  # noqa: E402

import typer  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from multi_agent_research_lab.core.config import get_settings  # noqa: E402
from multi_agent_research_lab.core.errors import StudentTodoError  # noqa: E402
from multi_agent_research_lab.core.schemas import ResearchQuery  # noqa: E402
from multi_agent_research_lab.core.state import ResearchState  # noqa: E402
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow  # noqa: E402
from multi_agent_research_lab.observability.logging import configure_logging  # noqa: E402
from multi_agent_research_lab.observability.tracing import (  # noqa: E402
    flush,
    get_langfuse_client,
    update_span,
)
from multi_agent_research_lab.services.llm_client import LLMClient  # noqa: E402

# Langfuse @observe — wraps the CLI functions so all nested spans belong to one trace
try:
    from langfuse import observe  # type: ignore[import-untyped]
except ImportError:

    def observe(*, name: str = ""):  # type: ignore[misc]
        def decorator(fn):  # type: ignore[type-arg]
            return fn

        return decorator


app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Eagerly initialise Langfuse so the first trace is ready
    get_langfuse_client()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Baseline command — single LLM call end-to-end
# ---------------------------------------------------------------------------


@app.command()
@observe(name="baseline-run")
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline (one LLM call, no search)."""
    _init()
    request = _parse_query(query)

    update_span(name="baseline-run", input={"query": query}, metadata={"tags": ["baseline"]})

    system_prompt = (
        "You are a knowledgeable research assistant. "
        "Answer the user's question clearly and concisely in 2-4 paragraphs."
    )

    t0 = time.perf_counter()
    llm = LLMClient()
    resp = llm.complete(system_prompt=system_prompt, user_prompt=query, name="baseline-completion")
    latency = time.perf_counter() - t0

    state = ResearchState(request=request)
    state.final_answer = resp.content

    update_span(
        output={"answer_chars": len(resp.content)},
        metadata={
            "latency_s": round(latency, 3),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        },
    )

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))

    # Print metrics table
    tbl = Table(title="Baseline Metrics", show_header=True)
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Latency", f"{latency:.2f}s")
    tbl.add_row("Input tokens", str(resp.input_tokens or "-"))
    tbl.add_row("Output tokens", str(resp.output_tokens or "-"))
    tbl.add_row("Est. cost", f"${resp.cost_usd:.5f}" if resp.cost_usd else "-")
    console.print(tbl)

    flush()


# ---------------------------------------------------------------------------
# Multi-agent command
# ---------------------------------------------------------------------------


@app.command("multi-agent")
@observe(name="multi-agent-run")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent supervisor → researcher → analyst → writer workflow."""
    _init()

    update_span(name="multi-agent-run", input={"query": query}, metadata={"tags": ["multi-agent"]})

    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        t0 = time.perf_counter()
        result = workflow.run(state)
        latency = time.perf_counter() - t0
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Summarise agent costs
    total_in = sum(r.metadata.get("input_tokens") or 0 for r in result.agent_results)
    total_out = sum(r.metadata.get("output_tokens") or 0 for r in result.agent_results)
    total_cost = sum(r.metadata.get("cost_usd") or 0.0 for r in result.agent_results)
    writer_res = next((r for r in result.agent_results if r.agent == "writer"), None)
    citation_cov = writer_res.metadata.get("citation_coverage") if writer_res else None

    update_span(
        output={"answer_chars": len(result.final_answer or "")},
        metadata={
            "latency_s": round(latency, 3),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": total_cost,
            "route_history": result.route_history,
            "citation_coverage": citation_cov,
        },
    )

    # Display results
    console.print(Panel.fit(result.final_answer or "(no answer)", title="[green]Final Answer"))
    console.print(f"\n[dim]Route:[/dim] {' → '.join(result.route_history)}")

    tbl = Table(title="Multi-Agent Metrics", show_header=True)
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Latency", f"{latency:.2f}s")
    tbl.add_row("Routing steps", str(len(result.route_history)))
    tbl.add_row("Sources found", str(len(result.sources)))
    tbl.add_row("Total input tokens", str(total_in))
    tbl.add_row("Total output tokens", str(total_out))
    tbl.add_row("Est. total cost", f"${total_cost:.5f}")
    tbl.add_row("Citation coverage", f"{citation_cov:.0%}" if citation_cov is not None else "-")
    if result.errors:
        tbl.add_row("[red]Errors", str(len(result.errors)))
    console.print(tbl)

    flush()


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------

_BENCHMARK_QUERIES = [
    "Research GraphRAG state-of-the-art",
    "What are the key differences between RAG and fine-tuning for LLMs?",
    "Explain multi-agent AI architectures and their tradeoffs",
]


@app.command()
@observe(name="benchmark-run")
def benchmark(
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Report output path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run baseline vs multi-agent benchmark and write a markdown report."""
    from pathlib import Path

    from multi_agent_research_lab.core.schemas import ResearchQuery
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report

    _init()
    update_span(name="benchmark-run", metadata={"tags": ["benchmark"]})

    console.print("[bold]Running benchmark...[/bold]")
    all_metrics = []
    workflow = MultiAgentWorkflow()
    llm = LLMClient()

    for q in _BENCHMARK_QUERIES:
        console.print(f"\n[cyan]Query:[/cyan] {q}")

        # Baseline runner
        def _baseline_runner(query: str) -> ResearchState:
            state = ResearchState(request=ResearchQuery(query=query))
            resp = llm.complete(
                system_prompt="You are a knowledgeable research assistant. Answer concisely.",
                user_prompt=query,
                name="benchmark-baseline",
            )
            state.final_answer = resp.content
            from multi_agent_research_lab.core.schemas import AgentName, AgentResult
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=resp.content,
                    metadata={"cost_usd": resp.cost_usd},
                )
            )
            return state

        # Multi-agent runner
        def _multi_runner(query: str) -> ResearchState:
            state = ResearchState(request=ResearchQuery(query=query))
            return workflow.run(state)

        _, b_metrics = run_benchmark(f"baseline | {q[:30]}", q, _baseline_runner)
        _, m_metrics = run_benchmark(f"multi-agent | {q[:30]}", q, _multi_runner)
        all_metrics.extend([b_metrics, m_metrics])
        console.print(
            f"  baseline {b_metrics.latency_seconds:.1f}s  "
            f"multi-agent {m_metrics.latency_seconds:.1f}s"
        )

    report = render_markdown_report(all_metrics)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    console.print(f"\n[green]Report written → {out_path}[/green]")
    flush()


if __name__ == "__main__":
    app()

