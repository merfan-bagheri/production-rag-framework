import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from rag_project.config import (
    DEFAULT_PROVIDER,
    GOOGLE_API_KEY_FILE,
    OLLAMA_PRIMARY_MODEL,
    GEMINI_DEFAULT_MODEL
)
from rag_project.ingestion.ingest import IngestionPipeline
from rag_project.generation.rag_pipeline import RAGPipeline
from rag_project.generation.chat_session import ChatSession
from rag_project.test_suite import run_test_suite
from rag_project.deep_benchmark import run_deep_benchmark

console = Console()

def print_banner(provider: str, model: str):
    console.print(Panel(
        f"[bold white]Xilinx LogiCORE IP Distributed Memory Generator RAG Expert System[/bold white]\n"
        f"[dim cyan]Dual-Stage Hybrid Search (pgvector + tsvector) + Neural Cross-Encoder Reranker[/dim cyan]\n"
        f"[bold yellow]Active Provider:[/bold yellow] [bold white]{provider.upper()}[/bold white] | "
        f"[bold yellow]Model:[/bold yellow] [bold green]{model}[/bold green]",
        title="[bold cyan]XILINX RAG AI (Multi-Engine)[/bold cyan]",
        border_style="cyan"
    ))

def handle_query(pipeline: RAGPipeline, question: str, auto_k: bool = True):
    """Execute a single query and format results with citations and latency metrics."""
    console.print(f"\n[bold green]Processing Query:[/bold green] [white]{question}[/white]\n")
    with console.status("[bold green]Retrieving, Reranking & Generating Grounded Answer...[/bold green]"):
        result = pipeline.query(question, auto_k=auto_k)

    strat = result.get("adaptive_strategy", {})
    title_suffix = f" (Auto-K: {strat.get('selected_k', 5)} chunks)" if strat.get("auto_k_applied") else ""

    console.print(Panel(
        Markdown(result.get("answer", "")),
        title=f"[bold cyan]Engineering Response (Provider: {result.get('provider', 'MULTI').upper()} | Model: {result.get('model', result.get('model_used', 'gemini-3.5-flash-lite'))}){title_suffix}[/bold cyan]",
        border_style="cyan"
    ))


    # Sources table
    sources_table = Table(title="Retrieved & Reranked Context Sources", show_lines=True)
    sources_table.add_column("#", style="bold yellow", width=4)
    sources_table.add_column("Page", style="magenta", width=8)
    sources_table.add_column("Section / Breadcrumb", style="cyan", width=35)
    sources_table.add_column("Type", style="green", width=12)
    sources_table.add_column("Rerank Score", style="bold white", width=14)
    sources_table.add_column("Content Snippet", style="dim", width=45)

    for idx, s in enumerate(result["sources"], 1):
        boost_str = f" (+{s['adaptive_boost']} boost)" if s.get("adaptive_boost") else ""
        sources_table.add_row(
            str(idx),
            f"Page {s.get('page_number', '?')}",
            s.get("breadcrumb", "N/A"),
            s.get("content_type", "prose"),
            f"{s.get('rerank_score', 'N/A')}{boost_str}",
            s.get("content", "").replace("\n", " ")[:80] + "...",
        )

    console.print(sources_table)

    # Performance timings
    timings = result.get("timings_ms", {})
    t_ret = timings.get("hybrid_retrieval", timings.get("retrieval", 0))
    t_rerank = timings.get("reranking", timings.get("rerank", 0))
    t_gen = timings.get("generation", 0)
    t_tot = timings.get("total", 0)
    console.print(
        f"[dim][Timing] Pre-Inference Latency: {timings.get('pre_inference_total', 0)}ms "
        f"(Retrieval: {t_ret}ms | Rerank: {t_rerank}ms) | "
        f"LLM Generation: {t_gen}ms | "
        f"Total: {t_tot}ms[/dim]\n"
    )


def interactive_session_loop(session: ChatSession, auto_k: bool = True):
    """Run stateful multi-turn conversational session."""
    console.print(Panel(
        "[bold yellow]Entering Stateful Conversational Mode (Multi-Turn RAG)[/bold yellow]\n"
        "[dim]Contextual coreference & pronouns are automatically resolved across turns.\n"
        "Commands: 'reset' (clear history), 'export' (save markdown), 'exit' or 'q' to quit.[/dim]",
        border_style="yellow"
    ))

    while True:
        try:
            query = console.input(f"[bold cyan]Hardware Chat [Turn {len(session.history)+1}] > [/bold cyan]").strip()
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[bold green]Goodbye![/bold green]")
                break
            if query.lower() == "reset":
                session.reset()
                console.print("[bold green]Conversation history reset.[/bold green]")
                continue
            if query.lower() == "export":
                md_out = session.export_markdown()
                out_path = Path("xilinx_chat_export.md")
                out_path.write_text(md_out, encoding="utf-8")
                console.print(f"[bold green]Conversation exported to {out_path.resolve()}[/bold green]")
                continue
            if not query:
                continue

            with console.status("[bold green]Resolving coreference, retrieving & generating response...[/bold green]"):
                result = session.ask(query, auto_k=auto_k)

            # If query was reformulated, show the resolved standalone query
            if result.get("reformulated_query"):
                console.print(f"[dim blue]↳ Context-Aware Standalone Query: \"{result['reformulated_query']}\"[/dim blue]")

            strat = result.get("adaptive_strategy", {})
            title_suffix = f" [Auto-K: {strat.get('selected_k', 5)} chunks]" if strat.get("auto_k_applied") else ""

            console.print(Panel(
                Markdown(result["answer"]),
                title=f"[bold cyan]Engineering Response (Turn {len(session.history)}){title_suffix}[/bold cyan]",
                border_style="cyan"
            ))

            # Performance timings
            timings = result["timings_ms"]
            console.print(
                f"[dim][Timing] Ref: {timings.get('reformulation', 0)}ms | "
                f"Pre-Inf: {timings.get('pre_inference_total', 0)}ms | "
                f"Gen: {timings.get('generation', 0)}ms | "
                f"Total: {timings.get('total', 0)}ms[/dim]\n"
            )

        except KeyboardInterrupt:
            console.print("\n[bold green]Exiting conversational session...[/bold green]")
            break
        except Exception as e:
            console.print(f"[bold red]Error processing query: {e}[/bold red]")

def run_web_server(port: int = 8000, host: str = "127.0.0.1"):
    """Launch ChatGPT-style FastAPI web application server."""
    import uvicorn
    console.print(Panel(
        f"[bold green]Starting Xilinx Hardware RAG Web UI Server[/bold green]\n"
        f"[bold cyan]Local URL:[/bold cyan] [bold underline white]http://{host}:{port}[/bold underline white]\n"
        f"[dim]ChatGPT-style interface with full settings, multi-turn memory & source inspector.[/dim]",
        title="[bold cyan]RAG WEB STUDIO[/bold cyan]",
        border_style="green"
    ))
    uvicorn.run("rag_project.web.app:app", host=host, port=port, reload=False)

def main():
    parser = argparse.ArgumentParser(description="Xilinx LogiCORE IP RAG Engine")
    parser.add_argument("--web", "-w", action="store_true", help="Launch ChatGPT-style Web UI server in browser")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--provider", "-p", choices=["ollama", "gemini"], default=DEFAULT_PROVIDER, help="LLM Provider (ollama or gemini)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name (e.g. gemini-3.7-flash, gemini-3.5-flash-lite, gemma3:4b)")
    parser.add_argument("--api-key-file", type=str, default=str(GOOGLE_API_KEY_FILE), help="Path to Google AI Studio API key file")
    parser.add_argument("--fast", action="store_true", help="Enable ultra-fast 5ms neural reranker mode")
    parser.add_argument("--auto-k", action="store_true", default=True, help="Enable dynamic adaptive chunking (5 to 8-10 chunks)")
    parser.add_argument("--no-auto-k", dest="auto_k", action="store_false", help="Disable adaptive chunking and use static Top-K")
    parser.add_argument("--ingest", action="store_true", help="Run single document ingestion into PostgreSQL")
    parser.add_argument("--batch-ingest", action="store_true", help="Run batch multi-document ingestion across ./docs")
    parser.add_argument("--enrich", action="store_true", help="Use Gemini API to enrich chunks during ingestion")
    parser.add_argument("--test", action="store_true", help="Run automated benchmark test suite")
    parser.add_argument("--deep-benchmark", action="store_true", help="Run 8-node deep stress-test benchmark suite")
    parser.add_argument("--test-conversational", action="store_true", help="Run multi-turn conversational coreference test suite")
    parser.add_argument("--test-high-recall", "--test-benchmark", action="store_true", help="Run 5-node high-recall leaf preservation validation benchmark")
    parser.add_argument("--test-multi-doc", action="store_true", help="Run 3-category multi-document production benchmark suite")
    parser.add_argument("--eval-10", "--test-10", action="store_true", help="Run complete 10-question multi-document evaluation benchmark harness")
    parser.add_argument("--eval-15", "--test-15", action="store_true", help="Run 15-point comprehensive multi-document architectural benchmark harness")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive multi-turn CLI session")

    args = parser.parse_args()

    active_provider = args.provider
    active_model = args.model or (GEMINI_DEFAULT_MODEL if active_provider == "gemini" else OLLAMA_PRIMARY_MODEL)

    print_banner(active_provider, active_model)

    if args.web:
        run_web_server(port=args.port)
        return

    if args.batch_ingest:
        from rag_project.ingestion.batch_ingest import MultiDocBatchIngestor
        console.print("[bold green]Starting Batch Multi-Document Ingestion Pipeline...[/bold green]")
        ingestor = MultiDocBatchIngestor()
        res = ingestor.run()
        console.print(f"[bold green]Batch ingestion complete! Total stored chunks across {res['total_docs']} manuals: {res['total_chunks']}[/bold green]")
        return

    if args.ingest:
        console.print("[bold green]Starting Document Ingestion Pipeline...[/bold green]")
        pipeline = IngestionPipeline()
        total_chunks = pipeline.run(enrich_with_gemini=args.enrich)
        console.print(f"[bold green]Ingestion complete! Total stored chunks: {total_chunks}[/bold green]")
        return

    if args.test:
        run_test_suite()
        return

    if args.deep_benchmark:
        run_deep_benchmark()
        return

    if args.test_conversational:
        from tests.test_conversational_rag import test_conversational_benchmark
        test_conversational_benchmark()
        return

    if args.test_high_recall:
        from test_benchmark_suite import run_high_recall_benchmark
        run_high_recall_benchmark()
        return

    if args.test_multi_doc:
        from benchmark_multi_doc import run_multi_doc_benchmark
        run_multi_doc_benchmark()
        return

    if args.eval_10:
        from eval_benchmark_10 import run_10_question_benchmark
        run_10_question_benchmark()
        return

    if args.eval_15:
        from eval_benchmark_15 import run_15_point_benchmark
        run_15_point_benchmark()
        return





    rag_pipeline = RAGPipeline(
        provider=active_provider,
        model=active_model,
        api_key_file=Path(args.api_key_file),
        fast_rerank=args.fast,
    )

    if args.query:
        handle_query(rag_pipeline, args.query, auto_k=args.auto_k)
    else:
        chat_session = ChatSession(pipeline=rag_pipeline, auto_k_default=args.auto_k)
        interactive_session_loop(chat_session, auto_k=args.auto_k)

if __name__ == "__main__":
    main()
