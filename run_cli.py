import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph import graph
from src.agents.runner import cache_payload, cached_answer, should_cache, summarize
from src.agents.state import initial_state
from src.core.config import settings
from src.core.semantic_cache import SemanticCache

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]HALO[/bold cyan] — [bold white]Production Agentic CRAG Assistant[/bold white]\n"
        "[dim]Hybrid RRF (Dense + BM25) • CrossEncoder Reranker • Query Decomposition • Confidence & Citations[/dim]",
        border_style="cyan"
    ))

def display_pipeline_step(node_name: str, update: dict):
    step_icons = {
        "decompose_query": f"🧩 [blue]Step 1: Decomposing query into sub-questions: {escape(str(update.get('sub_questions', [])))}[/blue]",
        "retrieve": "🔍 [yellow]Step 2: Hybrid RRF Search (ChromaDB + BM25) & Neural Rerank...[/yellow]",
        "grade_documents": "⚖️ [magenta]Step 3: Auditing chunk relevance against sub-queries...[/magenta]",
        "transform_query": "🔄 [blue]Step 4: Query insufficient. Transforming for external web search...[/blue]",
        "web_search": "🌐 [cyan]Step 5: Executing fallback search via Tavily/Web...[/cyan]",
        "generate": "✍️ [green]Step 6: Synthesizing verified answer with confidence & citations...[/green]",
        "reflect": f"🧪 [white]Verification: {escape(str(update.get('verdict', '')))}[/white]",
        "increment_retry": "⚠️ [red]Answer not verified — retrying with web context...[/red]",
        "graceful_fallback": "🛟 [yellow]Fallback: returning best excerpts with a disclaimer[/yellow]",
    }
    msg = step_icons.get(node_name, f"⚡ Processing node: {node_name}")
    console.print(msg)

def ask_question(question: str):
    console.print(f"\n[bold yellow]User Question:[/bold yellow] {escape(question)}\n")
    cache = SemanticCache.get_instance()
    hit = cached_answer(question, cache)
    if hit:
        console.print(f"⚡ [green]Semantic cache hit (similarity {hit['cached_similarity']})[/green]")
        result = hit
    else:
        final_state = {}
        with console.status("[bold green]Executing agent graph...", spinner="dots"):
            for output in graph.stream(initial_state(question)):
                for node_name, state_update in output.items():
                    display_pipeline_step(node_name, state_update or {})
                    final_state.update(state_update or {})
        result = summarize(question, final_state)
        if should_cache(result):
            cache.store(question, cache_payload(final_state))

    answer = result["generation"] or "No generation produced."
    confidence = result["confidence"]
    citations = result["citations"]
    documents = result["documents"]
    web_used = result["web_search_used"]
    retries = result["retry_count"]

    # Display Answer
    console.print("\n" + "="*60)
    conf_color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.5 else "red"
    label = "Partial Answer (unverified)" if result["is_fallback"] else "Verified Answer"
    title_text = f"[bold green]{label}[/bold green] [{conf_color}](Confidence: {int(confidence * 100)}%)[/{conf_color}]"
    console.print(Panel(Markdown(answer), title=title_text, border_style="green"))

    # Display Citations if any
    if citations:
        cit_table = Table(title="Extracted Source Citations", show_header=True, header_style="bold magenta")
        cit_table.add_column("Source", style="cyan")
        cit_table.add_column("Page", justify="center")
        cit_table.add_column("Evidence Snippet", style="dim")
        for c in citations:
            cit_table.add_row(escape(str(c.get("source"))), str(c.get("page") or "N/A"), escape(str(c.get("snippet", ""))))
        console.print(cit_table)

    # Display Audit Summary Table
    table = Table(title="Halo Quality & Audit Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Result")
    
    table.add_row("Calibrated Confidence", f"[{conf_color}]{int(confidence * 100)}%[/{conf_color}]")
    table.add_row("Hybrid Chunks Retained", str(len(documents)))
    table.add_row("Fallback Web Search Used", "[red]Yes[/red]" if web_used else "[green]No (Local Corpus Only)[/green]")
    table.add_row("Reflection Retries Triggered", str(retries))
    table.add_row("Active Model", settings.LLM_MODEL)
    
    console.print(table)
    console.print("="*60 + "\n")

def main():
    print_banner()
    if not settings.llm_configured:
        console.print("[bold red]Warning:[/bold red] set OPENAI_API_KEY (or LLM_BASE_URL for a local model) in .env.\n")
    
    while True:
        try:
            query = console.input("[bold cyan]halo > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Exiting Halo. Goodbye![/dim]")
                break
            ask_question(query)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {escape(str(e))}")

if __name__ == "__main__":
    main()
