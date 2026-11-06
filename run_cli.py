import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph import graph
from src.core.config import settings

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]HALO[/bold cyan] — [bold white]Production Agentic CRAG Assistant[/bold white]\n"
        "[dim]Hybrid RRF (Dense + BM25) • CrossEncoder Reranker • Query Decomposition • Confidence & Citations[/dim]",
        border_style="cyan"
    ))

def display_pipeline_step(node_name: str, update: dict):
    step_icons = {
        "decompose_query": f"🧩 [blue]Step 1: Decomposing query into sub-questions: {update.get('sub_questions', [])}[/blue]",
        "retrieve": "🔍 [yellow]Step 2: Hybrid RRF Search (ChromaDB + BM25) & Neural Rerank...[/yellow]",
        "grade_documents": "⚖️ [magenta]Step 3: Auditing chunk relevance against sub-queries...[/magenta]",
        "transform_query": "🔄 [blue]Step 4: Query insufficient. Transforming for external web search...[/blue]",
        "web_search": "🌐 [cyan]Step 5: Executing fallback search via Tavily/Web...[/cyan]",
        "generate": "✍️ [green]Step 6: Synthesizing verified answer with confidence & citations...[/green]",
        "increment_retry": "⚠️ [red]Step: Reflection triggered retry due to quality gate check...[/red]"
    }
    msg = step_icons.get(node_name, f"⚡ Processing node: {node_name}")
    console.print(msg)

def ask_question(question: str):
    console.print(f"\n[bold yellow]User Question:[/bold yellow] {question}\n")
    inputs = {"question": question, "retry_count": 0}
    
    final_state = {}
    with console.status("[bold green]Executing agent graph...", spinner="dots"):
        for output in graph.stream(inputs):
            for node_name, state_update in output.items():
                display_pipeline_step(node_name, state_update)
                final_state.update(state_update)

    answer = final_state.get("generation", "No generation produced.")
    confidence = final_state.get("confidence", 0.0)
    citations = final_state.get("citations", [])
    documents = final_state.get("documents", [])
    web_used = final_state.get("web_search_needed", False)
    retries = final_state.get("retry_count", 0)

    # Display Answer
    console.print("\n" + "="*60)
    conf_color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.5 else "red"
    title_text = f"[bold green]Verified Answer[/bold green] [{conf_color}](Confidence: {int(confidence * 100)}%)[/{conf_color}]"
    console.print(Panel(Markdown(answer), title=title_text, border_style="green"))

    # Display Citations if any
    if citations:
        cit_table = Table(title="Extracted Source Citations", show_header=True, header_style="bold magenta")
        cit_table.add_column("Source", style="cyan")
        cit_table.add_column("Page", justify="center")
        cit_table.add_column("Evidence Snippet", style="dim")
        for c in citations:
            cit_table.add_row(str(c.get("source")), str(c.get("page") or "N/A"), str(c.get("snippet", "")))
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
    if not settings.OPENAI_API_KEY:
        console.print("[bold red]Warning:[/bold red] OPENAI_API_KEY is not set in environment or .env file.\n")
    
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
            console.print(f"[bold red]Execution error:[/bold red] {e}")

if __name__ == "__main__":
    main()
