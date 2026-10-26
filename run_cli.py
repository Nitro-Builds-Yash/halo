"""
Interactive CLI for Halo Corrective RAG (CRAG) QA System
"""
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph import graph
from src.core.config import settings

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]HALO[/bold cyan] — [bold white]Agentic Corrective RAG (CRAG) Assistant[/bold white]\n"
        "[dim]Self-corrective retrieval, relevance grading & hallucination auditing[/dim]",
        border_style="cyan"
    ))

def display_pipeline_step(node_name: str):
    step_icons = {
        "retrieve": "🔍 [yellow]Step 1: Retrieving relevant documents from vector store...[/yellow]",
        "grade_documents": "⚖️ [magenta]Step 2: Grading retrieved documents for relevance...[/magenta]",
        "transform_query": "🔄 [blue]Step 3: Documents insufficient. Transforming query for web search...[/blue]",
        "web_search": "🌐 [cyan]Step 4: Executing fallback search via Tavily/Web...[/cyan]",
        "generate": "✍️ [green]Step 5: Synthesizing verified answer...[/green]",
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
                display_pipeline_step(node_name)
                final_state.update(state_update)

    answer = final_state.get("generation", "No generation produced.")
    documents = final_state.get("documents", [])
    web_used = final_state.get("web_search_needed", False)
    retries = final_state.get("retry_count", 0)

    # Display Answer
    console.print("\n" + "="*60)
    console.print(Panel(Markdown(answer), title="[bold green]Verified Answer[/bold green]", border_style="green"))

    # Display Audit Summary Table
    table = Table(title="Halo Quality & Audit Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Result")
    
    table.add_row("Vector Chunks Retained", str(len(documents)))
    table.add_row("Fallback Web Search Used", "[red]Yes[/red]" if web_used else "[green]No (Local Context Only)[/green]")
    table.add_row("Reflection Retries Triggered", str(retries))
    table.add_row("Active Model", settings.LLM_MODEL)
    
    console.print(table)
    console.print("="*60 + "\n")

def main():
    print_banner()
    if not settings.OPENAI_API_KEY:
        console.print("[bold red]Warning:[/bold red] OPENAI_API_KEY is not set in environment or .env file.")
        console.print("Please configure your .env file before running queries.\n")
    
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
