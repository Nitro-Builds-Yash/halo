"""The `halo` command.

    halo                      interactive chat (same as `halo chat`)
    halo ask "question"       answer one question and exit
    halo ingest ./data a.pdf  index files or folders (.pdf, .txt, .md)
    halo docs                 list indexed documents
    halo docs rm policy.pdf   remove a document from the index
    halo serve                start the FastAPI server
    halo ui                   start the Streamlit app
"""
import argparse
import os
import subprocess
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from halo import __version__
from halo.core.config import settings

console = Console()

STEPS = {
    "retrieve": "🔍 [yellow]Hybrid retrieval[/yellow] [dim]Chroma + BM25 → RRF → cross-encoder[/dim]",
    "grade_documents": "⚖️  [magenta]Relevance gate[/magenta] [dim]per-chunk scores + batched LLM grading[/dim]",
    "transform_query": "🔄 [blue]Query rewrite[/blue] [dim]local context insufficient[/dim]",
    "web_search": "🌐 [cyan]Web search[/cyan]",
    "generate": "✍️  [green]Answer synthesis[/green] [dim]citations + confidence[/dim]",
    "increment_retry": "⚠️  [red]Not verified[/red] [dim]retrying with web context[/dim]",
    "graceful_fallback": "🛟 [yellow]Fallback[/yellow] [dim]best excerpts with a disclaimer[/dim]",
}


# ---------------------------------------------------------------- chat / ask

def _print_step(node: str, update: dict):
    if node == "decompose_query":
        subs = update.get("sub_questions", [])
        console.print(f"🧩 [blue]Query plan[/blue] [dim]{escape('; '.join(subs))}[/dim]")
    elif node == "reflect":
        verdict = update.get("verdict", "")
        color = "green" if verdict == "useful" else "yellow"
        console.print(f"🧪 [{color}]Verification: {escape(verdict)}[/{color}]")
    elif node in STEPS:
        console.print(STEPS[node])


def _run(question: str, history: list | None = None) -> dict:
    from halo.agents.conversation import condense_question
    from halo.agents.graph import graph
    from halo.agents.runner import cache_payload, cached_answer, should_cache, summarize
    from halo.agents.state import initial_state
    from halo.core.semantic_cache import SemanticCache
    from halo.core.tracing import run_config

    standalone = condense_question(question, history)
    if standalone != question:
        console.print(f"💬 [blue]Follow-up understood as[/blue] [dim]{escape(standalone)}[/dim]")
    cache = SemanticCache.get_instance()
    hit = cached_answer(question, cache, standalone)
    if hit:
        console.print(f"⚡ [green]Semantic cache hit[/green] [dim](similarity {hit['cached_similarity']})[/dim]")
        return hit

    state = initial_state(standalone)
    with console.status("[bold green]Thinking...", spinner="dots"):
        for output in graph.stream(state, config=run_config("cli")):
            for node, update in output.items():
                _print_step(node, update or {})
                state.update(update or {})
    result = summarize(question, state, standalone)
    if should_cache(result):
        cache.store(standalone, cache_payload(state))
    return result


def _print_result(result: dict):
    confidence = result["confidence"]
    color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.5 else "red"
    label = "Partial answer (unverified)" if result["is_fallback"] else "Verified answer"
    console.print()
    console.print(Panel(
        Markdown(result["generation"] or "No answer produced."),
        title=f"[bold]{label}[/bold] [{color}]{int(confidence * 100)}% confidence[/{color}]",
        border_style="yellow" if result["is_fallback"] else "green",
    ))

    if result["citations"]:
        table = Table(title="Sources", header_style="bold magenta", show_lines=False)
        table.add_column("Source", style="cyan")
        table.add_column("Page", justify="center")
        table.add_column("Evidence", style="dim")
        for c in result["citations"]:
            table.add_row(escape(str(c.get("source"))), str(c.get("page") or "–"), escape(str(c.get("snippet", ""))))
        console.print(table)

    unsupported = [s for s in result.get("sentences", []) if s.get("supported") is False]
    if unsupported:
        console.print(f"[yellow]⚠ {len(unsupported)} sentence(s) not found in the sources:[/yellow]")
        for s in unsupported:
            console.print(f"  [yellow]•[/yellow] [dim]{escape(s['text'])}[/dim]")

    meta = [
        f"chunks {len(result['documents'])}",
        f"web search {'yes' if result['web_search_used'] else 'no'}",
        f"retries {result['retry_count']}",
        "cached" if result.get("cached") else settings.LLM_MODEL,
    ]
    console.print(f"[dim]{' · '.join(meta)}[/dim]\n")


def _ask(question: str, history: list | None = None) -> dict | None:
    try:
        result = _run(question, history)
    except Exception as e:  # keep the chat loop alive
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
        return None
    _print_result(result)
    return result


def _warn_if_unconfigured():
    if not settings.llm_configured:
        console.print("[bold red]Warning:[/bold red] set OPENAI_API_KEY (or LLM_BASE_URL for a local model) in .env\n")


def cmd_chat(_args):
    console.print(Panel.fit(
        f"[bold cyan]halo[/bold cyan] [dim]v{__version__}[/dim]  ask anything. retrieve truth.\n"
        "[dim]Hybrid retrieval · cross-encoder gate · verified answers with citations[/dim]\n"
        "[dim]Ask follow-ups naturally. 'new' starts a fresh conversation, 'exit' quits.[/dim]",
        border_style="cyan",
    ))
    _warn_if_unconfigured()
    history: list = []
    while True:
        try:
            question = console.input("[bold cyan]halo ›[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            return
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye.[/dim]")
            return
        if question.lower() in {"new", "reset", "clear"}:
            history = []
            console.print("[dim]Started a new conversation.[/dim]")
            continue
        if question:
            result = _ask(question, history)
            if result:
                history += [{"role": "user", "content": question},
                            {"role": "assistant", "content": result["generation"]}]


def cmd_ask(args):
    _warn_if_unconfigured()
    _ask(" ".join(args.question))


# ---------------------------------------------------------------- documents

def cmd_ingest(args):
    from halo.core.semantic_cache import SemanticCache
    from halo.rag.ingest import SUPPORTED_EXTENSIONS, ingest_file_list

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, n) for n in names if n.lower().endswith(SUPPORTED_EXTENSIONS)]
        elif os.path.isfile(path):
            files.append(path)
        else:
            console.print(f"[red]Not found:[/red] {escape(path)}")
    if not files:
        console.print(f"[yellow]No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) to ingest.[/yellow]")
        sys.exit(1)

    with console.status(f"Indexing {len(files)} file(s)...", spinner="dots"):
        n_docs, n_chunks = ingest_file_list(files)
    SemanticCache.get_instance().clear()
    console.print(f"✅ Indexed [bold]{n_docs}[/bold] page(s) into [bold]{n_chunks}[/bold] chunks.")


def cmd_docs(args):
    from halo.rag.ingest import delete_source, list_sources

    if args.action == "rm":
        if not args.name:
            console.print("[red]Usage:[/red] halo docs rm <filename>")
            sys.exit(2)
        from halo.core.semantic_cache import SemanticCache

        removed = delete_source(args.name)
        if not removed:
            console.print(f"[yellow]No indexed document named {escape(args.name)!r}[/yellow]")
            sys.exit(1)
        SemanticCache.get_instance().clear()
        console.print(f"🗑️  Removed {escape(args.name)} ({removed} chunks)")
        return

    docs = list_sources()
    if not docs:
        console.print("[dim]No documents indexed yet. Try: halo ingest ./data[/dim]")
        return
    table = Table(title="Indexed documents", header_style="bold cyan")
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Pages", justify="right")
    for d in docs:
        table.add_row(escape(d["source"]), str(d["chunks"]), str(d["pages"]))
    console.print(table)


# ---------------------------------------------------------------- servers

def cmd_serve(args):
    import uvicorn

    uvicorn.run("halo.api.server:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ui(args):
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(args.port)]
    sys.exit(subprocess.call(cmd))


# ---------------------------------------------------------------- entry point

def _backend() -> str:
    from halo.rag.bm25 import backend_name

    return backend_name()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="halo", description="Corrective RAG with verified, cited answers.")
    parser.add_argument("--version", action="version", version=f"halo {__version__} · keyword search: {_backend()}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive chat (default)").set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", help="answer one question and exit")
    p.add_argument("question", nargs="+")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ingest", help="index files or folders (.pdf, .txt, .md)")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("docs", help="list indexed documents, or remove one with `docs rm NAME`")
    p.add_argument("action", nargs="?", choices=["list", "rm"], default="list")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("serve", help="start the FastAPI server")
    p.add_argument("--host", default=settings.HOST)
    p.add_argument("--port", type=int, default=settings.PORT)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ui", help="start the Streamlit app")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    (getattr(args, "func", None) or cmd_chat)(args)


if __name__ == "__main__":
    main()
