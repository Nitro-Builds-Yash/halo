# Changelog

All notable changes to halo are documented here. Each version's section becomes its GitHub release notes.

## [2.6.0] - 2026-11-26

Production AI-engineering features.

#### Added
- **Prompt-injection guard** (`INJECTION_GUARD`): retrieved passages and web results are scanned for instructions aimed at the model (override, role-change, exfiltration, concealment, tool-hijack phrasing, chat-template markers, invisible characters). Matching lines are replaced before any LLM call, and every prompt marks passages as untrusted data. Answers include `security_flags`.
- **Usage and cost tracking**: per-question LLM calls, input/output tokens, estimated USD cost by model, fallback calls and latency, returned as `usage`. Built-in prices for common OpenAI models, overridable with `MODEL_PRICES`.
- **Spending limit** (`MAX_COST_PER_QUERY_USD`): stops before the next LLM call once a question's budget is spent.
- **`GET /metrics`**: Prometheus counters and a latency histogram covering requests by outcome, tokens, cost, cache hits, injection flags, budget stops and fallbacks.
- **Model routing** (`FAST_LLM_MODEL`): grading, planning, rewriting and condensing run on a cheaper model.
- **Model fallback** (`FALLBACK_LLM_MODEL`, `FALLBACK_BASE_URL`): failed calls are retried on a backup model, which can be local.
- **Agentic multi-hop search** (`AGENTIC_SEARCH`, `MAX_HOPS`): a planner runs follow-up searches for missing facts. Each hop is streamed and returned as `hops`.
- The web app, the `halo` CLI and `haloctl` show blocked instructions, follow-up searches and per-answer cost.

## [2.5.0] - 2026-11-26

A built-in web app, more Rust, and a cleaner layout.

#### Added
- **Web app** served by the API at `/` (no build step): chat with follow-ups, live pipeline steps, a confidence ring, sentence highlighting linked to evidence, clickable citations at the right PDF page, drag-and-drop uploads, document management, light and dark themes, and API-key settings.
- **More Rust in `halo-native` 0.2.0:** sentence-evidence scoring (`content_words`, `support_scores`, parallel) and Reciprocal Rank Fusion. Both produce the same output as Python, checked in CI.
- **`haloctl`:** a native Rust command-line client (`crates/halo-cli`) with `ask` (streams pipeline steps and evidence), `chat`, `upload`, `docs` and `health`. CI runs it against a real container.

#### Changed
- **Layout:** Rust lives in a `crates/` workspace (`halo-native`, `halo-cli`). Docker files moved to `deploy/`, and the benchmark to `evals/perf/`.
- **Restyled Streamlit app and `halo` CLI:** evidence per sentence, a confidence bar, and a cleaner header.
- **Docker:** run `docker compose -f deploy/docker-compose.yml up --build`. The Render blueprint points at `deploy/Dockerfile`.

## [2.4.0] - 2026-11-26

Native Rust keyword search.

#### Added
- **`halo-native`**: an optional Rust extension (PyO3) with Halo's BM25Plus index and tokenizer. Install it with `pip install ./rust` and Halo uses it automatically. Without it, Halo uses the pure-Python backend as before.
- The Rust index uses posting lists, so a query only touches documents that contain its terms. It tokenizes in parallel, releases the GIL during builds and searches, and gives the same scores and ranking as `rank-bm25` (checked in CI against both a reference port and `rank-bm25` itself, including Unicode tokenization).
- `HALO_NATIVE=0` turns the Rust backend off. `/health` and `halo --version` show which backend is active.
- `benchmarks/bm25_bench.py` compares the two backends. On a 20,000-document corpus in CI, the median query time fell from 26.3 ms to 0.43 ms (about 60×); index builds were about 1.2× faster. A new CI job runs `cargo fmt`, `clippy`, `cargo test`, the full test suite on the Rust backend, and the benchmark.

#### Install

```bash
pip install "halo @ git+https://github.com/Nitro-Builds-Yash/halo@v2.4.0"
pip install maturin && pip install "git+https://github.com/Nitro-Builds-Yash/halo@v2.4.0#subdirectory=rust"   # optional
```

## [2.3.0] - 2026-11-26

Sentence-level citations.

#### Added
- Every answer sentence now carries its own evidence: the passage that supports it, the exact supporting sentence, the source, the page and a link. The API returns these as `sentences`, along with an `unsupported_sentences` count.
- The generator labels each sentence with the passages it used. Halo re-checks every label against the passage text, corrects wrong labels, ignores invented ones, and flags sentences whose numbers or key terms don't appear in the sources.
- Unsupported sentences lower the answer's confidence: if every sentence is unsupported, it halves.
- The Streamlit app shows sentence evidence with ✅/⚠️ markers, and the CLI lists any unsupported sentences under the answer.

#### Install

```bash
pip install "halo @ git+https://github.com/Nitro-Builds-Yash/halo@v2.3.0"
```

## [2.2.0] - 2026-11-26

Follow-up questions, citation links, tracing and one-click deployment.

#### Added
- **Multi-turn chat:** follow-ups like "what about international orders?" are rewritten into a standalone question from the recent conversation, then retrieved, cached and verified as usual. The API takes an optional `history` and returns the `standalone_question`. The CLI and Streamlit app keep the conversation (`new` resets it in the CLI).
- **Citation links:** original uploads are stored, and each citation now carries a `url`: the web page, or `/api/v1/documents/{source}/file#page=N` for indexed documents. Removing a document removes its stored copy.
- **LangSmith tracing:** set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace every node and LLM call. Traces are tagged `api`, `stream`, `cli` or `ui`.
- **Render blueprint:** `render.yaml` and a Deploy to Render button. The Docker image now listens on `$PORT` when set.

#### Install

```bash
pip install "halo @ git+https://github.com/Nitro-Builds-Yash/halo@v2.2.0"
```

## [2.1.0] - 2026-11-25

Corrective RAG that answers questions about your PDFs, text and Markdown files, and **checks every answer against its sources** before returning it.

#### Install

```bash
pip install "halo @ git+https://github.com/Nitro-Builds-Yash/halo@v2.1.0"
# or, with the Streamlit UI:
pip install "halo[ui] @ git+https://github.com/Nitro-Builds-Yash/halo@v2.1.0"

halo ingest ./my-docs
halo
```

Or download the wheel attached to the release: `pip install halo-2.1.0-py3-none-any.whl`. Docker users can run `docker compose up --build` from the source archive.

#### Highlights
- **Verified answers:** every answer is checked for grounding and relevance, including answers built from web search or retries. Checks fail closed.
- **Hybrid search + relevance gate:** Chroma and BM25 fused with RRF, a per-chunk cross-encoder gate, and batched LLM grading for borderline chunks only.
- **Real citations:** source file and page (or URL) with a quote. Citations to sources the model never saw are removed.
- **Self-correction with limits:** a query rewrite and web search when local context is missing, at most one retry, then a clear fallback.
- **Three interfaces:** the `halo` CLI (`chat`, `ask`, `ingest`, `docs`, `serve`, `ui`), a Streamlit app with document management, and a FastAPI server with SSE streaming.
- **Runs locally:** OpenAI or any OpenAI-compatible server (Ollama, vLLM, LM Studio).
- **Production basics:** optional API key, sanitized and size-limited uploads, timeouts, Docker Compose with a Chroma server.

#### Fixed since the previous version
- Answers produced after a web search were always discarded for a fallback disclaimer.
- Cross-encoder score normalization was non-monotonic, so the relevance gate fired unpredictably.
- Path traversal in the upload endpoint.
- Re-uploading a file duplicated its chunks; PDF page numbers were off by one.
- The semantic cache served stale answers after new uploads, and its keys changed on every restart.
- Document and web text was rendered as raw HTML in the Streamlit app.

#### Breaking changes
- The code is now the `halo` package under `src/halo/`, imported as `halo.*` instead of `src.*`.
- `run_cli.py` and `streamlit_app.py` are replaced by `halo` and `halo ui`. `uvicorn main:app` still works.
- `MAX_RETRIES` is replaced by `MAX_REWRITES` and `MAX_REGENERATIONS`; see `.env.example`.
- The previous README benchmark table was removed: it came from a simulated script. `evals/run_eval.py` now runs a real RAGAS comparison on your own documents.

#### Quality
55 tests (LLMs and search mocked) and lint run on every pull request; `main` is protected and requires them to pass.

## [1.0.0]

Initial release: LangGraph CRAG pipeline with hybrid retrieval, a FastAPI server, a Streamlit UI and a CLI.
