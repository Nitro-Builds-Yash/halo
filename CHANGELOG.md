# Changelog

All notable changes to halo are documented here. Each version's section becomes its GitHub release notes.

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
