<p align="center">
  <img src="./docs/assets/hero.svg" width="100%" alt="halo: ask anything, retrieve truth. Answers from your documents, checked and cited.">
</p>

<p align="center">
  <a href="https://github.com/Nitro-Builds-Yash/halo/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Nitro-Builds-Yash/halo/ci.yml?branch=main&style=for-the-badge&label=CI&logo=githubactions&logoColor=white"></a>
  <a href="https://github.com/Nitro-Builds-Yash/halo/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/Nitro-Builds-Yash/halo?style=for-the-badge&label=release&color=238636"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-CRAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <a href="./LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-8B949E?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="#-quickstart"><b>Quickstart</b></a> ·
  <a href="#-how-it-works"><b>How it works</b></a> ·
  <a href="#%EF%B8%8F-usage"><b>Usage</b></a> ·
  <a href="#-api"><b>API</b></a> ·
  <a href="#%EF%B8%8F-configuration"><b>Configuration</b></a> ·
  <a href="#-evaluation"><b>Evaluation</b></a> ·
  <a href="#%EF%B8%8F-project-structure"><b>Structure</b></a>
</p>

<br>

**Halo** answers questions about your PDFs, text and Markdown files, and **checks every answer against its sources before returning it**. It finds evidence with hybrid search, filters it with a cross-encoder, searches the web when your documents fall short, and cites where each fact came from. If it still can't verify an answer, it says so and shows you the best evidence instead of guessing.

<br>

## ✨ Features

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>🔎 Hybrid search</h3>
      Dense vectors (Chroma) and BM25 keyword search, fused with Reciprocal Rank Fusion on shared chunk IDs. It finds both paraphrases and exact codes like <code>§4.1</code>.
    </td>
    <td width="33%" valign="top">
      <h3>⚖️ Relevance gate</h3>
      A cross-encoder scores every chunk. Confident chunks pass, weak ones are dropped, and only the uncertain middle goes to the LLM, in a single batched call.
    </td>
    <td width="33%" valign="top">
      <h3>✅ Verified answers</h3>
      Every answer is checked for grounding and for whether it actually answers the question, including answers built from web results or retries.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>📎 Sentence-level citations</h3>
      Every sentence of an answer points to the exact passage that supports it, with a link to the original page. Sentences no source supports are flagged and lower the confidence score.
    </td>
    <td valign="top">
      <h3>🔁 Self-correction</h3>
      When local context is missing or an answer fails its check, Halo rewrites the query and searches the web, with hard limits so it never loops.
    </td>
    <td valign="top">
      <h3>💬 Follow-up questions</h3>
      Ask "what about international orders?" and Halo rewrites it into a standalone question from the conversation, then searches, caches and verifies that. Verified answers are cached and the cache clears when documents change.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>🧰 Three ways in</h3>
      A <code>halo</code> CLI, a Streamlit app with document management, and a FastAPI server with streaming (SSE).
    </td>
    <td valign="top">
      <h3>🏠 Runs locally</h3>
      Works with OpenAI or any OpenAI-compatible server (Ollama, vLLM, LM Studio), so you can keep everything on your machine.
    </td>
    <td valign="top">
      <h3>🛡️ Production basics</h3>
      Optional API key, upload limits, timeouts, LangSmith tracing of every step, Docker Compose, one-click Render deploy, and a tested CI pipeline.
    </td>
  </tr>
</table>

## ⚡ Quickstart

```bash
git clone https://github.com/Nitro-Builds-Yash/halo && cd halo
pip install -e ".[ui]"            # add ,rerank for the cross-encoder (installs torch)
cp .env.example .env              # then set OPENAI_API_KEY

halo ingest ./my-docs             # index .pdf / .txt / .md files
halo                              # start chatting
```

```text
halo › What is the refund window?
🔍 Hybrid retrieval  Chroma + BM25 → RRF → cross-encoder
⚖️  Relevance gate   per-chunk scores + batched LLM grading
✍️  Answer synthesis citations + confidence
🧪 Verification: useful
╭─────────────── Verified answer  92% confidence ───────────────╮
│ Refunds are accepted within 30 days of purchase with the      │
│ original receipt.                                             │
╰───────────────────────────────────────────────────────────────╯
 Source        Page  Evidence
 policy.pdf    4     "returns are permitted within 30 days…"
```

<details>
<summary><b>🐳 Run with Docker</b></summary>
<br>

```bash
cp .env.example .env              # add your keys
docker compose up --build
```

| Service | URL |
| --- | --- |
| API + interactive docs | http://localhost:8000/docs |
| Streamlit app | http://localhost:8501 |

The API and app share one Chroma server, so documents uploaded in either show up in both. Build a smaller image without torch with `docker compose build --build-arg RERANK=0`.
</details>

<details>
<summary><b>☁️ Deploy to Render</b></summary>
<br>

<a href="https://render.com/deploy?repo=https://github.com/Nitro-Builds-Yash/halo"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>

The [`render.yaml`](./render.yaml) blueprint deploys the API with a 1 GB disk for the index. Render asks for your `OPENAI_API_KEY` (and optionally `TAVILY_API_KEY` and `LANGSMITH_API_KEY`) and generates an `API_KEY` for clients. It uses the Starter plan, because a persistent disk needs a paid instance; the image is built without torch to fit in 512 MB.
</details>

<details>
<summary><b>🔭 Trace every step with LangSmith</b></summary>
<br>

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=halo
```

Each question becomes one trace named `halo`, tagged by where it came from (`api`, `stream`, `cli`, `ui`), with every graph node, LLM call, latency and token count.
</details>

<details>
<summary><b>🏠 Run fully local with Ollama (no OpenAI)</b></summary>
<br>

```bash
ollama pull llama3.1 && ollama pull nomic-embed-text
```

```dotenv
# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text
```

Small local models are less reliable at structured output, so expect more fallback answers than with GPT-4o-mini. Re-ingest your documents after changing the embedding model.
</details>

## 🧠 How it works

<p align="center">
  <img src="./docs/assets/pipeline.svg" width="100%" alt="Pipeline: question, semantic cache, query plan, hybrid search, relevance gate, generate, verify, answer. If there is no relevant context, the query is rewritten and the web is searched. If an answer is not verified it is retried once, then falls back to cited excerpts.">
</p>

1. **Semantic cache:** a question that closely matches one already answered and verified (cosine similarity ≥ 0.92) is returned immediately.
2. **Query plan:** comparative or multi-part questions are split into up to three focused sub-questions.
3. **Hybrid search:** each sub-question searches Chroma and BM25. Results are fused with RRF and reranked by a cross-encoder.
4. **Relevance gate:** per chunk, a score ≥ 0.85 is kept, < 0.40 is dropped, and anything in between is graded by the LLM in one batched call.
5. **Generate:** the answer is written strictly from the kept context, with citations and a confidence score.
6. **Sentence evidence:** each sentence is matched to the passage that supports it. The model's own labels are re-checked against the passage text, so a wrong number ("60 days" when the source says 30) is flagged as unsupported.
7. **Verify:** two checks, whether every claim is supported by the sources and whether the answer resolves the question. If a checker errors, the answer counts as **not** verified.

When there's no relevant local context, or an answer fails verification, Halo rewrites the query and adds web results (at most one rewrite and one retry). If it still can't verify an answer, it returns the best excerpts with a clear disclaimer.

<details>
<summary>View the graph as Mermaid</summary>
<br>

```mermaid
flowchart LR
    Q([Question]) --> C{Cache hit?}
    C -- yes --> A([Verified answer])
    C -- no --> P[Query plan] --> R[Hybrid search<br/>Chroma + BM25 → RRF → rerank] --> G{Relevance gate}
    G -- relevant context --> Gen[Generate]
    G -- none --> W[Rewrite query] --> S[Web search] --> Gen
    Gen --> V{Verify}
    V -- verified --> A
    V -- not verified, retry left --> W
    V -- limits reached --> F([Fallback: excerpts + disclaimer])
```
</details>

## 🖥️ Usage

### Command line

| Command | What it does |
| --- | --- |
| `halo` | Interactive chat with live pipeline steps and follow-up questions (`new` resets the conversation) |
| `halo ask "question"` | Answer one question and exit |
| `halo ingest ./docs a.pdf` | Index files and folders (`.pdf`, `.txt`, `.md`). Re-ingesting a file replaces it. |
| `halo docs` / `halo docs rm a.pdf` | List indexed documents / remove one |
| `halo serve [--reload]` | Start the API on port 8000 |
| `halo ui` | Start the Streamlit app on port 8501 |

### Web app

`halo ui` opens a chat interface where you can upload and delete documents, watch each pipeline step as it runs, and expand the citations, confidence and agent trace for every answer.

## 🔌 API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status and capabilities |
| `POST` | `/api/v1/query` | Answer a question (semantic cache first) |
| `POST` | `/api/v1/chat/stream` | Server-Sent Events: one status event per step, then the final verified answer |
| `POST` | `/api/v1/documents/upload` | Upload `.pdf` / `.txt` / `.md` files (size-limited) |
| `GET` | `/api/v1/documents` | List indexed documents |
| `GET` | `/api/v1/documents/{source}/file` | The original file; citation `url`s point here (with `#page=N` for PDFs) |
| `DELETE` | `/api/v1/documents/{source}` | Remove a document from both indexes |

```bash
curl -s localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What about international orders?",
       "history": [{"role": "user", "content": "What is the refund window?"},
                   {"role": "assistant", "content": "30 days from purchase."}]}'
```

`history` is optional: send earlier messages (oldest first) to ask follow-ups. The response includes the `standalone_question` Halo actually answered.

<details>
<summary>Example response</summary>

```json
{
  "question": "What about international orders?",
  "standalone_question": "What is the refund window for international orders?",
  "generation": "International orders can be returned within 45 days of delivery.",
  "confidence": 0.9,
  "citations": [{ "source": "policy.pdf", "page": 5, "snippet": "international orders: 45 days from delivery",
                  "url": "/api/v1/documents/policy.pdf/file#page=5" }],
  "sentences": [{ "text": "International orders can be returned within 45 days of delivery.",
                  "supported": true, "score": 1.0,
                  "sources": [{ "source": "policy.pdf", "page": 5, "chunk": 2,
                                "snippet": "International orders can be returned within 45 days of delivery.",
                                "url": "/api/v1/documents/policy.pdf/file#page=5" }] }],
  "unsupported_sentences": 0,
  "verdict": "useful",
  "retry_count": 0,
  "rewrite_count": 0,
  "web_search_used": false,
  "is_fallback": false,
  "cached": false
}
```
</details>

If `API_KEY` is set, every `/api/v1/*` request needs an `X-API-Key` header (or `?api_key=`, so citation links open in a browser). Interactive docs are at `/docs`.

## ⚙️ Configuration

All settings live in `.env`; see [`.env.example`](./.env.example) for the full list. The ones you're most likely to change:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | – | OpenAI key (not needed with `LLM_BASE_URL`) |
| `LLM_MODEL` / `EMBEDDING_MODEL` | `gpt-4o-mini` / `text-embedding-3-small` | Models for answers and embeddings |
| `LLM_BASE_URL` | – | Any OpenAI-compatible server, e.g. Ollama |
| `TAVILY_API_KEY` | – | Web search via Tavily (falls back to DuckDuckGo) |
| `TOP_K_RETRIEVE` | `4` | Chunks retrieved per sub-question |
| `GATE_HIGH` / `GATE_LOW` | `0.85` / `0.40` | Relevance-gate thresholds |
| `MAX_REWRITES` / `MAX_REGENERATIONS` | `1` / `1` | Self-correction limits |
| `CACHE_SIMILARITY` | `0.92` | How similar a question must be to reuse an answer |
| `API_KEY` | – | Require `X-API-Key` on the API |
| `CHROMA_HOST` | – | Use a Chroma server instead of local files |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `false` / – | Trace every step in LangSmith |

## 📊 Evaluation

Halo includes a real evaluation harness. It runs a **naive RAG baseline** (dense top-k + one generation) and the **full Halo pipeline** on the same questions over your documents, then scores both with [RAGAS](https://docs.ragas.io): faithfulness, response relevancy, context precision and context recall. It also reports fallback rate, web-search rate and latency.

```bash
pip install -e ".[eval]"
halo ingest ./my-docs
cp evals/dataset.example.jsonl evals/dataset.jsonl     # write 30–50 Q&A pairs about your documents
python -m evals.run_eval --dataset evals/dataset.jsonl # results → evals/results/*.json
```

> [!NOTE]
> Benchmark results will be published here once they've been run on a real corpus.

## 🗂️ Project structure

```text
halo/
├── src/halo/
│   ├── agents/          # LangGraph pipeline
│   │   ├── graph.py     #   graph wiring and routing
│   │   ├── nodes/       #   decompose, retrieve, grade, rewrite, web search, generate, reflect, fallback
│   │   ├── runner.py    #   run a question with the semantic cache in front
│   │   └── state.py     #   shared graph state
│   ├── rag/             # ingestion, Chroma + BM25 indexes, hybrid retriever, cross-encoder
│   ├── api/             # FastAPI server and schemas
│   ├── core/            # settings, model clients, semantic cache, logging
│   ├── tools/           # web search
│   ├── ui/app.py        # Streamlit app
│   └── cli.py           # the `halo` command
├── evals/               # RAGAS evaluation harness
├── tests/               # offline test suite (LLMs and search mocked)
├── docs/                # images and interview notes
├── Dockerfile · docker-compose.yml
└── pyproject.toml
```

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest            # no API keys needed: every LLM and search call is mocked
ruff check .
```

CI runs lint and the full test suite on every push and pull request. `main` is protected, so changes go in through pull requests with passing checks.

## 📦 Releases

Each release is listed on the [Releases page](https://github.com/Nitro-Builds-Yash/halo/releases) with notes from [`CHANGELOG.md`](./CHANGELOG.md) and a wheel you can `pip install`. To publish a new one, bump `__version__` in `src/halo/__init__.py`, add a section to `CHANGELOG.md`, merge, then run **Actions → Release → Run workflow**.

## 🗺️ Roadmap

- [x] Corrective RAG graph with verification on every answer
- [x] Hybrid search, per-chunk relevance gate, batched grading
- [x] Document management, Docker, local-LLM support
- [x] Real RAGAS evaluation harness
- [ ] Publish benchmark results on a public corpus
- [x] Multi-turn chat (follow-ups rewritten into standalone questions)
- [x] Citations that link to the original document page
- [x] LangSmith tracing and one-click Render deploy
- [x] Sentence-level citations with unsupported sentences flagged
- [ ] Table extraction and OCR for scanned PDFs

## 📄 License

Released under the [MIT License](./LICENSE).

<p align="center">
  <img src="./docs/assets/status.svg" width="100%" alt="halo: hybrid search, relevance gate, every answer checked against its sources">
</p>
