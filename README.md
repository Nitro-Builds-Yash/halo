<p align="center">
  <img src="./assets/hero.svg" width="100%" alt="halo — agentic RAG QA system">
</p>

<p align="center">
  [![Download halo](https://img.shields.io/badge/DOWNLOAD-v1.0-0969da?style=for-the-badge&logo=python&logoColor=white)](https://github.com/Nitro-Builds-Yash/halo/releases/latest)
  [![Latest release](https://img.shields.io/github/v/release/Nitro-Builds-Yash/halo?display_name=tag&style=for-the-badge&label=RELEASE&color=238636)](https://github.com/Nitro-Builds-Yash/halo/releases/latest)
  ![Python 3.7+](https://img.shields.io/badge/PYTHON-3.7%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
  [![MIT license](https://img.shields.io/badge/LICENSE-MIT-8b949e?style=for-the-badge)](./LICENSE)
</p>

<p align="center">
  <strong>Ask anything. Retrieve truth.</strong><br>
  An agentic RAG pipeline with 4 autonomous quality gates — it retrieves, evaluates, audits for hallucinations, and grades answers before they reach you.
</p>

---

## The pipeline that questions itself

<p align="center">
  <img src="./assets/diagram.svg" width="100%" alt="halo agent pipeline">
</p>

halo routes every question through four autonomous agents. The Query Analyzer classifies intent and expands the query. The Hybrid Retriever combines BM25 keyword scoring and dense vector similarity using Reciprocal Rank Fusion. The Relevance Evaluator scores context quality and triggers a self-corrective re-query loop when scores fall below threshold. The Faithfulness Auditor cross-verifies every claim against retrieved evidence before any answer is released.

## Answers should not be guesses

Questions deserve verified answers. Most RAG systems generate text and hope it is correct. halo does not. Every piece of retrieved context passes a relevance threshold before the language model sees it. Every generated claim is cross-checked against the source chunks it came from. If the answer does not meet quality standards, the pipeline retries automatically.

<table>
  <tr>
    <td><strong>Self-corrective</strong><br>Low-relevance context triggers automatic re-query with expanded terms — no silent degradation.</td>
    <td><strong>Hallucination-free</strong><br>Every statement is verified against retrieved source chunks before delivery, with citations attached.</td>
  </tr>
  <tr>
    <td><strong>Hybrid retrieval</strong><br>BM25 Okapi keyword scoring fused with dense semantic vectors via Reciprocal Rank Fusion.</td>
    <td><strong>Multi-provider LLM</strong><br>Drop in Gemini, OpenAI, Claude, Ollama, or the built-in offline mock — one config line.</td>
  </tr>
</table>

## Install

1. `git clone https://github.com/Nitro-Builds-Yash/halo`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set your API key
4. `python run_cli.py`

> [!NOTE]
> Set `LLM_PROVIDER=mock` in `.env` to run fully offline with the built-in mock provider.

## Run

| Command | What it does |
| --- | --- |
| `python run_cli.py` | Interactive CLI with quality scorecard display |
| `uvicorn rag_qa_system.api.server:app --reload` | FastAPI REST server on localhost:8000 |
| `POST /api/query` | Submit a question; returns answer + full quality report |
| `POST /api/index` | Index a document folder into the retrieval store |
| `python -m pytest tests/` | Run the full test suite |

## How it works

halo builds a dual-index store from your documents — a BM25 keyword index and a dense embedding vector store. On each query, both stores are searched and results merged using Reciprocal Rank Fusion. Four quality agents evaluate, audit, and grade the retrieved context and generated answer in sequence. If any gate fails, the pipeline automatically retries with an expanded query before delivering the final response.

## License

Released under the [MIT License](./LICENSE).

<p align="center">
  <img src="./assets/live-status.svg" width="100%" alt="halo live status">
</p>
