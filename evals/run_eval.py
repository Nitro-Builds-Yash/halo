"""
Real evaluation: Naive RAG vs Halo CRAG on YOUR indexed corpus, scored with RAGAS.

Both systems actually run. Nothing is simulated or clamped.

Usage
-----
1. Ingest a corpus:         python -m src.rag.ingest --dir ./data
2. Write a dataset (JSONL): one object per line with
       {"question": "...", "reference": "hand-written correct answer"}
   Include some questions the corpus can NOT answer (reference: "Not in the documents.").
   See evals/dataset.example.jsonl.
3. Run:                     python -m evals.run_eval --dataset evals/dataset.jsonl
   Results are written to evals/results/<timestamp>.json (per-question + averages).

Requires: pip install -e ".[eval]"  and OPENAI_API_KEY (RAGAS uses an LLM judge).
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from src.agents.graph import graph
from src.agents.state import initial_state
from src.core.config import settings
from src.core.llm import get_embeddings, get_llm
from src.rag.vectorstore import get_vectorstore

NAIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the context. If the context doesn't contain the answer, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def load_dataset(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                row = json.loads(line)
                if "question" not in row or "reference" not in row:
                    raise ValueError(f"Each row needs 'question' and 'reference': {row}")
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def run_naive(question: str) -> Dict[str, Any]:
    """Baseline: dense top-k retrieval + a single generation, no grading or verification."""
    docs = get_vectorstore().similarity_search(question, k=settings.TOP_K_RETRIEVE)
    contexts = [d.page_content for d in docs]
    answer = (NAIVE_PROMPT | get_llm()).invoke({"context": "\n\n---\n\n".join(contexts), "question": question})
    return {"response": getattr(answer, "content", str(answer)), "contexts": contexts}


def run_halo(question: str) -> Dict[str, Any]:
    state = graph.invoke(initial_state(question))
    return {
        "response": state.get("generation", ""),
        "contexts": state.get("documents", []),
        "is_fallback": state.get("is_fallback", False),
        "verdict": state.get("verdict"),
        "web_search_used": any(m.get("type") == "web" for m in state.get("doc_metadata", [])),
    }


def score(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    dataset = EvaluationDataset.from_list([{
        "user_input": r["question"],
        "response": r["response"],
        "retrieved_contexts": r["contexts"] or [""],
        "reference": r["reference"],
    } for r in rows])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(get_llm(0.0)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
    )
    return result.to_pandas().to_dict(orient="records")


def mean(records: List[Dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and r[key] == r[key]]  # skip NaN
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--out-dir", default="evals/results")
    args = parser.parse_args()

    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is required (both pipelines and the RAGAS judge call OpenAI).")

    data = load_dataset(args.dataset)
    print(f"Evaluating {len(data)} questions with {settings.LLM_MODEL}...")

    naive_rows, halo_rows, latencies = [], [], {"naive": [], "halo": []}
    for i, item in enumerate(data, 1):
        print(f"[{i}/{len(data)}] {item['question'][:80]}")
        t = time.perf_counter()
        naive_rows.append({**item, **run_naive(item["question"])})
        latencies["naive"].append(time.perf_counter() - t)
        t = time.perf_counter()
        halo_rows.append({**item, **run_halo(item["question"])})
        latencies["halo"].append(time.perf_counter() - t)

    naive_scores, halo_scores = score(naive_rows), score(halo_rows)
    metric_keys = [k for k in halo_scores[0] if isinstance(halo_scores[0][k], (int, float))]

    summary = {
        "dataset": args.dataset,
        "n_questions": len(data),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "naive_rag": {k: mean(naive_scores, k) for k in metric_keys},
        "halo_crag": {k: mean(halo_scores, k) for k in metric_keys},
        "halo_fallback_rate": round(sum(r["is_fallback"] for r in halo_rows) / len(halo_rows), 4),
        "halo_web_search_rate": round(sum(r["web_search_used"] for r in halo_rows) / len(halo_rows), 4),
        "avg_latency_s": {k: round(sum(v) / len(v), 2) for k, v in latencies.items()},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "naive_rag": [{**r, **s} for r, s in zip(naive_rows, naive_scores)],
                   "halo_crag": [{**r, **s} for r, s in zip(halo_rows, halo_scores)]},
                  f, indent=2, default=str)

    print(json.dumps(summary, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
