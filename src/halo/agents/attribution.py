"""Sentence-level citations.

The generator labels each answer sentence with the context passages ([#n]) it relies on.
That label is not trusted: every sentence is checked against the passage text, and the
best-supported passage (and the sentence inside it) becomes the evidence. Sentences no
passage supports are marked `supported: False` so the UI can flag them.
"""
import re
from typing import Any, Dict, List, Optional, Sequence

from halo.rag.bm25 import native  # optional Rust extension (halo-native); None when not installed

SUPPORT_THRESHOLD = 0.5      # share of a sentence's content words found in the evidence
FALLBACK_THRESHOLD = 0.6     # stricter bar when the model named no passage, or a wrong one
_MIN_CONTENT_WORDS = 3       # shorter sentences ("Yes.", "In short:") are not scored

_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may more most must no not of on or our shall should so such than that the their them then there these they
this those to too under up was we were what when where which while who will with would you your also any
all each other about after before over per via within without both either only same just very much many
""".split())

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?%?")


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences; list items and lines count as separate sentences."""
    parts: List[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if line:
            parts += [s.strip() for s in _SENT_SPLIT.split(line) if s.strip()]
    return parts


def content_words(text: str) -> List[str]:
    if native:
        return native.content_words(text)
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and (len(w) > 2 or w.isdigit())]


def support_scores(sentence: str, passages: Sequence[str]) -> List[float]:
    """support_score of one sentence against many passages (parallel in Rust when available)."""
    if native:
        return native.support_scores(sentence, list(passages))
    return [support_score(sentence, p) for p in passages]


def support_score(sentence: str, passage: str) -> float:
    """Fraction of the sentence's distinct content words that appear in the passage.
    Numbers must match exactly, which catches the most damaging errors (30 vs 45 days)."""
    if native:
        return native.support_score(sentence, passage)
    words = set(content_words(sentence))
    if not words:
        return 0.0
    found = set(content_words(passage))
    return len(words & found) / len(words)


def best_evidence(sentence: str, passage: str, max_chars: int = 300) -> str:
    """The sentence of `passage` that best supports `sentence`, trimmed for display."""
    candidates = split_sentences(passage) or [passage]
    scores = support_scores(sentence, candidates)
    best = candidates[max(range(len(candidates)), key=scores.__getitem__)]
    return best if len(best) <= max_chars else best[: max_chars - 1] + "…"


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def attribute(answer: str, labelled: Optional[Sequence[Dict[str, Any]]],
              documents: List[str], metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one entry per answer sentence:
    {"text", "supported", "score", "sources": [{"source", "page", "snippet", "chunk"}]}.

    `labelled` is the generator's [{"text", "chunks": [1-based ids]}], or None/empty, in which
    case the answer is split here and every passage is considered.
    """
    if labelled:
        sentences = [(str(s.get("text", "")).strip(), list(s.get("chunks") or [])) for s in labelled]
        sentences = [(t, c) for t, c in sentences if t]
    else:
        sentences = [(s, []) for s in split_sentences(answer)]

    results = []
    for text, claimed in sentences:
        entry: Dict[str, Any] = {"text": text, "supported": None, "score": None, "sources": []}
        if len(content_words(text)) < _MIN_CONTENT_WORDS or not documents:
            results.append(entry)  # too short to judge, or nothing to check against
            continue

        valid_claims = [c - 1 for c in claimed if isinstance(c, int) and 1 <= c <= len(documents)]
        all_scores = support_scores(text, documents)
        scored = sorted(((all_scores[i], i) for i in range(len(documents))), reverse=True)
        claimed_scored = sorted(((all_scores[i], i) for i in valid_claims), reverse=True)

        if claimed_scored and claimed_scored[0][0] >= SUPPORT_THRESHOLD:
            score, top = claimed_scored[0]
            keep = [i for s, i in claimed_scored if s >= SUPPORT_THRESHOLD]
        elif scored and scored[0][0] >= FALLBACK_THRESHOLD:
            score, top = scored[0]
            keep = [top]
        else:
            score, top = (scored[0] if scored else (0.0, None))
            keep = []

        # A number in the sentence that appears in no kept passage means the claim isn't supported.
        missing_numbers = _numbers(text) - set().union(*(_numbers(documents[i]) for i in keep)) if keep else set()
        supported = bool(keep) and not missing_numbers

        entry["supported"] = supported
        entry["score"] = round(score, 3)
        for i in keep[:2]:
            meta = metas[i] if i < len(metas) else {}
            entry["sources"].append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page"),
                "snippet": best_evidence(text, documents[i]),
                "chunk": i + 1,
            })
        results.append(entry)
    return results


def unsupported_count(sentences: List[Dict[str, Any]]) -> int:
    return sum(1 for s in sentences if s.get("supported") is False)


def coverage(sentences: List[Dict[str, Any]]) -> Optional[float]:
    """Share of judged sentences that are supported; None if no sentence could be judged."""
    judged = [s for s in sentences if s.get("supported") is not None]
    if not judged:
        return None
    return sum(1 for s in judged if s["supported"]) / len(judged)
