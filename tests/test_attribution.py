import pytest

from halo.agents.attribution import (
    attribute,
    best_evidence,
    coverage,
    split_sentences,
    support_score,
    unsupported_count,
)

DOCS = [
    "Refunds are accepted within 30 days of purchase. A receipt is required for every refund.",
    "International orders can be returned within 45 days of delivery. Shipping costs are not refunded.",
    "Gift cards cannot be exchanged for cash.",
]
METAS = [{"source": "policy.pdf", "page": 4}, {"source": "policy.pdf", "page": 5}, {"source": "faq.md", "page": 1}]


def test_split_sentences_handles_lists_and_decimals():
    text = "Refunds take 30 days. The fee is 2.5 percent.\n- First item here\n2) Second item here"
    assert split_sentences(text) == ["Refunds take 30 days.", "The fee is 2.5 percent.",
                                     "First item here", "Second item here"]


def test_support_score():
    assert support_score("Refunds are accepted within 30 days.", DOCS[0]) == 1.0
    assert support_score("Refunds are accepted within 45 days.", DOCS[0]) < 1.0
    assert support_score("The", DOCS[0]) == 0.0


def test_labelled_sentences_are_verified():
    labelled = [{"text": "Refunds are accepted within 30 days of purchase.", "chunks": [1]},
                {"text": "International orders can be returned within 45 days.", "chunks": [2]}]
    out = attribute("", labelled, DOCS, METAS)
    assert [s["supported"] for s in out] == [True, True]
    assert out[0]["sources"][0] == {"source": "policy.pdf", "page": 4, "chunk": 1,
                                    "snippet": "Refunds are accepted within 30 days of purchase."}
    assert out[1]["sources"][0]["page"] == 5


def test_wrong_label_is_corrected_from_the_passages():
    labelled = [{"text": "International orders can be returned within 45 days of delivery.", "chunks": [3]}]
    out = attribute("", labelled, DOCS, METAS)
    assert out[0]["supported"] is True
    assert out[0]["sources"][0]["chunk"] == 2


def test_invented_chunk_numbers_are_ignored():
    out = attribute("", [{"text": "Gift cards cannot be exchanged for cash.", "chunks": [99, -1, 0]}], DOCS, METAS)
    assert out[0]["supported"] is True and out[0]["sources"][0]["source"] == "faq.md"


def test_wrong_number_is_unsupported():
    labelled = [{"text": "Refunds are accepted within 60 days of purchase.", "chunks": [1]}]
    out = attribute("", labelled, DOCS, METAS)
    assert out[0]["supported"] is False
    assert unsupported_count(out) == 1


def test_unrelated_claim_is_unsupported():
    out = attribute("", [{"text": "Store credit expires after twelve months.", "chunks": []}], DOCS, METAS)
    assert out[0]["supported"] is False and out[0]["sources"] == []


def test_unlabelled_answer_is_split_and_matched():
    answer = "Refunds are accepted within 30 days of purchase. Yes. Gift cards cannot be exchanged for cash."
    out = attribute(answer, None, DOCS, METAS)
    assert [s["text"] for s in out] == ["Refunds are accepted within 30 days of purchase.", "Yes.",
                                        "Gift cards cannot be exchanged for cash."]
    assert [s["supported"] for s in out] == [True, None, True]


def test_no_documents_means_unjudged():
    assert attribute("Refunds are accepted within 30 days.", None, [], [])[0]["supported"] is None


def test_best_evidence_picks_the_matching_sentence():
    assert best_evidence("A receipt is required for refunds.", DOCS[0]) == "A receipt is required for every refund."
    assert best_evidence("x y z", "word " * 200, max_chars=20).endswith("…")


@pytest.mark.parametrize("flags,expected", [([], None), ([None], None), ([True, False], 0.5), ([True, None], 1.0)])
def test_coverage(flags, expected):
    assert coverage([{"supported": f} for f in flags]) == expected
