//! Native hot paths for halo's keyword search.
//!
//! * `tokenize` matches Python's `re.sub(r"[^\w\s]", " ", text.lower()).split()`.
//! * `Bm25Index` matches `rank_bm25.BM25Plus` (k1 = 1.5, b = 0.75, delta = 1.0) score for score,
//!   but stores term frequencies in flat posting lists and scores only the documents a
//!   query term actually occurs in (plus a constant for the rest), instead of a dense pass
//!   over every document per term.
//!
//! Both are exposed to Python through PyO3; halo uses them when `halo_native` is installed.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Python `\w` (Unicode): letters, digits, connector punctuation like `_`, and combining marks.
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Lowercase, turn every non-word, non-space character into a separator, split on whitespace.
pub fn tokenize_str(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in lower.chars() {
        if is_word_char(c) {
            current.push(c);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Tokenize text the same way halo's Python BM25 index does.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    tokenize_str(text)
}

/// Tokenize many texts in parallel (used when building the index).
#[pyfunction]
fn tokenize_many(py: Python<'_>, texts: Vec<String>) -> Vec<Vec<String>> {
    py.allow_threads(|| texts.par_iter().map(|t| tokenize_str(t)).collect())
}

/// Core BM25Plus index, independent of Python so it can be unit-tested with `cargo test`.
pub struct Bm25 {
    k1: f64,
    b: f64,
    delta: f64,
    doc_len: Vec<f64>,
    avgdl: f64,
    /// term -> (idf, postings: (doc index, term frequency))
    terms: HashMap<String, (f64, Vec<(u32, u32)>)>,
}

impl Bm25 {
    pub fn new(corpus: &[Vec<String>], k1: f64, b: f64, delta: f64) -> Self {
        let n = corpus.len();
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_len = Vec::with_capacity(n);
        let mut total: u64 = 0;
        for (i, doc) in corpus.iter().enumerate() {
            doc_len.push(doc.len() as f64);
            total += doc.len() as u64;
            let mut counts: HashMap<&str, u32> = HashMap::new();
            for t in doc {
                *counts.entry(t.as_str()).or_insert(0) += 1;
            }
            for (t, c) in counts {
                postings
                    .entry(t.to_string())
                    .or_default()
                    .push((i as u32, c));
            }
        }
        let avgdl = if n > 0 { total as f64 / n as f64 } else { 0.0 };
        let terms = postings
            .into_iter()
            .map(|(t, mut p)| {
                p.sort_unstable_by_key(|&(d, _)| d);
                // BM25Plus idf: ln((N + 1) / document frequency), never negative.
                let idf = ((n as f64 + 1.0) / p.len() as f64).ln();
                (t, (idf, p))
            })
            .collect();
        Bm25 {
            k1,
            b,
            delta,
            doc_len,
            avgdl,
            terms,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    /// Scores for every document, identical to rank_bm25's BM25Plus.get_scores.
    pub fn scores(&self, query: &[String]) -> Vec<f64> {
        let n = self.len();
        let mut scores = vec![0.0f64; n];
        if n == 0 || self.avgdl == 0.0 {
            return scores;
        }
        for q in query {
            let Some((idf, postings)) = self.terms.get(q) else {
                continue;
            };
            // Documents without the term still get idf * delta in BM25Plus.
            let base = idf * self.delta;
            for s in scores.iter_mut() {
                *s += base;
            }
            for &(d, tf) in postings {
                let d = d as usize;
                let tf = tf as f64;
                let norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[d] / self.avgdl);
                scores[d] += idf * (tf * (self.k1 + 1.0)) / (norm + tf);
            }
        }
        scores
    }

    /// Top-k documents containing at least one query term, best first; ties keep index order.
    pub fn top_k(&self, query: &[String], k: usize) -> Vec<(usize, f64)> {
        let scores = self.scores(query);
        let mut hit = vec![false; self.len()];
        for q in query {
            if let Some((_, postings)) = self.terms.get(q) {
                for &(d, _) in postings {
                    hit[d as usize] = true;
                }
            }
        }
        let mut ranked: Vec<(usize, f64)> = (0..self.len())
            .filter(|&d| hit[d])
            .map(|d| (d, scores[d]))
            .collect();
        // Stable sort: equal scores stay in document order, like Python's sorted(..., reverse=True).
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }
}

/// BM25Plus index over pre-tokenized documents (same parameters and scores as rank_bm25).
#[pyclass(name = "Bm25Index", module = "halo_native", frozen)]
struct PyBm25 {
    inner: Bm25,
}

#[pymethods]
impl PyBm25 {
    #[new]
    #[pyo3(signature = (corpus, k1 = 1.5, b = 0.75, delta = 1.0))]
    fn new(py: Python<'_>, corpus: Vec<Vec<String>>, k1: f64, b: f64, delta: f64) -> Self {
        let inner = py.allow_threads(|| Bm25::new(&corpus, k1, b, delta));
        PyBm25 { inner }
    }

    /// Scores for every document, in corpus order.
    fn get_scores(&self, py: Python<'_>, query: Vec<String>) -> Vec<f64> {
        py.allow_threads(|| self.inner.scores(&query))
    }

    /// (document index, score) for the best k documents that contain a query term.
    fn top_k(&self, py: Python<'_>, query: Vec<String>, k: usize) -> Vec<(usize, f64)> {
        py.allow_threads(|| self.inner.top_k(&query, k))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn halo_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_many, m)?)?;
    m.add_class::<PyBm25>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        tokenize_str(s)
    }

    #[test]
    fn tokenizer_matches_python_rules() {
        assert_eq!(
            toks("Refund-policy: §4.1, 30 days!"),
            vec!["refund", "policy", "4", "1", "30", "days"]
        );
        assert_eq!(
            toks("snake_case and Café"),
            vec!["snake_case", "and", "café"]
        );
        assert!(toks("  ... ").is_empty());
    }

    #[test]
    fn bm25plus_scores_match_reference() {
        // Values from rank_bm25.BM25Plus on the same corpus and query.
        let corpus: Vec<Vec<String>> = [
            "refund within 30 days",
            "shipping takes 5 days",
            "refund refund policy",
        ]
        .iter()
        .map(|s| toks(s))
        .collect();
        let bm = Bm25::new(&corpus, 1.5, 0.75, 1.0);
        let s = bm.scores(&toks("refund days"));
        let expected = reference(&corpus, &toks("refund days"));
        for (a, e) in s.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "{a} vs {e}");
        }
    }

    #[test]
    fn top_k_requires_overlap_and_is_stable() {
        let corpus: Vec<Vec<String>> = ["a b", "c d", "a b"].iter().map(|s| toks(s)).collect();
        let bm = Bm25::new(&corpus, 1.5, 0.75, 1.0);
        let top: Vec<usize> = bm
            .top_k(&toks("a"), 5)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(top, vec![0, 2]);
        assert!(bm.top_k(&toks("zzz"), 5).is_empty());
    }

    #[test]
    fn empty_corpus() {
        let bm = Bm25::new(&[], 1.5, 0.75, 1.0);
        assert!(bm.scores(&toks("a")).is_empty());
        assert!(bm.top_k(&toks("a"), 3).is_empty());
    }

    /// Dense, straight-line port of rank_bm25.BM25Plus.get_scores used as the oracle.
    fn reference(corpus: &[Vec<String>], query: &[String]) -> Vec<f64> {
        let n = corpus.len() as f64;
        let avgdl = corpus.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
        let mut out = vec![0.0; corpus.len()];
        for q in query {
            let df = corpus.iter().filter(|d| d.contains(q)).count() as f64;
            if df == 0.0 {
                continue;
            }
            let idf = ((n + 1.0) / df).ln();
            for (i, d) in corpus.iter().enumerate() {
                let tf = d.iter().filter(|t| *t == q).count() as f64;
                let norm = 1.5 * (1.0 - 0.75 + 0.75 * d.len() as f64 / avgdl);
                out[i] += idf * (1.0 + tf * 2.5 / (norm + tf));
            }
        }
        out
    }
}
