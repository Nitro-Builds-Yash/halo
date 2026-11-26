//! Sentence-evidence scoring (mirrors `halo.agents.attribution`).
//!
//! `content_words` reproduces `re.findall(r"[a-z0-9]+(?:[.%][0-9]+)?%?", text.lower())`
//! minus stopwords and short non-numeric words; `support_score` is the share of a
//! sentence's distinct content words that appear in a passage.

use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "more", "most", "must", "no", "not", "of", "on", "or", "our", "shall", "should", "so",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "under", "up", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "would", "you", "your", "also", "any", "all", "each", "other",
    "about", "after", "before", "over", "per", "via", "within", "without", "both", "either",
    "only", "same", "just", "very", "much", "many",
];

fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

#[inline]
fn alnum(c: u8) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit()
}

/// Raw regex matches of `[a-z0-9]+(?:[.%][0-9]+)?%?` over the lowercased text.
fn raw_words(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let b = lower.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !alnum(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && alnum(b[i]) {
            i += 1;
        }
        // optional group: '.' or '%' followed by at least one digit
        if i + 1 < b.len() && (b[i] == b'.' || b[i] == b'%') && b[i + 1].is_ascii_digit() {
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && b[i] == b'%' {
            i += 1;
        }
        out.push(lower[start..i].to_string());
    }
    out
}

/// Content words of `text`, in order (duplicates kept), as Python's `content_words`.
pub fn content_words(text: &str) -> Vec<String> {
    raw_words(text)
        .into_iter()
        .filter(|w| {
            !is_stopword(w) && (w.chars().count() > 2 || w.bytes().all(|c| c.is_ascii_digit()))
        })
        .collect()
}

/// Share of the sentence's distinct content words found in the passage (0.0 if none).
pub fn support_score(sentence: &str, passage: &str) -> f64 {
    let words: HashSet<String> = content_words(sentence).into_iter().collect();
    if words.is_empty() {
        return 0.0;
    }
    let found: HashSet<String> = content_words(passage).into_iter().collect();
    words.iter().filter(|w| found.contains(*w)).count() as f64 / words.len() as f64
}

/// Scores of one sentence against many passages, computing the sentence's words once.
pub fn support_scores(sentence: &str, passages: &[String]) -> Vec<f64> {
    use rayon::prelude::*;
    let words: HashSet<String> = content_words(sentence).into_iter().collect();
    if words.is_empty() {
        return vec![0.0; passages.len()];
    }
    let n = words.len() as f64;
    passages
        .par_iter()
        .map(|p| {
            let found: HashSet<String> = content_words(p).into_iter().collect();
            words.iter().filter(|w| found.contains(*w)).count() as f64 / n
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_equivalence() {
        assert_eq!(
            raw_words("Refunds: 30 days, 2.5% fee; 50% off v2.3.0"),
            vec!["refunds", "30", "days", "2.5%", "fee", "50%", "off", "v2.3", "0"]
        );
        assert_eq!(raw_words("Café İstanbul"), vec!["caf", "i", "stanbul"]);
    }

    #[test]
    fn filters_stopwords_and_short_words() {
        assert_eq!(
            content_words("The refund is 30 days at no cost"),
            vec!["refund", "30", "days", "cost"]
        );
    }

    #[test]
    fn scores() {
        assert_eq!(
            support_score(
                "Refunds within 30 days.",
                "refunds are accepted within 30 days"
            ),
            1.0
        );
        assert!(
            (support_score("Refunds within 60 days.", "refunds within 30 days") - 2.0 / 3.0).abs()
                < 1e-12
        );
        assert_eq!(support_score("The", "anything"), 0.0);
        assert_eq!(
            support_scores("refund days", &["refund".into(), "days refund".into()]),
            vec![0.5, 1.0]
        );
    }
}
