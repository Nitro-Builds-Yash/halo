//! Reciprocal Rank Fusion (mirrors `HybridRetriever.retrieve`).

use std::collections::HashMap;

/// Fuse several ranked lists of keys: score(key) = Σ 1 / (k + rank + 1).
/// Results are sorted by score, best first; ties keep the order keys were first seen,
/// which matches Python's insertion-ordered dict + stable `sorted(..., reverse=True)`.
pub fn rrf_fuse(rankings: &[Vec<String>], k: f64) -> Vec<(String, f64)> {
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut fused: Vec<(String, f64)> = Vec::new();
    for list in rankings {
        for (rank, key) in list.iter().enumerate() {
            let add = 1.0 / (k + rank as f64 + 1.0);
            match index.get(key.as_str()) {
                Some(&i) => fused[i].1 += add,
                None => {
                    index.insert(key.as_str(), fused.len());
                    fused.push((key.clone(), add));
                }
            }
        }
    }
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuses_and_keeps_first_seen_order_on_ties() {
        let r = rrf_fuse(
            &[vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]],
            60.0,
        );
        let keys: Vec<&str> = r.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["b", "a", "c"]);
        assert!((r[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-15);
        let tie = rrf_fuse(&[vec!["x".into()], vec!["y".into()]], 60.0);
        assert_eq!(tie[0].0, "x");
    }

    #[test]
    fn empty() {
        assert!(rrf_fuse(&[], 60.0).is_empty());
    }
}
