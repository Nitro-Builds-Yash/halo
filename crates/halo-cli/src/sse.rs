//! Minimal Server-Sent Events parser for Halo's `/api/v1/chat/stream`.

use serde_json::Value;

/// Incremental parser: feed raw lines, get each complete event's JSON payload.
#[derive(Default)]
pub struct SseParser {
    data: String,
}

impl SseParser {
    /// Feed one line (without the trailing newline). Returns an event when a blank line ends one.
    pub fn push_line(&mut self, line: &str) -> Option<Value> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if self.data.is_empty() {
                return None;
            }
            let payload = std::mem::take(&mut self.data);
            return serde_json::from_str(&payload).ok();
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if !self.data.is_empty() {
                self.data.push('\n');
            }
            self.data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
        None // comments (":"), "event:", "id:" and "retry:" lines are ignored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events() {
        let mut p = SseParser::default();
        assert!(p
            .push_line(r#"data: {"event":"status","node":"retrieve"}"#)
            .is_none());
        let e = p.push_line("").unwrap();
        assert_eq!(e["node"], "retrieve");
        assert!(p.push_line("").is_none());
        assert!(p.push_line(": keep-alive").is_none());
        p.push_line("data: {\"event\":");
        p.push_line("data: \"done\"}");
        assert_eq!(p.push_line("\r").unwrap()["event"], "done");
    }

    #[test]
    fn ignores_bad_json() {
        let mut p = SseParser::default();
        p.push_line("data: not json");
        assert!(p.push_line("").is_none());
    }
}
