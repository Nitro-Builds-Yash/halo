//! Terminal output: colors (disabled for NO_COLOR or when not a terminal) and layout helpers.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn color_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal())
}

fn paint(code: &str, s: &str) -> String {
    if color_on() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("2", s)
}
pub fn cyan(s: &str) -> String {
    paint("36", s)
}
pub fn green(s: &str) -> String {
    paint("32", s)
}
pub fn yellow(s: &str) -> String {
    paint("33", s)
}
pub fn red(s: &str) -> String {
    paint("31", s)
}
pub fn magenta(s: &str) -> String {
    paint("35", s)
}

/// A 20-cell confidence bar, e.g. `██████████████░░░░░░ 72%`.
pub fn confidence_bar(c: f64) -> String {
    let c = c.clamp(0.0, 1.0);
    let filled = (c * 20.0).round() as usize;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(20 - filled));
    let pct = format!("{:>3.0}%", c * 100.0);
    let colored = if c >= 0.8 {
        green(&bar)
    } else if c >= 0.5 {
        yellow(&bar)
    } else {
        red(&bar)
    };
    format!("{colored} {pct}")
}

/// Wrap text to `width` columns with a left indent.
pub fn wrap(text: &str, width: usize, indent: &str) -> String {
    let mut out = String::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
                out.push_str(indent);
                out.push_str(&line);
                out.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push_str(indent);
        out.push_str(&line);
        out.push('\n');
    }
    out
}

pub fn step_label(node: &str) -> (&'static str, &'static str) {
    match node {
        "condense" => ("💬", "Follow-up understood"),
        "cache" => ("⚡", "Semantic cache hit"),
        "decompose_query" => ("🧩", "Planning the query"),
        "retrieve" => ("🔍", "Hybrid search"),
        "grade_documents" => ("⚖️ ", "Relevance gate"),
        "transform_query" => ("🔄", "Rewriting for web search"),
        "web_search" => ("🌐", "Web search"),
        "generate" => ("✍️ ", "Writing the answer"),
        "reflect" => ("🧪", "Verifying against sources"),
        "increment_retry" => ("↻ ", "Retrying with web context"),
        "graceful_fallback" => ("🛟", "Falling back to excerpts"),
        _ => ("• ", "Working"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_bars() {
        assert_eq!(wrap("aaa bbb ccc", 7, "  "), "  aaa bbb\n  ccc\n");
        assert!(confidence_bar(0.5).ends_with(" 50%"));
        assert!(confidence_bar(7.0).ends_with("100%"));
    }
}
