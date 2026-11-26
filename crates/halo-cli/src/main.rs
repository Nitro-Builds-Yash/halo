//! haloctl — native command-line client for the Halo API.
//!
//!     haloctl ask "What is the refund window?"
//!     haloctl chat
//!     haloctl upload policy.pdf manual.md
//!     haloctl docs            # list
//!     haloctl docs rm policy.pdf
//!     haloctl health
//!
//! Point it at a server with --url / HALO_URL and authenticate with --api-key / HALO_API_KEY.

mod sse;
mod ui;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use reqwest::blocking::{multipart, Client, RequestBuilder};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "haloctl",
    version,
    about = "Native client for the Halo corrective-RAG API"
)]
struct Cli {
    /// Halo API base URL
    #[arg(
        long,
        env = "HALO_URL",
        default_value = "http://localhost:8000",
        global = true
    )]
    url: String,
    /// API key (sent as X-API-Key)
    #[arg(long, env = "HALO_API_KEY", global = true, hide_env_values = true)]
    api_key: Option<String>,
    /// Print raw JSON instead of formatted output
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Ask one question and stream the pipeline steps
    Ask { question: Vec<String> },
    /// Interactive chat with follow-up questions ("new" resets, "exit" quits)
    Chat,
    /// Upload and index files (.pdf, .txt, .md)
    Upload { files: Vec<std::path::PathBuf> },
    /// List indexed documents, or remove one with `docs rm NAME`
    Docs {
        action: Option<String>,
        name: Option<String>,
    },
    /// Show server status and features
    Health,
}

struct Api {
    base: String,
    key: Option<String>,
    client: Client,
}

impl Api {
    fn new(base: &str, key: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Api {
            base: base.trim_end_matches('/').to_string(),
            key,
            client,
        })
    }

    fn auth(&self, rb: RequestBuilder) -> RequestBuilder {
        match &self.key {
            Some(k) => rb.header("X-API-Key", k),
            None => rb,
        }
    }

    fn check(resp: reqwest::blocking::Response) -> Result<reqwest::blocking::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let body = resp.text().unwrap_or_default();
        let detail = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v.get("detail").map(|d| d.to_string()))
            .unwrap_or(body);
        match status.as_u16() {
            401 => bail!("401 unauthorized: set --api-key or HALO_API_KEY"),
            _ => bail!("{status}: {detail}"),
        }
    }

    fn get(&self, path: &str) -> Result<Value> {
        let resp = self
            .auth(self.client.get(format!("{}{path}", self.base)))
            .send()
            .with_context(|| {
                format!(
                    "could not reach {} (is the server running? try `halo serve`)",
                    self.base
                )
            })?;
        Ok(Self::check(resp)?.json()?)
    }

    fn delete(&self, path: &str) -> Result<Value> {
        let resp = self
            .auth(self.client.delete(format!("{}{path}", self.base)))
            .send()?;
        Ok(Self::check(resp)?.json()?)
    }
}

fn print_answer(msg: &Value) {
    let fallback = msg["fallback"].as_bool().unwrap_or(false);
    let cached = msg["cached"].as_bool().unwrap_or(false);
    let title = if fallback {
        ui::yellow("◆ Partial answer (unverified)")
    } else {
        ui::green("◆ Verified answer")
    };
    println!(
        "\n{}{}",
        ui::bold(&title),
        if cached {
            ui::dim("  · from cache")
        } else {
            String::new()
        }
    );
    print!(
        "{}",
        ui::wrap(msg["content"].as_str().unwrap_or(""), 88, "  ")
    );
    if let Some(c) = msg["confidence"].as_f64() {
        println!("  {}", ui::confidence_bar(c));
    }

    if let Some(sentences) = msg["sentences"].as_array() {
        let judged: Vec<&Value> = sentences
            .iter()
            .filter(|s| !s["supported"].is_null())
            .collect();
        if !judged.is_empty() {
            let bad = judged.iter().filter(|s| s["supported"] == false).count();
            let head = if bad == 0 {
                ui::green("every sentence supported by the sources")
            } else {
                ui::yellow(&format!(
                    "{bad} of {} sentences not found in the sources",
                    judged.len()
                ))
            };
            println!("\n  {} {head}", ui::bold("Evidence"));
            for s in judged {
                let ok = s["supported"] == true;
                let mark = if ok {
                    ui::green("✓")
                } else {
                    ui::yellow("!")
                };
                println!("  {mark} {}", s["text"].as_str().unwrap_or(""));
                if let Some(src) = s["sources"].as_array().and_then(|a| a.first()) {
                    let page = src["page"]
                        .as_i64()
                        .map(|p| format!(" p.{p}"))
                        .unwrap_or_default();
                    println!(
                        "    {}",
                        ui::dim(&format!(
                            "{}{page}: “{}”",
                            src["source"].as_str().unwrap_or("?"),
                            src["snippet"].as_str().unwrap_or("")
                        ))
                    );
                }
            }
        }
    }

    if let Some(cits) = msg["citations"].as_array() {
        if !cits.is_empty() {
            println!("\n  {}", ui::bold("Sources"));
            for (i, c) in cits.iter().enumerate() {
                let page = c["page"]
                    .as_i64()
                    .map(|p| format!(", page {p}"))
                    .unwrap_or_default();
                println!(
                    "  {} {}{page}",
                    ui::magenta(&format!("[{}]", i + 1)),
                    ui::cyan(c["source"].as_str().unwrap_or("?"))
                );
            }
        }
    }
    println!();
}

/// Stream one question; returns the final message event.
fn ask(api: &Api, question: &str, history: &[Value], raw: bool) -> Result<Value> {
    let started = Instant::now();
    let resp = api
        .auth(api.client.post(format!("{}/api/v1/chat/stream", api.base)))
        .json(&json!({ "question": question, "history": history }))
        .send()
        .with_context(|| {
            format!(
                "could not reach {} (is the server running? try `halo serve`)",
                api.base
            )
        })?;
    let resp = Api::check(resp)?;

    let mut parser = sse::SseParser::default();
    let mut answer = None;
    for line in BufReader::new(resp).lines() {
        let Some(event) = parser.push_line(&line?) else {
            continue;
        };
        if raw {
            println!("{event}");
        }
        match event["event"].as_str() {
            Some("status") if !raw => {
                let node = event["node"].as_str().unwrap_or("");
                let (icon, label) = ui::step_label(node);
                let detail = if node == "condense" {
                    event["message"].as_str().unwrap_or("").to_string()
                } else {
                    String::new()
                };
                println!(
                    "  {icon} {} {}",
                    ui::cyan(label),
                    ui::dim(&format!(
                        "{:.1}s {}",
                        started.elapsed().as_secs_f64(),
                        detail.trim_start_matches("Follow-up understood as: ")
                    ))
                );
                std::io::stdout().flush().ok();
            }
            Some("message") => answer = Some(event),
            Some("error") => bail!(
                "{}",
                event["error"]
                    .as_str()
                    .unwrap_or("the server reported an error")
            ),
            _ => {}
        }
    }
    let answer = answer.context("the stream ended without an answer")?;
    if !raw {
        print_answer(&answer);
    }
    Ok(answer)
}

fn chat(api: &Api, raw: bool) -> Result<()> {
    println!(
        "{}  {}",
        ui::bold(&ui::cyan("◆ halo")),
        ui::dim(&format!("connected to {}", api.base))
    );
    println!(
        "{}",
        ui::dim(
            "Ask anything about your documents. Follow-ups work. 'new' resets · 'exit' quits.\n"
        )
    );
    let mut history: Vec<Value> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("{} ", ui::bold(&ui::cyan("halo ›")));
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let q = line.trim();
        match q {
            "" => continue,
            "exit" | "quit" | "q" => return Ok(()),
            "new" | "reset" | "clear" => {
                history.clear();
                println!("{}", ui::dim("Started a new conversation."));
                continue;
            }
            _ => {}
        }
        match ask(api, q, &history, raw) {
            Ok(ans) => {
                history.push(json!({ "role": "user", "content": q }));
                history.push(json!({ "role": "assistant", "content": ans["content"].as_str().unwrap_or("") }));
                let excess = history.len().saturating_sub(12);
                history.drain(..excess);
            }
            Err(e) => eprintln!("{} {e:#}", ui::red("error:")),
        }
    }
}

fn upload(api: &Api, files: &[std::path::PathBuf], raw: bool) -> Result<()> {
    if files.is_empty() {
        bail!("give at least one file: haloctl upload policy.pdf");
    }
    let mut form = multipart::Form::new();
    for f in files {
        form = form
            .file("files", f)
            .with_context(|| format!("cannot read {}", f.display()))?;
    }
    let resp = api
        .auth(
            api.client
                .post(format!("{}/api/v1/documents/upload", api.base)),
        )
        .multipart(form)
        .send()?;
    let v: Value = Api::check(resp)?.json()?;
    if raw {
        println!("{v}");
        return Ok(());
    }
    println!(
        "{} {}",
        ui::green("✓"),
        v["message"].as_str().unwrap_or("uploaded")
    );
    for r in v["rejected_files"].as_array().into_iter().flatten() {
        println!("{} {}", ui::yellow("!"), r.as_str().unwrap_or(""));
    }
    Ok(())
}

fn docs(api: &Api, action: Option<&str>, name: Option<&str>, raw: bool) -> Result<()> {
    if action == Some("rm") {
        let name = name.context("usage: haloctl docs rm <filename>")?;
        let v = api.delete(&format!("/api/v1/documents/{}", urlencode(name)))?;
        if raw {
            println!("{v}");
        } else {
            println!(
                "{} removed {name} ({} chunks)",
                ui::green("✓"),
                v["chunks_removed"]
            );
        }
        return Ok(());
    }
    let v = api.get("/api/v1/documents")?;
    if raw {
        println!("{v}");
        return Ok(());
    }
    let docs = v["documents"].as_array().cloned().unwrap_or_default();
    if docs.is_empty() {
        println!(
            "{}",
            ui::dim("No documents indexed yet. Try: haloctl upload policy.pdf")
        );
        return Ok(());
    }
    let width = docs
        .iter()
        .filter_map(|d| d["source"].as_str())
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(6)
        .max(6);
    println!(
        "{}",
        ui::bold(&format!(
            "{:<width$}  {:>6}  {:>5}",
            "SOURCE", "CHUNKS", "PAGES"
        ))
    );
    for d in &docs {
        println!(
            "{}  {:>6}  {:>5}",
            ui::cyan(&format!("{:<width$}", d["source"].as_str().unwrap_or(""))),
            d["chunks"],
            d["pages"]
        );
    }
    println!(
        "{}",
        ui::dim(&format!(
            "{} document(s), {} chunks",
            docs.len(),
            v["total_chunks"]
        ))
    );
    Ok(())
}

fn health(api: &Api, raw: bool) -> Result<()> {
    let v = api.get("/health")?;
    if raw {
        println!("{v}");
        return Ok(());
    }
    println!(
        "{} {} {}",
        ui::green("●"),
        ui::bold(&format!("halo {}", v["version"].as_str().unwrap_or("?"))),
        ui::dim(&api.base)
    );
    println!(
        "  keyword search  {}",
        v["keyword_backend"].as_str().unwrap_or("?")
    );
    println!("  auth required   {}", v["auth_required"]);
    println!("  tracing         {}", v["tracing"]);
    let feats: Vec<&str> = v["features"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| f.as_str())
        .collect();
    println!("  features        {}", ui::dim(&feats.join(", ")));
    Ok(())
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn main() {
    let cli = Cli::parse();
    let result = Api::new(&cli.url, cli.api_key.clone()).and_then(|api| match &cli.command {
        Some(Command::Ask { question }) if !question.is_empty() => {
            ask(&api, &question.join(" "), &[], cli.json).map(|_| ())
        }
        Some(Command::Ask { .. }) => Err(anyhow::anyhow!("usage: haloctl ask \"your question\"")),
        Some(Command::Chat) | None => chat(&api, cli.json),
        Some(Command::Upload { files }) => upload(&api, files, cli.json),
        Some(Command::Docs { action, name }) => {
            docs(&api, action.as_deref(), name.as_deref(), cli.json)
        }
        Some(Command::Health) => health(&api, cli.json),
    });
    if let Err(e) = result {
        eprintln!("{} {e:#}", ui::red("error:"));
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_names() {
        assert_eq!(
            urlencode("my policy (v2).pdf"),
            "my%20policy%20%28v2%29.pdf"
        );
    }

    #[test]
    fn cli_parses() {
        let c =
            Cli::try_parse_from(["haloctl", "--url", "http://x:1/", "ask", "what", "now"]).unwrap();
        assert!(
            matches!(c.command, Some(Command::Ask { ref question }) if question.join(" ") == "what now")
        );
        assert_eq!(Api::new(&c.url, None).unwrap().base, "http://x:1");
    }
}
