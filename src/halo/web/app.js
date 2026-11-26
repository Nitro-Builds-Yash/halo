/* halo web app: talks to the Halo API on the same origin. No dependencies. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), input = $("input"), sendBtn = $("sendBtn");
  const state = { history: [], busy: false, key: load("halo.apiKey") || "" };

  const STEP_LABELS = {
    condense: "Understanding follow-up", cache: "Cache hit", decompose_query: "Planning",
    retrieve: "Hybrid search", grade_documents: "Relevance gate", transform_query: "Rewriting query",
    web_search: "Web search", generate: "Writing answer", reflect: "Verifying",
    increment_retry: "Retrying", graceful_fallback: "Fallback", multi_hop: "Follow-up check", hop: "Follow-up search",
  };

  // ---------- small helpers ----------
  function load(k) { try { return localStorage.getItem(k); } catch { return null; } }
  function save(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } }
  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function headers(extra = {}) { return state.key ? { ...extra, "X-API-Key": state.key } : extra; }
  function toast(text, kind = "") {
    const t = el("div", `toast ${kind}`, esc(text));
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 4200);
  }
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }
  function safeUrl(url) {
    if (!url) return null;
    if (/^https?:\/\//i.test(url)) return url;
    if (url.startsWith("/api/")) {
      if (!state.key) return url;
      const [path, hash] = url.split("#");
      return `${path}${path.includes("?") ? "&" : "?"}api_key=${encodeURIComponent(state.key)}${hash ? "#" + hash : ""}`;
    }
    return null;
  }

  // Minimal, safe markdown: escape first, then bold/italic/code, lists and paragraphs.
  function markdown(src) {
    const inline = (s) => esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    const out = [];
    let list = null;
    for (const raw of String(src || "").split("\n")) {
      const line = raw.trimEnd();
      const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ul || ol) {
        const type = ul ? "ul" : "ol";
        if (list !== type) { if (list) out.push(`</${list}>`); out.push(`<${type}>`); list = type; }
        out.push(`<li>${inline((ul || ol)[1])}</li>`);
        continue;
      }
      if (list) { out.push(`</${list}>`); list = null; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list}>`);
    return out.join("");
  }

  // ---------- server status + documents ----------
  async function refreshHealth() {
    const status = $("status");
    try {
      const r = await fetch("/health");
      const h = await r.json();
      status.className = "status ok";
      $("statusText").textContent = `halo ${h.version} · online`;
      const backend = h.keyword_backend || "";
      const pill = $("backendPill");
      pill.textContent = backend.startsWith("rust") ? "rust search" : backend ? "python search" : "";
      pill.className = "pill" + (backend.startsWith("rust") ? " rust" : "");
      pill.title = backend;
    } catch {
      status.className = "status err";
      $("statusText").textContent = "server unreachable";
    }
  }

  async function refreshDocs() {
    const list = $("docList");
    try {
      const r = await fetch("/api/v1/documents", { headers: headers() });
      if (r.status === 401) { list.innerHTML = '<li class="empty-docs">Add your API key in settings</li>'; return; }
      const data = await r.json();
      $("docCount").textContent = data.documents.length;
      list.innerHTML = "";
      if (!data.documents.length) { list.innerHTML = '<li class="empty-docs">No documents yet</li>'; return; }
      for (const d of data.documents) {
        const ext = (d.source.split(".").pop() || "").toLowerCase();
        const li = el("li", "doc");
        li.innerHTML = `
          <span class="doc-icon ${["pdf", "md", "txt"].includes(ext) ? ext : "txt"}">${esc(ext.toUpperCase().slice(0, 4))}</span>
          <div class="doc-meta"><div class="doc-name" title="${esc(d.source)}">${esc(d.source)}</div>
            <div class="doc-sub">${d.chunks} chunks${d.pages ? ` · ${d.pages} page${d.pages > 1 ? "s" : ""}` : ""}</div></div>
          <button class="doc-del" title="Remove from index" aria-label="Remove ${esc(d.source)}">
            <svg viewBox="0 0 24 24" width="16" height="16"><path d="M4 7h16M10 11v6m4-6v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
          </button>`;
        li.querySelector(".doc-del").addEventListener("click", () => removeDoc(d.source));
        list.appendChild(li);
      }
    } catch {
      list.innerHTML = '<li class="empty-docs">Could not load documents</li>';
    }
  }

  async function removeDoc(source) {
    const r = await fetch(`/api/v1/documents/${encodeURIComponent(source)}`, { method: "DELETE", headers: headers() });
    if (r.ok) toast(`Removed ${source}`, "ok"); else toast(`Could not remove ${source}`, "err");
    refreshDocs();
  }

  async function upload(files) {
    if (!files.length) return;
    const dz = $("dropzone");
    dz.classList.add("busy");
    const form = new FormData();
    for (const f of files) form.append("files", f);
    try {
      const r = await fetch("/api/v1/documents/upload", { method: "POST", headers: headers(), body: form });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || `Upload failed (${r.status})`);
      toast(data.message || "Uploaded", "ok");
      (data.rejected_files || []).forEach((m) => toast(m, "err"));
    } catch (e) {
      toast(e.message, "err");
    } finally {
      dz.classList.remove("busy");
      refreshDocs();
    }
  }

  // ---------- rendering an answer ----------
  function ring(conf) {
    const r = 14, c = 2 * Math.PI * r, v = Math.max(0, Math.min(1, conf || 0));
    const color = v >= 0.8 ? "var(--good)" : v >= 0.5 ? "var(--warn)" : "var(--bad)";
    return `<svg class="ring" viewBox="0 0 34 34"><circle class="track" cx="17" cy="17" r="${r}"/>
      <circle class="val" cx="17" cy="17" r="${r}" stroke="${color}" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - v)}"/></svg>`;
  }

  function sourceKey(s) { return `${s.source}|${s.page ?? ""}`; }

  function renderAnswer(card, msg) {
    const sentences = (msg.sentences || []).filter((s) => s.text);
    const judged = sentences.filter((s) => s.supported !== null && s.supported !== undefined);
    const unsupported = judged.filter((s) => !s.supported).length;

    // number the distinct sources: citations first, then sentence evidence
    const order = [], seen = new Map();
    const addSrc = (s) => { const k = sourceKey(s); if (!seen.has(k)) { seen.set(k, order.length + 1); order.push(s); } return seen.get(k); };
    (msg.citations || []).forEach(addSrc);
    sentences.forEach((s) => (s.sources || []).forEach(addSrc));

    let body = markdown(msg.content);
    sentences.forEach((s, i) => {
      if (s.supported === null || s.supported === undefined) return;
      const needle = esc(s.text);
      const at = body.indexOf(needle);
      if (at < 0) return;
      const refs = (s.sources || []).map((src) => `<span class="cite">${seen.get(sourceKey(src))}</span>`).join("");
      body = body.slice(0, at) + `<span class="sent ${s.supported ? "ok" : "bad"}" data-i="${i}">${needle}</span>${refs}` + body.slice(at + needle.length);
    });

    const badge = msg.cached ? '<span class="badge cache">⚡ Cached</span>'
      : msg.fallback ? '<span class="badge warn">Partial · not verified</span>'
      : '<span class="badge ok">✓ Verified</span>';
    card.className = `card${msg.fallback ? " fallback" : ""}`;
    card.innerHTML = `
      <div class="answer-head">${badge}
        <div class="conf" title="Confidence">${ring(msg.confidence)}<span>${Math.round((msg.confidence || 0) * 100)}%</span></div>
      </div>
      <div class="answer">${body || "<p>No answer produced.</p>"}</div>`;

    if (judged.length) {
      const ev = el("details", "evidence");
      if (unsupported) ev.open = true;
      ev.innerHTML = `<summary>${unsupported ? `⚠ ${unsupported} of ${judged.length} sentences not found in the sources`
        : `✓ All ${judged.length} sentences supported by the sources`}</summary>`;
      const listEl = el("div", "ev-list");
      sentences.forEach((s, i) => {
        if (s.supported === null || s.supported === undefined) return;
        const src = (s.sources || [])[0];
        const where = src ? `${esc(src.source)}${src.page != null ? `, p.${esc(src.page)}` : ""}` : "";
        const item = el("div", `ev ${s.supported ? "ok" : "bad"}`, `
          <span class="ev-mark">${s.supported ? "✓" : "!"}</span>
          <div><div>${esc(s.text)}</div>
            <div class="ev-src">${src ? `${where}: <q>${esc(src.snippet)}</q>` : "No passage in the sources supports this sentence."}</div></div>`);
        item.dataset.i = i;
        item.addEventListener("mouseenter", () => card.querySelector(`.sent[data-i="${i}"]`)?.classList.add("lit"));
        item.addEventListener("mouseleave", () => card.querySelector(`.sent[data-i="${i}"]`)?.classList.remove("lit"));
        listEl.appendChild(item);
      });
      ev.appendChild(listEl);
      card.appendChild(ev);
      card.querySelectorAll(".sent").forEach((span) => {
        const s = sentences[+span.dataset.i];
        const src = (s.sources || [])[0];
        span.title = s.supported && src ? `${src.source}${src.page != null ? ` p.${src.page}` : ""}: “${src.snippet}”` : "Not found in the sources";
      });
    }

    const flags = msg.security_flags || [];
    if (flags.length) {
      const box = el("div", "guard", `<b>🛡 ${flags.length} hidden instruction${flags.length > 1 ? "s" : ""} blocked.</b>
        Text in the sources tried to instruct the AI; halo ${flags[0].action === "flagged" ? "flagged" : "removed"} it before answering.`);
      const list = el("ul");
      flags.slice(0, 4).forEach((f) => list.appendChild(el("li", "", `${esc(f.source || "source")}${f.page != null ? `, p.${esc(f.page)}` : ""}: <q>${esc(f.excerpt)}</q>`)));
      box.appendChild(list);
      card.appendChild(box);
    }

    if (order.length) {
      const wrap = el("div", "sources");
      order.forEach((s, i) => {
        const url = safeUrl(s.url);
        const label = `${s.source}${s.page != null ? ` · p.${s.page}` : ""}`;
        const node = el(url ? "a" : "span", "source", `<b>${i + 1}</b><span>${esc(label)}</span>`);
        if (url) { node.href = url; node.target = "_blank"; node.rel = "noopener"; }
        wrap.appendChild(node);
      });
      card.appendChild(wrap);
    }

    const meta = [];
    (msg.hops || []).forEach((h) => meta.push(`🧭 searched “${esc(h.query)}”`));
    const u = msg.usage;
    if (u) {
      const tokens = (u.input_tokens || 0) + (u.output_tokens || 0);
      meta.push(`${u.llm_calls} LLM call${u.llm_calls === 1 ? "" : "s"}`, `${tokens.toLocaleString()} tokens`,
        `~$${(u.cost_usd || 0).toFixed(4)}`, `${((u.latency_ms || 0) / 1000).toFixed(1)}s`);
      if (u.fallback_calls) meta.push(`${u.fallback_calls} via fallback model`);
    }
    if (meta.length) card.appendChild(el("div", "meta-row", meta.map((m) => `<span>${m}</span>`).join("")));
  }

  // ---------- chat flow ----------
  function addUser(text) {
    $("empty")?.remove();
    const m = el("div", "msg user", `<div class="avatar">You</div><div class="bubble"></div>`);
    m.querySelector(".bubble").textContent = text;
    chat.appendChild(m);
    return m;
  }

  function addBot() {
    const m = el("div", "msg bot", `<div class="avatar"></div><div class="bubble">
      <div class="steps"></div><div class="card"><div class="typing"><i></i><i></i><i></i></div></div></div>`);
    chat.appendChild(m);
    scrollDown();
    return m;
  }

  function markStep(steps, node, extra) {
    const prev = steps.querySelector(".step.active");
    if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
    const s = el("span", "step active", esc(STEP_LABELS[node] || node));
    if (node === "increment_retry" || node === "graceful_fallback") s.classList.add("warn");
    if (extra) s.title = extra;
    steps.appendChild(s);
  }

  async function ask(question) {
    if (state.busy || !question.trim()) return;
    state.busy = true;
    sendBtn.disabled = true;
    const userMsg = addUser(question);
    if (!state.history.length) $("topbarTitle").textContent = question;
    const bot = addBot();
    const steps = bot.querySelector(".steps"), card = bot.querySelector(".card");

    try {
      const r = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ question, history: state.history.slice(-12) }),
      });
      if (r.status === 401) throw new Error("This server needs an API key: open Settings to add it.");
      if (!r.ok) throw new Error(`Request failed (${r.status})`);

      const reader = r.body.getReader(), decoder = new TextDecoder();
      let buf = "", answer = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, cut); buf = buf.slice(cut + 2);
          const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
          if (!data) continue;
          let ev; try { ev = JSON.parse(data); } catch { continue; }
          if (ev.event === "status") {
            markStep(steps, ev.node, ev.message);
            if (ev.node === "condense") {
              userMsg.querySelector(".bubble").appendChild(el("div", "rewritten", esc(ev.message)));
            }
            scrollDown();
          } else if (ev.event === "message") {
            answer = ev;
          } else if (ev.event === "error") {
            throw new Error(ev.error || "The server reported an error.");
          }
        }
      }
      steps.querySelector(".step.active")?.classList.replace("active", "done");
      if (!answer) throw new Error("The stream ended without an answer.");
      renderAnswer(card, answer);
      state.history.push({ role: "user", content: question }, { role: "assistant", content: answer.content || "" });
    } catch (e) {
      steps.querySelector(".step.active")?.classList.replace("active", "warn");
      card.className = "card error-card";
      card.textContent = e.message;
    } finally {
      state.busy = false;
      sendBtn.disabled = !input.value.trim();
      scrollDown();
      input.focus();
    }
  }

  function newChat() {
    state.history = [];
    chat.innerHTML = "";
    const tpl = document.createElement("template");
    tpl.innerHTML = emptyHtml;
    chat.appendChild(tpl.content);
    bindSuggestions();
    $("topbarTitle").textContent = "New conversation";
    input.focus();
  }

  // ---------- wiring ----------
  const emptyHtml = $("empty").outerHTML;
  function bindSuggestions() {
    document.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
  }

  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
  input.addEventListener("input", () => { autosize(); sendBtn.disabled = state.busy || !input.value.trim(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value; input.value = ""; autosize(); ask(q);
  });
  $("newChat").addEventListener("click", newChat);
  $("menuBtn").addEventListener("click", () => $("sidebar").classList.toggle("open"));

  const dz = $("dropzone"), fileInput = $("fileInput");
  fileInput.addEventListener("change", () => { upload([...fileInput.files]); fileInput.value = ""; });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

  const root = document.documentElement;
  root.dataset.theme = load("halo.theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  $("themeBtn").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    save("halo.theme", root.dataset.theme);
  });

  const dlg = $("settings");
  $("settingsBtn").addEventListener("click", () => { $("apiKey").value = state.key; dlg.showModal(); });
  $("saveSettings").addEventListener("click", () => {
    state.key = $("apiKey").value.trim();
    save("halo.apiKey", state.key);
    toast("Settings saved", "ok");
    refreshDocs();
  });

  bindSuggestions();
  refreshHealth();
  refreshDocs();
  input.focus();
})();
