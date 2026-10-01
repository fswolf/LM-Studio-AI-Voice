"""A window into her reasoning.

    thought-viewer/start.sh      # or: python thought-viewer/viewer.py

opens http://127.0.0.1:8791. The list down the left is every turn she
thought about, newest first; whatever is selected is laid out in full
on the right - what you said, what the thinking concluded, what she
actually said, every flag the review raised, then the scratchpad
itself, round by round, verbatim. Arrow through the list and the right
side follows.

While the assistant is running the page keeps itself current, so it
can sit on a second monitor and show what she was thinking as she
says it.

Localhost only, and read-mostly: a turn can be starred, annotated or
deleted, but the reasoning is never edited. What the model thought is
what it thought.
"""
import json
import os
import sys
import threading
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PORT = 8791
PAGE_SIZE = 40


def _log():
    import thoughtlog

    return thoughtlog


def _budget():
    """generation.max_tokens from config.json, read fresh - it is the
    ceiling a long think block runs into, so each turn can say how
    close it got."""
    try:
        with open(os.path.join(BASE_DIR, "config.json")) as f:
            return int(json.load(f).get("generation", {}).get("max_tokens") or 0)
    except Exception:
        return 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="application/json", extra=()):
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")

        for key, value in extra:
            self.send_header(key, value)

        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        arg = lambda key: qs.get(key, [""])[0].strip()

        if url.path == "/":
            self._send(200, PAGE, "text/html")
        elif url.path == "/api/rows":
            log = _log()
            where = dict(search=arg("q"), kind=arg("kind"), flag=arg("flag"),
                         day=arg("day"), model=arg("model"))
            offset = max(0, int(arg("offset") or 0))
            self._send(200, json.dumps({
                "total": log.count(**where),
                "stats": log.stats(),
                "offset": offset,
                "page_size": PAGE_SIZE,
                "enabled": log.enabled(),
                "budget": _budget(),
                "rows": log.rows(limit=PAGE_SIZE, offset=offset, **where),
            }))
        elif url.path == "/api/export":
            self._send(200, _log().export(), "application/x-ndjson",
                       [("Content-Disposition",
                         'attachment; filename="luna-thoughts.jsonl"')])
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            action = self.path[5:] if self.path.startswith("/api/") else ""
            log = _log()

            if action == "delete":
                log.delete_row(int(body["id"]))
            elif action == "clear":
                log.clear(keep_starred=bool(body.get("keep_starred", True)))
            elif action == "star":
                log.star(int(body["id"]), bool(body.get("on", True)))
            elif action == "note":
                log.set_note(int(body["id"]), body.get("note", ""))
            else:
                raise ValueError(f"unknown action {action!r}")

            self._send(200, '{"ok":true}')
        except Exception as e:
            self._send(400, json.dumps({"ok": False, "error": str(e)}))


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luna's thoughts</title>
<style>
  :root {
    --bg: #13111c; --panel: #1b1829; --panel2: #201c33; --line: #2d2548;
    --text: #d9d4ec; --dim: #8880a8; --accent: #c49dff; --accent-dim: #7c5cc4;
    --cyan: #7fcfcf; --danger: #e06c8a; --ok: #7ad4a0; --amber: #e0b06c;
    --gold: #e8c35a; --think-bg: #15122a; --think-border: #3a3060;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 system-ui, sans-serif; display: flex; flex-direction: column;
  }
  button {
    background: none; border: 1px solid var(--line); color: var(--dim);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; cursor: pointer;
    white-space: nowrap; font-family: inherit;
  }
  button:hover { color: var(--text); border-color: var(--accent-dim); }
  button.on { color: var(--accent); border-color: var(--accent-dim); }
  button.live.on { color: var(--ok); border-color: var(--ok); }
  button.danger:hover { color: var(--danger); border-color: var(--danger); }
  button:disabled { opacity: .3; cursor: default; }
  mark { background: #5a4a1a; color: #ffe9a0; border-radius: 2px; }
  kbd { font: 11px ui-monospace, monospace; color: var(--dim); }

  /* ---- top ---- */
  header { padding: 14px 18px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0 14px 0 0; color: var(--accent); }
  #meta { color: var(--dim); font-size: 12.5px; }
  #meta .warn { color: var(--amber); }
  #meta .off { color: var(--danger); }
  #search {
    flex: 1; min-width: 220px; padding: 7px 11px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--text);
    font-size: 13px; outline: none;
  }
  #search:focus { border-color: var(--accent-dim); }
  #flagchips button { color: #e8b8c8; }
  #flagchips button.on { color: var(--danger); border-color: var(--danger); }
  #flagchips button i { font-style: normal; color: var(--dim); margin-left: 4px; }
  #daychip { color: var(--cyan); border-color: #3f6b6b; }
  #models button i { font-style: normal; color: var(--dim); margin-left: 5px; }
  #models button.on { color: var(--cyan); border-color: #3f6b6b; }
  .pill.model { color: var(--dim); font-family: ui-monospace, monospace; font-size: 10px; }

  /* ---- two panes ---- */
  main { flex: 1; display: grid; grid-template-columns: 400px 1fr; min-height: 0; }
  #list { overflow-y: auto; border-right: 1px solid var(--line); padding: 6px 0 40px; }
  #detail { overflow-y: auto; padding: 18px 26px 60px; }

  .day {
    position: sticky; top: 0; background: var(--bg); z-index: 1;
    color: var(--dim); font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; padding: 10px 16px 4px; cursor: pointer;
  }
  .day:hover { color: var(--cyan); }
  .item {
    padding: 8px 16px 9px; border-left: 3px solid transparent; cursor: pointer;
  }
  .item:hover { background: var(--panel); }
  .item.sel { background: var(--panel2); border-left-color: var(--accent); }
  .item .l1 { display: flex; gap: 7px; align-items: center; font-size: 12px; color: var(--dim); }
  .item .l1 .t { font-variant-numeric: tabular-nums; }
  .item .you { margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .item .hl { color: #9f93c4; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-style: italic; }
  .pill {
    font-size: 10.5px; border-radius: 20px; padding: 0 7px; border: 1px solid var(--line); color: var(--dim);
    white-space: nowrap;
  }
  .pill.src { color: var(--accent); border-color: var(--accent-dim); }
  .pill.tool { color: var(--cyan); border-color: #3f6b6b; }
  .pill.flag { color: var(--danger); border-color: var(--danger); }
  .pill.warn { color: var(--amber); border-color: #8a6a3a; }
  .pill.star { color: var(--gold); border-color: #8a7a3a; }
  .empty { color: var(--dim); font-size: 13px; padding: 20px 16px; }
  #pager { display: flex; gap: 8px; align-items: center; padding: 12px 16px; color: var(--dim); font-size: 12px; }

  /* ---- detail ---- */
  #detail .hint { color: var(--dim); font-size: 13px; max-width: 520px; line-height: 1.7; }
  .dhead { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
  .dhead .when { color: var(--dim); font-size: 13px; }
  .starbtn { border: 0; font-size: 18px; padding: 0 4px; color: var(--line); line-height: 1; }
  .starbtn.on, .starbtn:hover { color: var(--gold); }
  .dhead .sp { flex: 1; }
  .block { margin: 0 0 14px; }
  .block .lab { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 3px; }
  .block .txt { white-space: pre-wrap; word-break: break-word; }
  .block.you .txt { font-size: 15px; }
  .block.decided .txt { color: #b9aee0; font-style: italic; border-left: 2px solid var(--think-border); padding-left: 10px; }
  .block.said .txt { color: #a8d8b8; border-left: 3px solid var(--ok); padding-left: 12px; }
  .flagbox { background: #2a1a24; border: 1px solid #5a2a40; border-radius: 8px; padding: 9px 12px; margin: 0 0 14px; }
  .flagbox div { font-size: 13px; color: #e8b8c8; }
  .flagbox b { color: var(--danger); }
  .round { margin: 0 0 10px; }
  .round .rl { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; font-size: 12px; color: var(--cyan); }
  .round .rl .n { color: var(--dim); }
  .round .rl code { color: var(--cyan); background: #1b2a2a; padding: 0 5px; border-radius: 4px; font-size: 11.5px; }
  .think {
    background: var(--think-bg); border: 1px solid var(--think-border); border-radius: 8px;
    padding: 12px 14px; font: 13px/1.65 ui-monospace, "Cascadia Mono", monospace;
    white-space: pre-wrap; word-break: break-word; color: #c8c0e0;
  }
  .think .cut { color: var(--danger); font-style: italic; }
  textarea.note {
    width: 100%; min-height: 40px; resize: vertical; margin-top: 4px;
    background: var(--bg); color: #e8dca8; border: 1px dashed var(--line); border-radius: 6px;
    padding: 7px 10px; font: 13px/1.5 system-ui, sans-serif;
  }
  textarea.note:focus { border-style: solid; border-color: var(--accent-dim); outline: none; }
  .dfoot { display: flex; gap: 8px; margin-top: 16px; align-items: center; color: var(--dim); font-size: 12px; }
  .dfoot .sp { flex: 1; }
  #status { position: fixed; bottom: 12px; right: 16px; font-size: 13px; color: var(--ok); opacity: 0; transition: opacity .3s; }
  #back { display: none; }

  @media (max-width: 880px) {
    main { grid-template-columns: 1fr; }
    #list { border-right: 0; }
    body.showing #list { display: none; }
    body:not(.showing) #detail { display: none; }
    #back { display: inline-block; }
  }
</style></head><body>

<header>
  <div class="row">
    <h1>Luna's thoughts</h1>
    <div id="meta">loading…</div>
  </div>
  <div class="row" style="margin-top:10px">
    <input id="search" placeholder="search her thinking, your words, her answer, your notes   ( / )" autocomplete="off">
    <span id="kindchips">
      <button class="on" data-kind="">all</button>
      <button data-kind="tools">used tools</button>
      <button data-kind="voice">voice</button>
      <button data-kind="reminder">reminders</button>
      <button data-kind="starred">★ starred</button>
      <button data-kind="flagged">flagged</button>
    </span>
    <button id="daychip" style="display:none" onclick="setDay('')"></button>
    <button class="live on" onclick="toggleLive()" title="refresh as new turns arrive">live</button>
    <a href="/api/export" download><button>export</button></a>
    <button class="danger" onclick="clearAll()">clear</button>
  </div>
  <div class="row" id="flagchips" style="margin-top:8px"></div>
  <div class="row" id="models" style="margin-top:8px"></div>
</header>

<main>
  <div id="list"></div>
  <div id="detail"></div>
</main>
<div id="status"></div>

<script>
const $ = id => document.getElementById(id);
const S = {offset: 0, total: 0, pageSize: 40, q: "", kind: "", flag: "", day: "", model: "",
           live: true, sel: null, rows: [], budget: 0, newest: -1, timer: null};

function flash(msg, bad) {
  const el = $("status"); el.textContent = msg;
  el.style.color = bad ? "var(--danger)" : "var(--ok)";
  el.style.opacity = 1; setTimeout(() => el.style.opacity = 0, 1800);
}
async function api(action, body) {
  const r = await fetch("/api/" + action, {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  const d = await r.json().catch(() => ({}));
  if (!r.ok || !d.ok) { flash(d.error || "failed", true); throw new Error(); }
  return d;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function marked(text) {
  const frag = document.createDocumentFragment();
  if (!S.q) { frag.appendChild(document.createTextNode(text)); return frag; }
  const re = new RegExp(S.q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
  let last = 0, m;
  while ((m = re.exec(text))) {
    frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    frag.appendChild(el("mark", null, m[0]));
    last = m.index + m[0].length; if (!m[0].length) re.lastIndex++;
  }
  frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}
const k = n => n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
const tokens = r => Math.round(r.reasoning.length / 4);
function dayLabel(ts) {
  const d = new Date(ts), now = new Date(), y = new Date(now); y.setDate(now.getDate() - 1);
  const same = (a, b) => a.toDateString() === b.toDateString();
  if (same(d, now)) return "today"; if (same(d, y)) return "yesterday";
  return d.toLocaleDateString(undefined, {weekday: "long", month: "short", day: "numeric"});
}

// ---- loading ----
async function load(quiet) {
  const p = new URLSearchParams({offset: S.offset, kind: S.kind, flag: S.flag, day: S.day, model: S.model});
  if (S.q) p.set("q", S.q);
  const d = await fetch("/api/rows?" + p).then(r => r.json());
  const top = d.rows.length ? d.rows[0].id : 0;
  if (quiet && top === S.newest && d.total === S.total) return;
  S.newest = top; S.total = d.total; S.pageSize = d.page_size; S.budget = d.budget; S.rows = d.rows;
  meta(d); flagChips(d.stats); modelChips(d.stats); list();
  if (S.sel == null && d.rows.length && !quiet) select(d.rows[0].id, true);
  else if (S.sel != null) {
    const r = d.rows.find(x => x.id === S.sel);
    if (r) detail(r); else if (!quiet) { S.sel = null; if (d.rows.length) select(d.rows[0].id, true); else detail(null); }
  }
  if (!d.rows.length) detail(null);
}
function meta(d) {
  const st = d.stats, pct = st.total ? Math.round(100 * st.cut_off / st.total) : 0;
  $("meta").innerHTML =
    `${st.total} turn${st.total === 1 ? "" : "s"}` +
    ((S.q || S.kind || S.flag || S.day || S.model) ? ` · ${S.total} shown` : "") +
    (st.total ? ` · median ${k(st.chars)} chars${st.seconds ? `, ${st.seconds}s` : ""} · ${st.tools} used tools` : "") +
    (st.flagged ? ` · <span class="warn">${st.flagged} flagged</span>` : "") +
    (st.cut_off ? ` · <span class="${pct >= 15 ? "warn" : ""}">${st.cut_off} cut off (${pct}%)${pct >= 15 ? " - raise max_tokens or lower generation.reasoning" : ""}</span>` : "") +
    (d.enabled ? "" : ` · <span class="off">recording is off - /set thoughts.enabled true</span>`);
  const dc = $("daychip");
  dc.style.display = S.day ? "" : "none";
  dc.textContent = S.day ? `${dayLabel(S.day + "T12:00")} ✕` : "";
}
function flagChips(st) {
  const box = $("flagchips"); box.replaceChildren();
  const codes = Object.entries(st.by_flag || {}).sort((a, b) => b[1] - a[1]);
  if (!codes.length) return;
  box.appendChild(el("span", null, "flags:")).style.cssText = "color:var(--dim);font-size:12px";
  for (const [code, n] of codes) {
    const b = el("button", S.flag === code ? "on" : "", code);
    b.appendChild(el("i", null, String(n)));
    b.onclick = () => { S.flag = S.flag === code ? "" : code; S.offset = 0; S.sel = null; load(); };
    box.appendChild(b);
  }
}

function modelChips(st) {
  // Only once there's a comparison to make. One model is a fact, not a
  // breakdown; two is "which one lies less", as a number from your own turns.
  const box = $("models"); box.replaceChildren();
  const ms = st.by_model || [];
  if (ms.length < 2) { box.style.display = "none"; return; }
  box.style.display = "";
  box.appendChild(el("span", null, "models:")).style.cssText = "color:var(--dim);font-size:12px";
  for (const m of ms) {
    const pct = m.total ? Math.round(100 * m.flagged / m.total) : 0;
    const b = el("button", S.model === m.model ? "on" : "", shortModel(m.model));
    b.title = m.model;
    b.appendChild(el("i", null, `${m.total} turns, ${pct}% flagged`));
    b.onclick = () => { S.model = S.model === m.model ? "" : m.model; S.offset = 0; S.sel = null; load(); };
    box.appendChild(b);
  }
}
function shortModel(m) {
  // "qwen3.5-9b-the-defiant-fable-uncensored-heretic-neo-imatrix-max-mtp" is
  // not a chip. Keep the family and the size.
  const s = (m || "(unknown)").replace(/^.*\//, "");
  return s.length > 28 ? s.slice(0, 26) + "…" : s;
}

// ---- the list ----
function list() {
  const box = $("list"); box.replaceChildren();
  if (!S.rows.length) {
    box.appendChild(el("div", "empty", (S.q || S.kind || S.flag || S.day)
      ? "nothing matches" : "nothing recorded yet - she writes a row here every time she thinks before answering"));
    return;
  }
  let day = "";
  for (const r of S.rows) {
    const d = r.timestamp.slice(0, 10);
    if (d !== day) {
      const h = el("div", "day", dayLabel(r.timestamp)); h.title = "only this day";
      h.onclick = () => setDay(S.day === d ? "" : d); box.appendChild(h); day = d;
    }
    box.appendChild(item(r));
  }
  const pg = el("div"); pg.id = "pager";
  const prev = el("button", null, "← newer"), next = el("button", null, "older →");
  prev.disabled = S.offset === 0; next.disabled = S.offset + S.pageSize >= S.total;
  prev.onclick = () => page(-1); next.onclick = () => page(1);
  pg.append(prev, el("span", null, `page ${Math.floor(S.offset / S.pageSize) + 1} of ${Math.max(1, Math.ceil(S.total / S.pageSize))}`), next);
  box.appendChild(pg);
}
function item(r) {
  const it = el("div", "item" + (r.id === S.sel ? " sel" : "")); it.dataset.id = r.id;
  it.onclick = () => select(r.id);
  const l1 = el("div", "l1");
  l1.appendChild(el("span", "t", r.timestamp.slice(11, 16)));
  if (r.starred) l1.appendChild(el("span", "pill star", "★"));
  if (r.source && r.source !== "typed") l1.appendChild(el("span", "pill src", r.source));
  for (const t of r.tools.slice(0, 3)) l1.appendChild(el("span", "pill tool", t));
  if (r.tools.length > 3) l1.appendChild(el("span", "pill tool", "+" + (r.tools.length - 3)));
  if (r.flags.length) l1.appendChild(el("span", "pill flag", r.flags.length === 1 ? r.flags[0].split(":")[0] : r.flags.length + " flags"));
  if (S.budget && tokens(r) > S.budget * 0.6) l1.appendChild(el("span", "pill warn", "~" + k(tokens(r)) + " tok"));
  it.appendChild(l1);
  const you = el("div", "you"); you.appendChild(marked(r.user_text)); it.appendChild(you);
  if (r.highlight) { const h = el("div", "hl"); h.appendChild(marked(r.highlight)); it.appendChild(h); }
  return it;
}
function select(id, silent) {
  S.sel = id;
  for (const it of document.querySelectorAll(".item")) it.classList.toggle("sel", +it.dataset.id === id);
  const r = S.rows.find(x => x.id === id);
  if (r) detail(r);
  if (!silent) document.body.classList.add("showing");
  location.hash = id;
  document.querySelector(".item.sel")?.scrollIntoView({block: "nearest"});
}
function page(dir) { S.offset = Math.max(0, S.offset + dir * S.pageSize); S.sel = null; load(); }
function setDay(d) { S.day = d; S.offset = 0; S.sel = null; load(); }

// ---- the detail pane ----
function detail(r) {
  const d = $("detail"); d.replaceChildren();
  if (!r) {
    d.appendChild(el("div", "hint",
      "Pick a turn on the left. Keys: j / k move, s stars, n jumps to the note, " +
      "/ searches, Esc clears, [ ] page. Click a day heading to see only that day; " +
      "click a flag above to see only turns with it. Hover the model pill on a turn " +
      "for its prompt hash and session."));
    return;
  }
  const head = el("div", "dhead");
  const star = el("button", "starbtn" + (r.starred ? " on" : ""), r.starred ? "★" : "☆");
  star.onclick = async () => { r.starred = !r.starred; star.classList.toggle("on", r.starred);
    star.textContent = r.starred ? "★" : "☆"; await api("star", {id: r.id, on: r.starred}); list(); };
  head.appendChild(star);
  head.appendChild(el("span", "when", r.timestamp.replace("T", "  ")));
  if (r.source && r.source !== "typed") head.appendChild(el("span", "pill src", r.source));
  if (r.mood_e || r.mood_w) head.appendChild(el("span", "pill", [r.mood_e, r.mood_w].filter(Boolean).join(" · ")));
  const tk = tokens(r), over = S.budget && tk > S.budget * 0.6;
  const tp = el("span", "pill" + (over ? " warn" : ""), `~${k(tk)} tok` + (S.budget ? ` of ${k(S.budget)}` : ""));
  head.appendChild(tp);
  if (r.seconds) head.appendChild(el("span", "pill", r.seconds + "s"));
  if (r.rounds > 1) head.appendChild(el("span", "pill", r.rounds + " rounds"));
  if (r.model) { const mp = el("span", "pill model", shortModel(r.model)); mp.title = `${r.model} · prompt ${r.prompt_hash || "?"} · session ${r.session || "?"}`; head.appendChild(mp); }
  head.appendChild(el("span", "sp"));
  const back = el("button", null, "← list"); back.id = "back";
  back.onclick = () => document.body.classList.remove("showing"); head.appendChild(back);
  d.appendChild(head);

  block(d, "you", "you said", r.user_text);
  if (r.highlight) block(d, "decided", "the thinking concluded", r.highlight);
  block(d, "said", "she said", r.answer || "(nothing)");

  if (r.flags.length) {
    const fb = el("div", "flagbox");
    for (const f of r.flags) {
      const [code, ...why] = f.split(": ");
      const line = el("div"); line.appendChild(el("b", null, "!! " + code));
      line.appendChild(document.createTextNode(why.length ? " — " + why.join(": ") : ""));
      fb.appendChild(line);
    }
    d.appendChild(fb);
  }

  // The scratchpad, round by round. thoughtlog._join writes a heading
  // line before each round when there was more than one; a single
  // round has none and is shown as one block.
  const lab = el("div", "block"); lab.appendChild(el("div", "lab", "what she thought - verbatim")); d.appendChild(lab);
  const parts = r.reasoning.split(/^── (round \d+ of \d+) · (.*?) ──$/m);
  const rounds = [];
  if (parts.length === 1) rounds.push({n: "", did: "", text: parts[0]});
  else for (let i = 1; i < parts.length; i += 3) rounds.push({n: parts[i], did: parts[i + 1], text: parts[i + 2]});
  for (const rd of rounds) {
    const box = el("div", "round");
    if (rd.n) {
      const rl = el("div", "rl"); rl.appendChild(el("span", "n", rd.n));
      const m = rd.did.match(/^then called (.*)$/);
      if (m) { rl.appendChild(el("span", null, "then called")); for (const c of splitCalls(m[1])) rl.appendChild(el("code", null, c)); }
      else rl.appendChild(el("span", null, rd.did));
      box.appendChild(rl);
    }
    const th = el("div", "think");
    for (const line of rd.text.trim().split("\n")) {
      if (/^\[cut off - /.test(line)) th.appendChild(el("span", "cut", line));
      else th.appendChild(marked(line));
      th.appendChild(document.createTextNode("\n"));
    }
    box.appendChild(th); d.appendChild(box);
  }

  const nb = el("div", "block"); nb.appendChild(el("div", "lab", "your note"));
  const note = el("textarea", "note"); note.id = "note"; note.placeholder = "anything to remember about this one…";
  note.value = r.note;
  note.onchange = async () => { r.note = note.value; await api("note", {id: r.id, note: note.value}); flash("note saved"); };
  nb.appendChild(note); d.appendChild(nb);

  const foot = el("div", "dfoot");
  foot.appendChild(el("span", null, `${k(r.reasoning.length)} chars of thinking`));
  foot.appendChild(el("span", "sp"));
  const copy = el("button", null, "copy as text");
  copy.onclick = async () => {
    const t = `you: ${r.user_text}\n\n${r.reasoning}\n\nshe said: ${r.answer}` + (r.flags.length ? `\n\nflags:\n- ${r.flags.join("\n- ")}` : "");
    try { await navigator.clipboard.writeText(t); flash("copied"); } catch { flash("clipboard blocked", true); }
  };
  const del = el("button", "danger", "delete");
  del.onclick = async () => { if (!confirm("Delete this turn?")) return;
    await api("delete", {id: r.id}); S.sel = null; S.newest = -1; load(); };
  foot.append(copy, del); d.appendChild(foot);
  d.scrollTop = 0;
}
function block(parent, cls, label, text) {
  const b = el("div", "block " + cls); b.appendChild(el("div", "lab", label));
  const t = el("div", "txt"); t.appendChild(marked(text)); b.appendChild(t); parent.appendChild(b);
}
function splitCalls(s) {
  // "a, b({"x": 1, "y": 2})" - commas inside parentheses don't split.
  const out = []; let depth = 0, cur = "";
  for (const ch of s) {
    if (ch === "(") depth++; if (ch === ")") depth--;
    if (ch === "," && depth === 0) { out.push(cur.trim()); cur = ""; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim()); return out;
}

// ---- controls ----
async function clearAll() {
  if (!confirm("Delete every recorded turn except the starred ones?\n\n(Unstar first if you want those gone too.)")) return;
  await api("clear", {keep_starred: true}); S.sel = null; S.offset = 0; S.newest = -1; flash("cleared"); load();
}
function toggleLive() { S.live = !S.live; document.querySelector("button.live").classList.toggle("on", S.live); schedule(); }
function schedule() {
  clearInterval(S.timer);
  if (S.live) S.timer = setInterval(() => { if (!S.offset && !S.q && !S.kind && !S.flag && !S.day && !S.model) load(true); }, 4000);
}
for (const b of document.querySelectorAll("#kindchips button")) b.onclick = () => {
  S.kind = b.dataset.kind; S.offset = 0; S.sel = null;
  for (const o of document.querySelectorAll("#kindchips button")) o.classList.toggle("on", o === b);
  load();
};
let deb = null;
$("search").addEventListener("input", e => {
  clearTimeout(deb); deb = setTimeout(() => { S.q = e.target.value.trim(); S.offset = 0; S.sel = null; load(); }, 250);
});
document.addEventListener("keydown", e => {
  const typing = /INPUT|TEXTAREA/.test(document.activeElement.tagName);
  if (e.key === "Escape") { document.activeElement.blur(); if (S.q) { $("search").value = ""; S.q = ""; S.offset = 0; load(); } return; }
  if (e.key === "/" && !typing) { e.preventDefault(); $("search").focus(); return; }
  if (typing) return;
  const i = S.rows.findIndex(x => x.id === S.sel);
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); if (i < S.rows.length - 1) select(S.rows[i + 1].id, true); }
  if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); if (i > 0) select(S.rows[i - 1].id, true); }
  if (e.key === "s" && i >= 0) document.querySelector(".starbtn")?.click();
  if (e.key === "n" && i >= 0) $("note")?.focus();
  if (e.key === "[") page(-1);
  if (e.key === "]") page(1);
});

const want = parseInt(location.hash.slice(1));
if (want) S.sel = want;
load(); schedule();
</script>
</body></html>
"""


def main():
    url = f"http://127.0.0.1:{PORT}"

    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"thought viewer is already up at {url} - opening it")
        webbrowser.open(url)

        return

    print(f"thought viewer: {url}  (Ctrl+C stops it)")
    print(_log().summary())
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
