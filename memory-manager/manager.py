"""A window into her memory.

    python memory-manager/manager.py

opens a small page on http://127.0.0.1:8790 - view, search, add, edit,
retire, restore and delete facts, and edit the user_preferences block
of memory.json, whichever backend is in charge. It's a separate
process on purpose: memory is the one thing you occasionally want to
inspect while suspecting the assistant of being wrong about it.

It reads config.json fresh on every refresh, so flipping
long_term_memory.backend in the running assistant shows up here on the
next reload without restarting anything.

Playing nicely with a running assistant:

  * sqlite - completely safe. SQLite coordinates the two processes
    itself (WAL), and the assistant reads rows per query rather than
    caching them, so an edit here is in her next prompt.
  * json - edits here rewrite memory.json, but a running assistant
    holds its copy in RAM and writes it back whenever it saves,
    clobbering yours. The page says so when that backend is live.
    Preferences are json-file edits either way, so the same warning
    applies to them.

Localhost only. It's your memory; nothing here should ever be
reachable from another machine.
"""
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MEMORY_FILE = os.path.join(BASE_DIR, "agent", "memory.json")

PORT = 8790

_json_lock = threading.Lock()


def _factstore():
    import factstore

    return factstore


def backend():
    """What config.json says right now, downgraded to json if this
    python's sqlite can't actually serve it."""
    try:
        with open(CONFIG_FILE) as f:
            wanted = str(
                json.load(f).get("long_term_memory", {}).get("backend", "json")
            ).lower()
    except Exception:
        wanted = "json"

    if wanted.startswith("sql") and _factstore().available():
        return "sqlite"

    return "json"


# ---------------------------------------------------------------------------
# The json backend, read fresh per operation - this process doesn't own
# the file, so holding a copy would only widen the window for clobbering
# whatever the assistant wrote in between.
# ---------------------------------------------------------------------------
def _load_memory():
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
    except Exception:
        data = {}

    data.setdefault("long_term_facts", [])

    return data


def _save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# One state blob for the page
# ---------------------------------------------------------------------------
def state():
    which = backend()

    if which == "sqlite":
        store = _factstore()
        active = [
            {"id": i, "fact": fact, "subject": subject, "created": created[:10]}
            for i, fact, subject, created, _w in store.rows(limit=100000)
        ]
        retired = [
            {"id": i, "fact": fact, "subject": subject,
             "created": created[:10], "retired": when[:10]}
            for i, fact, subject, created, when
            in store.rows(retired=True, limit=100000)
        ]
        warning = ""
    else:
        facts = _load_memory()["long_term_facts"]
        active = [
            {"id": i, "fact": fact, "subject": "", "created": ""}
            for i, fact in enumerate(facts)
        ]
        retired = []
        warning = (
            "json backend: if the assistant is running, it can overwrite "
            "edits made here the next time it saves. The sqlite backend "
            "doesn't have this problem."
        )

    return {
        "backend": which,
        "active": active,
        "retired": retired,
        "prefs": json.dumps(
            _load_memory().get("user_preferences", {}), indent=2
        ),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------
def apply(action, body):
    which = backend()
    fact = str(body.get("fact", "")).strip()
    subject = str(body.get("subject", "")).strip()

    if action == "prefs":
        prefs = json.loads(body.get("text", "{}"))  # invalid json -> error

        if not isinstance(prefs, dict):
            raise ValueError("preferences must be a JSON object")

        with _json_lock:
            data = _load_memory()
            data["user_preferences"] = prefs
            _save_memory(data)

        return

    if which == "sqlite":
        store = _factstore()
        fact_id = int(body.get("id", -1))

        if action == "add":
            if not fact:
                raise ValueError("empty fact")

            store.add(fact, subject)
        elif action == "edit":
            if not fact:
                raise ValueError("empty fact")

            store.set_row(fact_id, fact, subject)
        elif action == "retire":
            store.retire_row(fact_id)
        elif action == "restore":
            store.restore_row(fact_id)
        elif action == "delete":
            store.delete_row(fact_id)
        else:
            raise ValueError(f"unknown action {action!r}")

        return

    # json backend: the id is the list index
    with _json_lock:
        data = _load_memory()
        facts = data["long_term_facts"]
        index = int(body.get("id", -1))

        if action == "add":
            if not fact:
                raise ValueError("empty fact")

            if fact not in facts:
                facts.append(fact)
        elif action == "edit":
            if not fact or not 0 <= index < len(facts):
                raise ValueError("bad edit")

            facts[index] = fact
        elif action == "delete":
            if not 0 <= index < len(facts):
                raise ValueError("bad index")

            facts.pop(index)
        else:
            raise ValueError(f"{action!r} needs the sqlite backend")

        _save_memory(data)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet - it's a local page, not a service

    def _send(self, code, body, content_type="application/json"):
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html")
        elif self.path == "/api/state":
            self._send(200, json.dumps(state()))
        else:
            self._send(404, '{"error": "not found"}')

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self._send(404, '{"error": "not found"}')

            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            apply(self.path[5:], body)
            self._send(200, '{"ok": true}')
        except Exception as e:
            self._send(400, json.dumps({"ok": False, "error": str(e)}))


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luna's memory</title>
<style>
  :root {
    --bg: #16121f; --panel: #1e1830; --line: #322a4a;
    --text: #d8d2e8; --dim: #8a80a5; --accent: #b48aff;
    --accent-dim: #7c5cc4; --danger: #e06c8a; --ok: #7ad4a0;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 60px;
    background: var(--bg); color: var(--text);
    font: 15px/1.5 system-ui, sans-serif;
    max-width: 900px; margin-inline: auto;
  }
  h1 { font-size: 20px; margin: 0 0 4px; color: var(--accent); }
  h2 { font-size: 15px; margin: 28px 0 8px; color: var(--dim);
       text-transform: uppercase; letter-spacing: .08em; }
  #meta { color: var(--dim); font-size: 13px; margin-bottom: 12px; }
  #warning {
    display: none; background: #3a2438; border: 1px solid #6a3a55;
    color: #e8b8c8; padding: 8px 12px; border-radius: 8px;
    font-size: 13px; margin: 10px 0;
  }
  #search, #addrow input {
    width: 100%; padding: 9px 12px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel);
    color: var(--text); font-size: 14px; outline: none;
  }
  #search:focus, #addrow input:focus { border-color: var(--accent-dim); }
  .fact {
    display: flex; gap: 10px; align-items: flex-start;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 9px 12px; margin: 6px 0;
  }
  .fact .text { flex: 1; word-break: break-word; }
  .fact .subject {
    color: var(--accent-dim); font-size: 12px;
    border: 1px solid var(--line); border-radius: 20px;
    padding: 1px 9px; white-space: nowrap; margin-top: 2px;
  }
  .fact .date { color: var(--dim); font-size: 12px;
                white-space: nowrap; margin-top: 3px; }
  .fact.retired { opacity: .6; }
  .fact.retired .text { text-decoration: line-through; }
  button {
    background: none; border: 1px solid var(--line); color: var(--dim);
    border-radius: 6px; padding: 3px 10px; font-size: 12px;
    cursor: pointer; white-space: nowrap;
  }
  button:hover { color: var(--text); border-color: var(--accent-dim); }
  button.danger:hover { color: var(--danger); border-color: var(--danger); }
  input.inline, textarea {
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--accent-dim); border-radius: 6px;
    padding: 6px 9px; font-size: 14px; outline: none;
  }
  input.inline.subject-input { width: 130px; }
  #addrow { display: flex; gap: 8px; margin: 10px 0 4px; }
  #addrow input { flex: 1; }
  #addrow input.short { flex: 0 0 140px; }
  textarea { font-family: ui-monospace, monospace; font-size: 13px;
             min-height: 220px; resize: vertical; }
  .row-buttons { display: flex; gap: 6px; }
  #status { position: fixed; bottom: 14px; right: 16px; font-size: 13px;
            color: var(--ok); opacity: 0; transition: opacity .3s; }
  .empty { color: var(--dim); font-size: 13px; padding: 8px 2px; }
</style></head><body>

<h1>Luna's memory</h1>
<div id="meta">loading...</div>
<div id="warning"></div>

<h2>Add a fact</h2>
<div id="addrow">
  <input id="newfact" placeholder="Ryan has ..." >
  <input id="newsubject" class="short" placeholder="subject (optional)">
  <button onclick="addFact()">add</button>
</div>

<h2>Facts</h2>
<input id="search" placeholder="filter..." oninput="render()">
<div id="active"></div>

<h2 id="retired-h">Retired</h2>
<div id="retired"></div>

<h2>Preferences (memory.json)</h2>
<textarea id="prefs" spellcheck="false"></textarea>
<div style="margin-top:8px"><button onclick="savePrefs()">save preferences</button></div>

<div id="status"></div>

<script>
let S = {active: [], retired: [], backend: "json"};

function flash(msg, bad) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.color = bad ? "var(--danger)" : "var(--ok)";
  el.style.opacity = 1;
  setTimeout(() => el.style.opacity = 0, 1800);
}

async function api(action, body) {
  const r = await fetch("/api/" + action, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { flash(data.error || "failed", true); throw new Error(); }
  return data;
}

async function refresh(keepPrefs) {
  const r = await fetch("/api/state");
  S = await r.json();
  document.getElementById("meta").textContent =
    `backend: ${S.backend} - ${S.active.length} active` +
    (S.backend === "sqlite" ? `, ${S.retired.length} retired` : "") +
    ` - refresh the page after switching backends in the assistant`;
  const warn = document.getElementById("warning");
  warn.style.display = S.warning ? "block" : "none";
  warn.textContent = S.warning;
  if (!keepPrefs) document.getElementById("prefs").value = S.prefs;
  render();
}

function factRow(f, retired) {
  const row = document.createElement("div");
  row.className = "fact" + (retired ? " retired" : "");
  const text = document.createElement("div");
  text.className = "text";
  text.textContent = f.fact;
  row.appendChild(text);
  if (f.subject) {
    const s = document.createElement("div");
    s.className = "subject"; s.textContent = f.subject;
    row.appendChild(s);
  }
  if (retired && f.retired) {
    const d = document.createElement("div");
    d.className = "date"; d.textContent = "retired " + f.retired;
    row.appendChild(d);
  } else if (f.created) {
    const d = document.createElement("div");
    d.className = "date"; d.textContent = f.created;
    row.appendChild(d);
  }

  const buttons = document.createElement("div");
  buttons.className = "row-buttons";

  function btn(label, cls, fn) {
    const b = document.createElement("button");
    b.textContent = label; if (cls) b.className = cls;
    b.onclick = fn; buttons.appendChild(b);
  }

  if (!retired) {
    btn("edit", "", () => editRow(row, f));
    if (S.backend === "sqlite")
      btn("retire", "", async () => { await api("retire", {id: f.id}); refresh(true); });
  } else {
    btn("restore", "", async () => { await api("restore", {id: f.id}); refresh(true); });
  }
  btn("delete", "danger", async () => {
    if (!confirm("Delete outright?\n\n" + f.fact)) return;
    await api("delete", {id: f.id}); refresh(true);
  });

  row.appendChild(buttons);
  return row;
}

function editRow(row, f) {
  row.replaceChildren();
  const text = document.createElement("input");
  text.className = "inline"; text.value = f.fact;
  row.appendChild(text);
  let subject = null;
  if (S.backend === "sqlite") {
    subject = document.createElement("input");
    subject.className = "inline subject-input";
    subject.value = f.subject || ""; subject.placeholder = "subject";
    row.appendChild(subject);
  }
  const buttons = document.createElement("div");
  buttons.className = "row-buttons";
  const save = document.createElement("button");
  save.textContent = "save";
  save.onclick = async () => {
    await api("edit", {id: f.id, fact: text.value,
                       subject: subject ? subject.value : ""});
    flash("saved"); refresh(true);
  };
  const cancel = document.createElement("button");
  cancel.textContent = "cancel"; cancel.onclick = () => refresh(true);
  buttons.append(save, cancel);
  row.appendChild(buttons);
  text.focus();
  text.onkeydown = e => { if (e.key === "Enter") save.onclick(); };
}

function render() {
  const filter = document.getElementById("search").value.toLowerCase();
  const match = f => !filter || f.fact.toLowerCase().includes(filter)
                   || (f.subject || "").toLowerCase().includes(filter);

  const active = document.getElementById("active");
  active.replaceChildren();
  const shown = S.active.filter(match);
  if (!shown.length)
    active.innerHTML = '<div class="empty">nothing here</div>';
  shown.forEach(f => active.appendChild(factRow(f, false)));

  const retiredWrap = document.getElementById("retired");
  const heading = document.getElementById("retired-h");
  const show = S.backend === "sqlite";
  heading.style.display = show ? "" : "none";
  retiredWrap.style.display = show ? "" : "none";
  if (show) {
    retiredWrap.replaceChildren();
    const rows = S.retired.filter(match);
    if (!rows.length)
      retiredWrap.innerHTML = '<div class="empty">nothing retired</div>';
    rows.forEach(f => retiredWrap.appendChild(factRow(f, true)));
  }
}

async function addFact() {
  const fact = document.getElementById("newfact");
  const subject = document.getElementById("newsubject");
  if (!fact.value.trim()) return;
  await api("add", {fact: fact.value, subject: subject.value});
  fact.value = ""; subject.value = "";
  flash("added"); refresh(true);
}
document.getElementById("newfact").onkeydown =
  e => { if (e.key === "Enter") addFact(); };

async function savePrefs() {
  await api("prefs", {text: document.getElementById("prefs").value});
  flash("preferences saved");
}

refresh();
</script>
</body></html>
"""


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"memory manager: {url}  (Ctrl+C stops it)")
    print(f"backend right now: {backend()}")

    threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
