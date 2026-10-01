"""Her reasoning, kept.

Every turn that came from Ryan - typed or spoken - leaves one row here
if the model thought before it answered: the <think> scratchpad, the
reasoning_content field some servers stream instead, or both. Stream
chat and IRC never land here; those turns pass remember=False through
llm.ask() and are forgotten on purpose.

What is stored is the thinking as it happened, round by round. A turn
that calls a tool thinks at least twice - once deciding to call it,
once with the result in hand - and the first of those is usually the
interesting one, so each round is kept separately with what it went
on to call.

    /thoughts            opens the viewer (thought-viewer/)
    /thoughts 5          the last five, in the conversation

agent/thoughts.db, WAL mode, so the viewer and the assistant can both
have it open. Nothing in here raises: a log that can't be written is
a warning in the log, never a turn that fails.
"""
import json
import os
import re
import sqlite3
import threading

from datetime import datetime

import config
import logbook

from config import BASE_DIR

DB_FILE = os.path.join(BASE_DIR, "agent", "thoughts.db")

_lock = threading.Lock()

_COLUMNS = (
    "id", "timestamp", "source", "user_text", "reasoning", "answer",
    "tools", "rounds", "seconds", "mood_e", "mood_w", "starred", "note",
    "flags", "agent", "session", "model", "prompt_hash",
)

FALLBACK_ANSWER = "...sorry, I got tangled up there. Say that again?"

# One id per process, minted at import. Every row written by this run
# carries it, so a session can be told apart from the one before it
# even when they're ten minutes apart on the same day.
SESSION = datetime.now().strftime("%Y%m%d-%H%M%S")


def session_id():
    return SESSION


def _conn():
    conn = sqlite3.connect(DB_FILE, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thoughts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            source     TEXT,     -- typed | voice
            user_text  TEXT,     -- what he said, first 300 chars
            reasoning  TEXT,     -- the scratchpad, rounds joined
            answer     TEXT,     -- what she said, first 600 chars
            tools      TEXT,     -- JSON list of tool names, in call order
            rounds     INTEGER,  -- how many times the model was run
            seconds    REAL,     -- wall time for the whole turn
            mood_e     TEXT,     -- energy band word at the time
            mood_w     TEXT      -- warmth band word at the time
        )
    """)

    # The first cut of this table had has_tools instead of the three
    # columns after answer. Nobody should still have it, but a missing
    # column is a crash on every turn, so check.
    have = {row[1] for row in conn.execute("PRAGMA table_info(thoughts)")}

    for column, kind in (("tools", "TEXT"), ("rounds", "INTEGER"),
                         ("seconds", "REAL"), ("mood_e", "TEXT DEFAULT ''"),
                         ("mood_w", "TEXT DEFAULT ''"), ("starred", "INTEGER DEFAULT 0"),
                         ("note", "TEXT DEFAULT ''"), ("flags", "TEXT DEFAULT '[]'"),
                         ("agent", "TEXT DEFAULT ''"), ("session", "TEXT DEFAULT ''"),
                         ("model", "TEXT DEFAULT ''"), ("prompt_hash", "TEXT DEFAULT ''")):
        if column not in have:
            conn.execute(f"ALTER TABLE thoughts ADD COLUMN {column} {kind}")

    conn.commit()

    return conn


def _run(what, default):
    """Run `what(conn)` under the lock, swallowing anything it raises."""
    try:
        with _lock:
            conn = _conn()

            try:
                return what(conn)
            finally:
                conn.close()
    except Exception as e:
        logbook.warn("thoughtlog", "%s", e)

        return default


CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
_seen = [None]


def enabled():
    """Whether turns are being recorded.

    Three places can flip this - /set, the settings pane, and the
    viewer's own switch - and the viewer is a different process. So the
    switch lives where all three can reach it, thoughts.enabled in
    config.json, and this re-reads it whenever the file has changed
    since last time. One stat() per turn; the file is only parsed when
    something actually wrote to it.
    """
    try:
        stamp = os.path.getmtime(CONFIG_FILE)
    except OSError:
        stamp = None

    if stamp is not None and stamp != _seen[0]:
        _seen[0] = stamp

        try:
            with open(CONFIG_FILE) as f:
                block = json.load(f).get("thoughts", {})

            if isinstance(block, dict) and "enabled" in block:
                config.THOUGHTS_ENABLED = bool(block["enabled"])
        except Exception:
            pass  # a half-written or broken file keeps the last answer

    return bool(getattr(config, "THOUGHTS_ENABLED", True))


def set_enabled(on):
    """Flip it from outside the assistant - the viewer's switch. Saved
    the same way /set saves it, so the running app picks it up on its
    next turn and it survives a restart."""
    config.save_setting("thoughts.enabled", "true" if on else "false")
    _seen[0] = None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def record(user_text, rounds, answer, source="typed", tools=(),
           seconds=0.0, mood=("", ""), results=(), known_tools=(),
           agent="", model="", prompt_hash="", checks=None):
    """One turn. `rounds` is a list of {"text": ..., "called": [...]} in
    the order the model ran, from llm._record_thoughts. Nothing is
    stored for a turn with no reasoning in it.

    `results` are the tool results of the turn, looked at for errors
    and then dropped - they are not stored. `known_tools` is every
    tool she could have called, for spotting the ones she only talked
    about calling.

    agent, model and prompt_hash are who answered, with what, having
    been shown which system prompt. One assistant doesn't need them;
    the day there are several, or one prompt is changed and the flag
    rate moves, they are the only way to say which."""
    if not enabled():
        return

    rounds = [r for r in rounds or () if (r.get("text") or "").strip()]

    if not rounds:
        return

    reasoning = _join(rounds)
    names = [str(t) for t in tools or ()]
    flags = review(reasoning, str(answer or ""), names, results, known_tools,
                   checks=checks)

    def write(conn):
        conn.execute(
            "INSERT INTO thoughts (timestamp, source, user_text, reasoning, "
            "answer, tools, rounds, seconds, mood_e, mood_w, flags, "
            "agent, session, model, prompt_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), source,
             str(user_text or "")[:300], reasoning, str(answer or "")[:600],
             json.dumps(names), len(rounds), round(float(seconds or 0), 1),
             mood[0], mood[1], json.dumps(flags),
             str(agent or ""), SESSION, str(model or ""), str(prompt_hash or "")),
        )
        _trim(conn)
        conn.commit()

    _run(write, None)


def _join(rounds):
    """Rounds as one text, each under a heading that says what the
    model did next. One round gets no heading - it would only say
    'round 1 of 1'."""
    if len(rounds) == 1:
        return rounds[0]["text"].strip()

    parts = []

    for i, r in enumerate(rounds, 1):
        called = r.get("called") or []

        # A round with no call that wasn't the last one produced
        # nothing usable - that is why there was another round.
        if called:
            did = f"then called {', '.join(called)}"
        elif i < len(rounds):
            did = "then said nothing usable"
        else:
            did = "then answered"
        parts.append(f"── round {i} of {len(rounds)} · {did} ──\n{r['text'].strip()}")

    return "\n\n".join(parts)


def _trim(conn):
    cap = max(1, int(getattr(config, "THOUGHTS_MAX_RECORDS", 2000) or 2000))
    conn.execute(
        "DELETE FROM thoughts WHERE id NOT IN "
        "(SELECT id FROM thoughts ORDER BY id DESC LIMIT ?)", (cap,)
    )


def delete_row(row_id):
    _run(lambda c: (c.execute("DELETE FROM thoughts WHERE id=?",
                              (int(row_id),)), c.commit()), None)


def clear(keep_starred=True):
    """Everything, or everything you didn't star."""
    sql = "DELETE FROM thoughts" + (" WHERE starred = 0" if keep_starred else "")
    _run(lambda c: (c.execute(sql), c.commit()), None)


def star(row_id, on=True):
    _run(lambda c: (c.execute("UPDATE thoughts SET starred=? WHERE id=?",
                              (1 if on else 0, int(row_id))), c.commit()), None)


def set_note(row_id, note):
    """Your words next to hers. The reasoning itself is never edited."""
    _run(lambda c: (c.execute("UPDATE thoughts SET note=? WHERE id=?",
                              (str(note or "").strip()[:2000], int(row_id))),
                    c.commit()), None)


def export():
    """Every row as JSON lines, newest first, for keeping or sharing."""
    return "\n".join(json.dumps(r, ensure_ascii=False)
                     for r in rows(limit=10 ** 9))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
# What the viewer's filter chips mean, as SQL. "cut" is the one that
# earns its place: every turn where the model ran out of room while
# thinking, which is the list you want open when tuning max_tokens.
_KINDS = {
    "tools": "tools != '[]'",
    "cut": "reasoning LIKE '%[cut off - %'",
    "voice": "source = 'voice'",
    "reminder": "source = 'reminder'",
    "starred": "starred = 1",
    "flagged": "flags != '[]'",
}


def _where(search, kind="", flag="", day="", model=""):
    terms, params = [], ()

    if model:
        terms.append("model = ?")
        params += (model,)

    if search:
        like = f"%{search}%"
        terms.append("(reasoning LIKE ? OR user_text LIKE ? OR answer LIKE ? "
                     "OR note LIKE ?)")
        params = (like, like, like, like)

    if kind in _KINDS:
        terms.append(_KINDS[kind])

    if flag:  # one flag code, e.g. "guessed" - stored as '"guessed: ...'
        terms.append("flags LIKE ?")
        params += (f'%"{flag}: %',)

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day or ""):
        terms.append("timestamp LIKE ?")
        params += (day + "%",)

    return (" WHERE " + " AND ".join(terms) if terms else ""), params


def rows(limit=50, offset=0, search="", kind="", flag="", day="", model=""):
    """Newest first, as dicts. tools and flags come back as lists."""
    clause, params = _where(search, kind, flag, day, model)

    def read(conn):
        cur = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM thoughts{clause} "
            "ORDER BY id DESC LIMIT ? OFFSET ?", params + (limit, offset)
        )
        out = []

        for values in cur.fetchall():
            row = dict(zip(_COLUMNS, values))

            try:
                row["tools"] = json.loads(row["tools"] or "[]")
            except ValueError:
                row["tools"] = []

            row["highlight"] = highlight(row["reasoning"])
            row["starred"] = bool(row["starred"])
            row["note"] = row["note"] or ""

            try:
                row["flags"] = json.loads(row["flags"] or "[]")
            except ValueError:
                row["flags"] = []
            out.append(row)

        return out

    return _run(read, [])


def count(search="", kind="", flag="", day="", model=""):
    clause, params = _where(search, kind, flag, day, model)

    return _run(lambda c: c.execute(
        f"SELECT COUNT(*) FROM thoughts{clause}", params
    ).fetchone()[0], 0)


def stats():
    """The numbers that say whether generation is tuned right.

    cut_off is the one to watch: a few percent is a long question now
    and then; a quarter of turns means generation.reasoning is set
    above what max_tokens leaves room for. chars is the median length
    of a scratchpad, which is what to size max_tokens against.
    """
    def read(conn):
        total = conn.execute("SELECT COUNT(*) FROM thoughts").fetchone()[0]

        if not total:
            return {"total": 0, "cut_off": 0, "tools": 0, "chars": 0, "seconds": 0}

        cut = conn.execute(
            f"SELECT COUNT(*) FROM thoughts WHERE {_KINDS['cut']}"
        ).fetchone()[0]
        tools = conn.execute(
            f"SELECT COUNT(*) FROM thoughts WHERE {_KINDS['tools']}"
        ).fetchone()[0]
        flagged = conn.execute(
            f"SELECT COUNT(*) FROM thoughts WHERE {_KINDS['flagged']}"
        ).fetchone()[0]
        by_flag = {}

        for (raw,) in conn.execute("SELECT flags FROM thoughts WHERE flags != '[]'"):
            try:
                for f in json.loads(raw):
                    code = f.split(":", 1)[0]
                    by_flag[code] = by_flag.get(code, 0) + 1
            except ValueError:
                continue

        by_model = [
            {"model": m or "(unknown)", "total": n, "flagged": f}
            for m, n, f in conn.execute(
                "SELECT model, COUNT(*), SUM(flags != '[]') FROM thoughts "
                "GROUP BY model ORDER BY COUNT(*) DESC"
            )
        ]
        days = conn.execute(
            "SELECT substr(timestamp, 1, 10) AS d, COUNT(*) FROM thoughts "
            "GROUP BY d ORDER BY d DESC LIMIT 60"
        ).fetchall()
        lengths = [r[0] for r in conn.execute(
            "SELECT LENGTH(reasoning) FROM thoughts ORDER BY 1"
        )]
        times = [r[0] for r in conn.execute(
            "SELECT seconds FROM thoughts WHERE seconds > 0 ORDER BY 1"
        )]

        return {
            "total": total,
            "cut_off": cut,
            "tools": tools,
            "flagged": flagged,
            "by_flag": by_flag,
            "by_model": by_model,
            "days": days,
            "chars": lengths[len(lengths) // 2],
            "seconds": times[len(times) // 2] if times else 0,
        }

    return _run(read, {"total": 0, "cut_off": 0, "tools": 0, "flagged": 0,
                       "by_flag": {}, "by_model": [], "days": [],
                       "chars": 0, "seconds": 0})


# ---------------------------------------------------------------------------
# The review
#
# Things that look off, found by rule rather than by asking a model -
# a second opinion from the same 9B model is not a second opinion, and
# every flag here can be checked against the text by eye. These are
# heuristics: a flag means "look at this one", never "this was wrong".
# ---------------------------------------------------------------------------
_UNSURE = re.compile(
    r"\b(i (don't|do not|can't|cannot|won't) (actually |really )?know|"
    r"not sure|no way (to|of) know|can't (verify|check|be sure)|"
    r"(make|making) (something|it|this) up|(just )?guess(ing)?|"
    r"(invent|fabricat)\w*)\b", re.IGNORECASE,
)
_HEDGED = re.compile(
    r"\b(don't know|not sure|can't|couldn't|no idea|unsure|"
    r"i('m| am) not certain|i think|probably|might|maybe)\b", re.IGNORECASE,
)
_DATE_MATH = re.compile(
    r"\b(\d+\s+(days?|hours?|weeks?|months?)\s+(from|until|ago|later|before|after)|"
    r"(subtract|add|count(ing)?|calculat\w*|work(ing)? out)\b[^.]{0,40}\b"
    r"(days?|hours?|minutes?|weeks?|dates?)|"
    r"(that('s| is| would be)|so it('s| is)|which (is|makes it))\s+\w+day\b)",
    re.IGNORECASE,
)
_DATE_TOOLS = {"get_datetime", "time_until"}
_SCRATCH_OPENING = re.compile(
    r"^\s*(okay|ok|alright|hmm|so),?\s+(the user|he|ryan|let me|i need to|i should)\b",
    re.IGNORECASE,
)
_TOOL_ERROR = re.compile(
    r"^\s*(error|denied|failed|couldn't|could not|can't|cannot|unable|"
    r"no (results?|such)|not found|timed? ?out|refused)\b", re.IGNORECASE,
)
_OUT_OF_CHARACTER = re.compile(
    r"\b(as an ai\b|(large )?language model|i('m| am) (just )?an? (ai|assistant|"
    r"program)\b|i (don't|do not) have (feelings|emotions|a body|personal))",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z][a-z0-9_]+")

# The checks, by name, so an agent can say which apply to it. Luna
# gets all of them; an agent with no date tools would drop "date math",
# one that is meant to sound like an assistant would drop "broke
# character". Each is (what it reads, what it says) - the ones that
# read behaviour (was a tool called, did the answer hedge) are stronger
# evidence than the ones that read the thinking, which is the model's
# own account of itself and not always a faithful one.
CHECKS = ("promised a tool", "guessed", "date math", "leaked", "tool failed",
          "no answer", "cut off", "broke character")


def review(reasoning, answer, called, results=(), known_tools=(),
           checks=None, date_tools=_DATE_TOOLS, fallback=FALLBACK_ANSWER):
    """Flags for one turn, as "code: detail" strings. Empty is good."""
    on = set(CHECKS if checks is None else checks)
    flags = []
    called = set(called or ())
    body = _HEADING.sub("", reasoning)
    mentioned = set(_WORD.findall(body.lower())) & set(known_tools or ())

    if "promised a tool" in on:
        for name in sorted(mentioned - called):
            flags.append(f"promised a tool: talked about {name} and never called it")

    if ("guessed" in on and not called and _UNSURE.search(body)
            and not _HEDGED.search(answer)):
        flags.append("guessed: the thinking admits it doesn't know, no tool "
                     "was called, and the answer doesn't say so")

    if "date math" in on and _DATE_MATH.search(body) and not (called & set(date_tools)):
        flags.append("did date math: worked out a date or duration itself "
                     f"instead of calling {' or '.join(sorted(date_tools))}")

    if "leaked" in on and ("<think" in answer.lower() or _SCRATCH_OPENING.match(answer)):
        flags.append("leaked: the answer reads like the scratchpad, not a reply")

    if "tool failed" in on:
        for result in results or ():
            if _TOOL_ERROR.match(str(result or "")):
                flags.append(f"tool failed: {str(result).strip()[:90]}")
                break

    if "no answer" in on and answer.strip() == fallback:
        flags.append("no answer: nothing usable came back after the thinking")

    if "cut off" in on and "[cut off - " in reasoning:
        flags.append("cut off: the think block never closed")

    if "broke character" in on and (m := _OUT_OF_CHARACTER.search(answer)):
        flags.append(f"broke character: the answer says \"{m.group(0)}\"")

    return flags


def summary():
    """One line for /thoughts: how many, and how big the file is."""
    n = count()

    try:
        size = os.path.getsize(DB_FILE) / 1024
        size = f"{size / 1024:.1f} MB" if size > 1024 else f"{size:.0f} KB"
    except OSError:
        size = "no file yet"

    state = "on" if enabled() else "off (/set thoughts.enabled true)"

    return f"{n} thought record{'s' if n != 1 else ''}, {size} - recording {state}"


# ---------------------------------------------------------------------------
# The highlight
# ---------------------------------------------------------------------------
_HEADING = re.compile(r"^── .*? ──$", re.MULTILINE)
_CONCLUDING = re.compile(
    r"^(so|okay|ok|alright|right|i'll|i will|i should|let me|the answer|"
    r"final|in short|to sum|therefore|best to)\b", re.IGNORECASE,
)


_SENTENCES = re.compile(r"(?<=[.!?…])\s+")


def highlight(reasoning, width=180):
    """The line of a scratchpad worth reading first.

    A think block ends with the decision - "So I'll just tell him the
    time" - and opens with a restatement of the question, which is the
    part you already know. So: the last paragraph that reads as a
    conclusion, else the last paragraph. And since this model writes
    its whole scratchpad as one paragraph, a long pick is cut from the
    front, at a sentence boundary, never from the back.
    """
    text = _HEADING.sub("", reasoning or "")
    text = re.sub(r"\[cut off - .*?\]", "", text, flags=re.DOTALL)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return ""

    pick = next(
        (p for p in reversed(paragraphs) if _CONCLUDING.match(p)),
        paragraphs[-1],
    )
    pick = " ".join(pick.split())

    if len(pick) <= width:
        return pick

    # From the end, whole sentences, until the next one wouldn't fit.
    # A concluding sentence inside the window is a better place to
    # start than wherever the budget happened to run out.
    sentences = _SENTENCES.split(pick)
    tail, best = [], None

    for sentence in reversed(sentences):
        if sum(len(s) + 1 for s in tail) + len(sentence) > width:
            break

        tail.insert(0, sentence)

        if _CONCLUDING.match(sentence):
            best = list(tail)

    tail = best or tail

    if not tail:  # one sentence longer than the window: take its end
        return "…" + pick[-(width - 1):].lstrip()

    return ("…" if len(tail) < len(sentences) else "") + " ".join(tail)
