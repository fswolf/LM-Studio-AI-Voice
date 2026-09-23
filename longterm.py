import json
import os
import re
import threading
import requests

import config
import factstore
import logbook

# The tunables are read as config.X at each use rather than imported by
# value, so /set changes actually reach them.
from config import memory, LM_URL, BASE_DIR

MEMORY_FILE = os.path.join(BASE_DIR, "agent", "memory.json")

_lock = threading.Lock()

memory.setdefault("long_term_facts", [])


# ---------------------------------------------------------------------------
# Which backend is in charge
#
# long_term_memory.backend chooses: "json" is the original capped list
# in memory.json, "sqlite" is the permanent searchable table in
# factstore.py. The sqlite side is experimental, so the switch is live
# (/set long_term_memory.backend json, no restart) and changing it
# loses nothing in either direction:
#
#   json -> sqlite   the JSON facts are absorbed into the table
#                    (idempotently - flipping twice doesn't duplicate)
#   sqlite -> json   the newest max_facts facts are written back to
#                    memory.json; the rest STAY in the table, waiting
#
# So the db is always the superset and json is always the limited view,
# which is what makes the setting safe to flip on a bad day.
# ---------------------------------------------------------------------------
_active_backend = [None]


def backend():
    """"json" or "sqlite", after running any changeover."""
    wanted = str(getattr(config, "LONG_TERM_MEMORY_BACKEND", "json")).lower()
    wanted = "sqlite" if wanted.startswith("sql") else "json"

    if wanted == "sqlite" and not factstore.available():
        wanted = "json"  # no FTS5 in this python - said once in the log

    with _lock:
        previous = _active_backend[0]

        if previous == wanted:
            return wanted

        _active_backend[0] = wanted

    # Changeover, outside the lock - both sides take their own locks.
    if wanted == "sqlite":
        moved = factstore.import_facts(memory.get("long_term_facts", []))

        if moved or previous is not None:
            logbook.info("longterm", "backend: sqlite (+%d from json)", moved)
    elif previous == "sqlite":
        memory["long_term_facts"] = factstore.export_facts(
            config.LONG_TERM_MEMORY_MAX_FACTS
        )
        save()
        logbook.info("longterm", "backend: json (newest %d exported)",
                     len(memory["long_term_facts"]))

    _announce()

    return wanted

# A real "fact" is one short, plain sentence describing something
# durable about the user - not a full assistant reply. Caps length and
# rejects multi-line / in-character text so a model that ignores the
# "one short sentence" instruction in the extraction prompt can't dump
# an entire chatty reply (cat sounds, emoji, search results, etc.)
# straight into permanent memory.
_MAX_FACT_CHARS = 220
_VOICE_TELLS = ("mrrp", "nya~", "senpai", "purring", "\U0001F63E".lower())


def status():
    """One short line for the Memory row: backend and what's in it."""
    if not config.LONG_TERM_MEMORY_ENABLED:
        return "off"

    if backend() == "sqlite":
        active, retired = factstore.counts()
        line = f"sqlite · {active} fact{'s' if active != 1 else ''}"

        return f"{line}, {retired} retired" if retired else line

    count = len(memory.get("long_term_facts", []))

    return f"json · {count}/{config.LONG_TERM_MEMORY_MAX_FACTS} facts"


def _announce():
    """Push the current status to the header. Lazy import: ui is a
    consumer of this module, not a dependency of it."""
    try:
        import ui

        ui.set_memory(status())
    except Exception:
        pass  # headless, or the UI isn't up yet


def get_facts() -> list:
    if backend() == "sqlite":
        return factstore.active_facts()

    return list(memory.get("long_term_facts", []))


# ---------------------------------------------------------------------------
# Choosing which facts to put in front of the model
#
# Every fact used to go into every system prompt. That is fine at thirty
# and wasteful at three hundred - and worse than wasteful, because a
# wall of unrelated trivia is exactly what makes a small model start
# answering questions nobody asked ("speaking of your cat...").
#
# Below the cap nothing changes and everything is sent. Above it, the
# facts that share vocabulary with what was just said are sent, plus the
# most recent few regardless - the newest facts are usually the ones
# still in play.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "him",
    "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "not",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them",
    "then", "there", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "am", "been", "being", "get", "got", "just", "like",
    "want", "need", "know", "think", "make", "let", "about",
}


def _words(text):
    return {
        word for word in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def relevant_facts(query, limit=None):
    """The facts worth showing for this turn, oldest first."""
    limit = limit or config.LONG_TERM_MEMORY_CONTEXT_FACTS
    _announce()  # cheap, and it catches edits made in the memory manager

    if backend() == "sqlite":
        return factstore.search(query, limit)

    facts = get_facts()

    if len(facts) <= limit:
        return facts

    # The newest few always travel, relevant or not.
    keep = max(1, limit // 3)
    recent = facts[-keep:]
    candidates = facts[:-keep]

    terms = _words(query)

    if not terms:
        return facts[-limit:]

    scored = []

    for index, fact in enumerate(candidates):
        overlap = terms & _words(fact)

        if not overlap:
            continue

        # Favour the fact that says more about fewer things - a short
        # fact sharing two words beats a rambling one sharing three.
        score = len(overlap) / (1 + len(_words(fact)) ** 0.5)
        scored.append((score, index, fact))

    scored.sort(reverse=True)
    chosen = {index for _score, index, _fact in scored[:limit - keep]}

    return [f for i, f in enumerate(candidates) if i in chosen] + recent


# ---------------------------------------------------------------------------
# Editing what's remembered
#
# Adding was the only thing possible before, which meant a fact she
# misheard was permanent short of hand-editing memory.json. Both of
# these take a phrase rather than an index, for the same reason
# set_reminder takes one: the model is good at repeating what was said
# and bad at bookkeeping.
# ---------------------------------------------------------------------------
def find_fact(phrase):
    """Best match for a loose description. Returns (index, fact) or
    (None, candidates) when it's too close to call."""
    facts = get_facts()

    if not facts:
        return None, []

    phrase = str(phrase or "").strip()
    lowered = phrase.lower()

    if not lowered:
        return None, []

    # An exact or containing match wins outright.
    for index, fact in enumerate(facts):
        if fact.lower() == lowered:
            return index, fact

    contains = [
        (index, fact) for index, fact in enumerate(facts)
        if lowered in fact.lower() or fact.lower() in lowered
    ]

    if len(contains) == 1:
        return contains[0]

    terms = _words(phrase)

    if not terms:
        return None, [fact for _index, fact in contains]

    scored = sorted(
        (
            (len(terms & _words(fact)) / (1 + len(terms ^ _words(fact))),
             index, fact)
            for index, fact in enumerate(facts)
        ),
        reverse=True,
    )

    best = [entry for entry in scored if entry[0] > 0]

    if not best:
        return None, []

    # Clearly ahead of the runner-up, or it's a guess not worth making.
    if len(best) == 1 or best[0][0] >= best[1][0] * 1.5:
        return best[0][1], best[0][2]

    return None, [fact for _score, _index, fact in best[:4]]


def forget_fact(phrase):
    """Remove one fact. Returns (removed_text, candidates).

    Both backends really delete here. "Forget that" is the one case
    where keeping a retired copy would be the opposite of what was
    asked."""
    index, found = find_fact(phrase)

    if index is None:
        return None, found

    if backend() == "sqlite":
        removed = factstore.remove(found)
        _announce()
        return (found, []) if removed else (None, [])

    with _lock:
        facts = memory.setdefault("long_term_facts", [])

        if index >= len(facts):
            return None, []

        removed = facts.pop(index)

    save()
    _announce()

    return removed, []


def update_fact(phrase, replacement):
    """Correct one fact in place. Returns (old, new) or (None, candidates)."""
    replacement = str(replacement or "").strip()

    if not _looks_like_valid_fact(replacement):
        return None, []

    index, found = find_fact(phrase)

    if index is None:
        return None, found

    if backend() == "sqlite":
        # A correction, not a change in the world - rewritten in place
        # rather than superseded, so no history row.
        if factstore.update(found, replacement):
            return (found, replacement), []

        return None, []

    with _lock:
        facts = memory.setdefault("long_term_facts", [])

        if index >= len(facts):
            return None, []

        old = facts[index]
        facts[index] = replacement

    save()

    return (old, replacement), []


def save():
    with _lock:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)


def _looks_like_valid_fact(fact: str) -> bool:
    if not fact or fact.upper() == "NONE":
        return False
    if len(fact) > _MAX_FACT_CHARS:
        return False
    if "\n" in fact:  # a real fact is one line - multi-line means a leaked reply
        return False
    lowered = fact.lower()
    if any(tell in lowered for tell in _VOICE_TELLS):
        return False
    return True


def add_fact(fact: str, subject: str = "") -> bool:
    """Returns True if the fact was stored, False if rejected or already
    known - the remember_fact tool reports that back to the model."""
    fact = fact.strip()
    if not _looks_like_valid_fact(fact):
        return False
    if backend() == "sqlite":
        # No cap here - permanence is the point of this backend.
        stored = factstore.add(fact, subject)
        _announce()
        return stored
    stored = False
    with _lock:
        facts = memory.setdefault("long_term_facts", [])
        if fact not in facts:  # simple exact-match de-dupe
            facts.append(fact)
            stored = True
        # Cap the list so it can't grow forever - drop the oldest
        # entries first once we're over the limit. (Read off config
        # live, so /set long_term_memory.max_facts actually applies.)
        overflow = len(facts) - config.LONG_TERM_MEMORY_MAX_FACTS
        if overflow > 0:
            del facts[:overflow]
    save()
    _announce()
    return stored


def _ask_extractor(model, prompt):
    response = requests.post(
        LM_URL,
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )

    return response.json()["choices"][0]["message"]["content"].strip()


def _extract_fact_sqlite(model, user_text, answer):
    """Extraction with the table in charge: the model may also notice
    that a new statement makes an old fact false.

    The old prompt could only answer "new fact" or "nothing", so "I got
    a 7900 XTX" landed *next to* "Ryan has a 6950 XT" and both got
    served forever after. Here the model gets a third verb - and the
    facts it might be contradicting are chosen by the same search the
    conversation uses, not the last 25 by age, which stops working the
    moment the table outgrows its window.

    Every parse failure degrades toward add(): the worst outcome of a
    mangled reply is the coexistence bug we already had, never a lost
    fact and never a wrongly retired one.
    """
    known = factstore.search(f"{user_text} {answer}", limit=12)
    known_block = ""

    if known:
        known_block = (
            "Facts already remembered that may be relevant:\n"
            + "\n".join(f"- {f}" for f in known)
            + "\n\n"
        )

    used = factstore.subjects()
    subject_line = (
        "Reuse one of these subjects when it fits: " + ", ".join(used) + ". "
        if used else ""
    )

    prompt = (
        "You are deciding whether to permanently remember something from "
        "this exchange. Be strict - most exchanges contain nothing worth "
        "permanently remembering. Only report a fact if it is durable and "
        "specific (identity, an explicitly stated preference, a concrete "
        "standing project or commitment) - NOT routine chit-chat.\n\n"
        f"{known_block}"
        "Reply with exactly ONE line, in one of these three forms:\n"
        "NONE\n"
        "NEW | <one-word subject> | <the fact, one short sentence>\n"
        "REPLACE | <the old fact, copied exactly from the list above> | "
        "<the new fact, one short sentence>\n\n"
        "Use REPLACE only when the exchange shows a listed fact is no "
        "longer true (something was replaced, changed, ended). If both "
        "could still be true at once, use NEW. "
        f"{subject_line}"
        "When in doubt, reply NONE.\n\n"
        f"User: {user_text}\nAssistant: {answer}"
    )

    try:
        raw = _ask_extractor(model, prompt)
    except Exception:
        return  # extraction failing shouldn't ever break the conversation

    # A reasoning model may think out loud first; the answer is the
    # last non-empty line once any <think> block is gone.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if not lines:
        return

    line = lines[-1]

    if line.upper() == "NONE":
        return

    parts = [p.strip() for p in line.split("|")]
    verb = parts[0].upper()

    if verb == "REPLACE" and len(parts) >= 3:
        old, new = parts[1], parts[-1]

        if not _looks_like_valid_fact(new):
            return

        # Exact echo first; the fuzzy matcher when the model reworded
        # the old fact, which it will.
        if factstore.replace(old, new):
            logbook.info("longterm", "superseded: %r -> %r", old, new)
            _announce()

            return

        index, found = find_fact(old)

        if index is not None and factstore.replace(found, new):
            logbook.info("longterm", "superseded: %r -> %r", found, new)
            _announce()

            return

        add_fact(new)  # couldn't place the old one - degrade to coexist

        return

    if verb == "NEW" and len(parts) >= 2:
        subject = parts[1] if len(parts) >= 3 else ""

        if len(subject.split()) > 1:  # that's a fact, not a subject
            subject = ""

        if _looks_like_valid_fact(parts[-1]):
            add_fact(parts[-1], subject)

        return

    # Not in the format at all - an older-style bare fact still counts.
    if len(parts) == 1 and _looks_like_valid_fact(line):
        add_fact(line)


def _extract_fact(model, user_text, answer):
    if backend() == "sqlite":
        _extract_fact_sqlite(model, user_text, answer)

        return

    existing = get_facts()
    existing_block = ""
    if existing:
        existing_block = (
            "Facts already remembered (do NOT log anything that repeats "
            "or reworders any of these):\n"
            + "\n".join(f"- {f}" for f in existing[-25:])
            + "\n\n"
        )

    prompt = (
        "You are deciding whether to permanently remember something from "
        "this exchange. Be strict - most exchanges contain nothing worth "
        "permanently remembering. Only report a fact if it is durable and "
        "specific (identity, an explicitly stated preference, a concrete "
        "standing project or commitment) - NOT routine chit-chat, and NOT "
        "already covered by the existing facts below in different words.\n\n"
        f"{existing_block}"
        "If this exchange contains a genuinely new, specific, durable "
        "fact, reply with ONLY that fact as one short sentence. Otherwise "
        "reply with exactly NONE. When in doubt, reply NONE.\n\n"
        f"User: {user_text}\nAssistant: {answer}"
    )
    try:
        response = requests.post(
            LM_URL,
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        result = response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return  # extraction failing shouldn't ever break the conversation

    if _looks_like_valid_fact(result):
        add_fact(result)


def extract_in_background(model, user_text, answer):
    if not config.LONG_TERM_MEMORY_ENABLED:
        return
    threading.Thread(
        target=_extract_fact, args=(model, user_text, answer), daemon=True
    ).start()