import json
import os
import re
import threading
import requests

from config import (
    memory,
    LM_URL,
    LONG_TERM_MEMORY_ENABLED,
    LONG_TERM_MEMORY_MAX_FACTS,
    LONG_TERM_MEMORY_CONTEXT_FACTS,
    BASE_DIR,
)

MEMORY_FILE = os.path.join(BASE_DIR, "agent", "memory.json")

_lock = threading.Lock()

memory.setdefault("long_term_facts", [])

# A real "fact" is one short, plain sentence describing something
# durable about the user - not a full assistant reply. Caps length and
# rejects multi-line / in-character text so a model that ignores the
# "one short sentence" instruction in the extraction prompt can't dump
# an entire chatty reply (cat sounds, emoji, search results, etc.)
# straight into permanent memory.
_MAX_FACT_CHARS = 220
_VOICE_TELLS = ("mrrp", "nya~", "senpai", "purring", "\U0001F63E".lower())


def get_facts() -> list:
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
    facts = get_facts()
    limit = limit or LONG_TERM_MEMORY_CONTEXT_FACTS

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
    """Remove one fact. Returns (removed_text, candidates)."""
    index, found = find_fact(phrase)

    if index is None:
        return None, found

    with _lock:
        facts = memory.setdefault("long_term_facts", [])

        if index >= len(facts):
            return None, []

        removed = facts.pop(index)

    save()

    return removed, []


def update_fact(phrase, replacement):
    """Correct one fact in place. Returns (old, new) or (None, candidates)."""
    replacement = str(replacement or "").strip()

    if not _looks_like_valid_fact(replacement):
        return None, []

    index, found = find_fact(phrase)

    if index is None:
        return None, found

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


def add_fact(fact: str) -> bool:
    """Returns True if the fact was stored, False if rejected or already
    known - the remember_fact tool reports that back to the model."""
    fact = fact.strip()
    if not _looks_like_valid_fact(fact):
        return False
    stored = False
    with _lock:
        facts = memory.setdefault("long_term_facts", [])
        if fact not in facts:  # simple exact-match de-dupe
            facts.append(fact)
            stored = True
        # Cap the list so it can't grow forever - drop the oldest
        # entries first once we're over the limit.
        overflow = len(facts) - LONG_TERM_MEMORY_MAX_FACTS
        if overflow > 0:
            del facts[:overflow]
    save()
    return stored


def _extract_fact(model, user_text, answer):
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
    if not LONG_TERM_MEMORY_ENABLED:
        return
    threading.Thread(
        target=_extract_fact, args=(model, user_text, answer), daemon=True
    ).start()