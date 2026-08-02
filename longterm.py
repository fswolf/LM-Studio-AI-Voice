import json
import os
import threading
import requests

from config import (
    memory,
    LM_URL,
    LONG_TERM_MEMORY_ENABLED,
    LONG_TERM_MEMORY_MAX_FACTS,
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


def add_fact(fact: str):
    fact = fact.strip()
    if not _looks_like_valid_fact(fact):
        return
    with _lock:
        facts = memory.setdefault("long_term_facts", [])
        if fact not in facts:  # simple exact-match de-dupe
            facts.append(fact)
        # Cap the list so it can't grow forever - drop the oldest
        # entries first once we're over the limit.
        overflow = len(facts) - LONG_TERM_MEMORY_MAX_FACTS
        if overflow > 0:
            del facts[:overflow]
    save()


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