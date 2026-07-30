"""
longterm.py

Long-term memory, distinct from history.py's rolling short-term
context. After each exchange, asks the model itself whether anything
worth remembering permanently came up (a preference, a project detail,
a decision) and if so, saves it into agent/memory.json under
"long_term_facts" so it survives even after history's summary fades.

This runs as a background thread so it doesn't block the reply the
user is waiting on - but note it still costs an extra model request
every turn. If your local model server (e.g. LM Studio) only serves
one request at a time, this can still delay your *next* message while
it's finishing up, even though it's non-blocking here in Python.
Disable via agent.json -> "long_term_memory": {"enabled": false}.
"""

import json
import os
import threading
import requests

from config import memory, LM_URL, LONG_TERM_MEMORY_ENABLED, BASE_DIR

MEMORY_FILE = os.path.join(BASE_DIR, "agent", "memory.json")

_lock = threading.Lock()

memory.setdefault("long_term_facts", [])


def get_facts() -> list:
    return list(memory.get("long_term_facts", []))


def save():
    with _lock:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)


def add_fact(fact: str):
    fact = fact.strip()
    if not fact:
        return
    with _lock:
        facts = memory.setdefault("long_term_facts", [])
        if fact not in facts:  # simple exact-match de-dupe
            facts.append(fact)
    save()


def _extract_fact(model, user_text, answer):
    prompt = (
        "Based on this exchange, is there ONE important fact about the "
        "user, their preferences, or an ongoing project worth remembering "
        "permanently? Reply with ONLY that single fact as one short "
        "sentence, or reply with exactly NONE if nothing stands out.\n\n"
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

    if result and result.upper() != "NONE":
        add_fact(result)


def extract_in_background(model, user_text, answer):
    if not LONG_TERM_MEMORY_ENABLED:
        return
    threading.Thread(
        target=_extract_fact, args=(model, user_text, answer), daemon=True
    ).start()
