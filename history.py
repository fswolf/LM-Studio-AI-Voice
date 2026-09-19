"""Conversation history, and the running summary that stands in for the
part of it that's been dropped.

Two things were wrong with the old version.

The summary only ever grew. Every time older turns were folded away,
another paragraph was appended to it and nothing ever re-read the
whole. After a few weeks of daily use the "summary" was longer than the
history it replaced, which is the opposite of the job. It is now
re-compressed whenever it passes a cap, so it converges instead of
climbing.

And summarizing blocked the reply. It fired on the turn that tipped the
message count over the limit, in front of the user, adding a whole
extra model call to that one answer - so every fifteenth thing you said
took twice as long for no visible reason. It now runs on a worker and
applies its result when it's ready.
"""
import json
import os
import threading

import requests

from datetime import datetime

from config import (
    LM_URL,
    MAX_RAW_MESSAGES,
    SUMMARIZE_CHUNK,
    SUMMARY_MAX_CHARS,
    BASE_DIR,
)

HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "conversation.json")

_data = {"summary": "", "messages": []}

# Guards _data against the reminder scanner and a typed turn landing at
# the same moment, and against the background summarizer writing back
# into a list that has moved on since it started.
_lock = threading.RLock()
_summarizing = False


def load():
    global _data

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                _data = json.load(f)
        except (json.JSONDecodeError, OSError):
            _data = {"summary": "", "messages": []}
    else:
        _data = {"summary": "", "messages": []}


def save():
    os.makedirs(HISTORY_DIR, exist_ok=True)

    with _lock:
        snapshot = json.dumps(_data, indent=2)

    try:
        with open(HISTORY_FILE, "w") as f:
            f.write(snapshot)
    except OSError:
        pass  # best effort; a failed save shouldn't kill the turn


def get_summary() -> str:
    with _lock:
        return _data.get("summary", "")


def get_messages() -> list:
    """Recent raw messages, oldest first, as {"role","content"} dicts.

    Timestamps are stripped here since this is what gets sent straight
    to the model's API - use get_messages_full() if you need them.
    """
    with _lock:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in _data.get("messages", [])
        ]


def get_messages_full() -> list:
    """Same as get_messages(), but keeps the "timestamp" field on each
    message - use this for display purposes (e.g. showing timestamps
    in the UI) rather than feeding it to the model."""
    with _lock:
        return list(_data.get("messages", []))


def add_message(role: str, content: str):
    with _lock:
        _data.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


def clear():
    with _lock:
        _data["messages"] = []
        _data["summary"] = ""

    save()


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def _ask_model(model, prompt):
    response = requests.post(
        LM_URL,
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
    )
    data = response.json()

    if "choices" not in data:
        detail = data.get("error", data)

        raise RuntimeError(f"LM Studio error during summarization: {detail}")

    return data["choices"][0]["message"]["content"].strip()


def _summarize_chunk(model, chunk):
    """Compress a batch of older turns into a few sentences."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)

    return _ask_model(model, (
        "Summarize the key facts, decisions, and context from this "
        "conversation excerpt in a few concise sentences, written for an "
        "AI assistant's own future reference. Skip pleasantries, keep only "
        "information worth remembering.\n\n" + transcript
    ))


def _compress(model, summary):
    """Fold a summary that's grown too long back into itself.

    This is the part that was missing. Without it the notes accumulate
    one paragraph per fold forever, and eventually the thing meant to
    save context is the largest thing in the prompt.
    """
    return _ask_model(model, (
        "These are running notes an AI assistant keeps about an ongoing "
        "conversation. They have grown too long. Rewrite them as a "
        "shorter set of notes that keeps every durable fact, decision, "
        "name, preference and open thread, and drops repetition, "
        "chronology and anything already superseded by a later note. "
        "Write plain sentences, no headings, no preamble.\n\n" + summary
    ))


def _fold(model):
    """Move the oldest turns into the summary. Runs on a worker."""
    global _summarizing

    try:
        with _lock:
            messages = _data.get("messages", [])

            if len(messages) <= MAX_RAW_MESSAGES:
                return

            chunk = list(messages[:SUMMARIZE_CHUNK])
            existing = _data.get("summary", "")

        # The model call happens outside the lock - it's the slow part,
        # and holding a lock across it would stall every other turn.
        piece = _summarize_chunk(model, chunk)
        merged = f"{existing}\n{piece}".strip() if existing else piece

        if len(merged) > SUMMARY_MAX_CHARS:
            try:
                merged = _compress(model, merged)
            except Exception:
                # Compression failing is survivable; an over-long
                # summary is better than a lost one. Trim the oldest
                # half so it can't grow without bound either way.
                merged = merged[-SUMMARY_MAX_CHARS:]

        with _lock:
            messages = _data.get("messages", [])

            # Another turn may have landed while we were waiting. The
            # chunk we summarized is still the oldest, so dropping that
            # many from the front is still correct - but only if they
            # haven't already been dropped by a previous fold.
            if messages[:SUMMARIZE_CHUNK] != chunk:
                return

            _data["summary"] = merged
            _data["messages"] = messages[SUMMARIZE_CHUNK:]

        save()
    except Exception:
        pass  # summarization is a nicety; never let it break a turn
    finally:
        _summarizing = False


def maybe_summarize(model):
    """Fold older turns away if there are too many.

    Returns immediately. The work happens on a worker thread, so the
    turn that happens to tip the count over the limit isn't the one
    that pays for it.
    """
    global _summarizing

    with _lock:
        if _summarizing or len(_data.get("messages", [])) <= MAX_RAW_MESSAGES:
            return

        _summarizing = True

    threading.Thread(target=_fold, args=(model,), daemon=True).start()
