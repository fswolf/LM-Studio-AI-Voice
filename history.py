import json
import os
import requests

from datetime import datetime

from config import LM_URL, MAX_RAW_MESSAGES, SUMMARIZE_CHUNK, BASE_DIR

HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "conversation.json")

_data = {"summary": "", "messages": []}


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
    with open(HISTORY_FILE, "w") as f:
        json.dump(_data, f, indent=2)


def get_summary() -> str:
    return _data.get("summary", "")


def get_messages() -> list:
    """Recent raw messages, oldest first, as {"role","content"} dicts.

    Timestamps are stripped here since this is what gets sent straight
    to the model's API - use get_messages_full() if you need them.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in _data.get("messages", [])
    ]


def get_messages_full() -> list:
    """Same as get_messages(), but keeps the "timestamp" field on each
    message - use this for display purposes (e.g. showing timestamps
    in the UI) rather than feeding it to the model."""
    return list(_data.get("messages", []))


def add_message(role: str, content: str):
    _data.setdefault("messages", []).append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def _summarize_with_model(model, chunk):
    """Ask the model to compress a batch of older turns into a few sentences."""
    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)

    prompt = (
        "Summarize the key facts, decisions, and context from this "
        "conversation excerpt in a few concise sentences, written for an "
        "AI assistant's own future reference. Skip pleasantries, keep only "
        "information worth remembering.\n\n" + convo_text
    )

    response = requests.post(
        LM_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return response.json()["choices"][0]["message"]["content"].strip()


def maybe_summarize(model):
    """
    If the raw message list has grown past MAX_RAW_MESSAGES, fold the
    oldest SUMMARIZE_CHUNK of them into the running summary and drop
    them from the raw list. This call blocks on an extra model request
    when it triggers, so there's a brief added delay right at that turn.
    """
    messages = _data.get("messages", [])
    if len(messages) <= MAX_RAW_MESSAGES:
        return

    chunk = messages[:SUMMARIZE_CHUNK]
    remaining = messages[SUMMARIZE_CHUNK:]

    new_piece = _summarize_with_model(model, chunk)

    existing = _data.get("summary", "")
    _data["summary"] = (existing + "\n" + new_piece).strip() if existing else new_piece
    _data["messages"] = remaining