import json
import os
import threading
import time

import requests
from datetime import datetime, timedelta

from config import LM_URL, AGENT_NAME, REMINDERS_ENABLED, REMINDER_CHECK_INTERVAL_SECONDS, BASE_DIR

REMINDERS_DIR = os.path.join(BASE_DIR, "reminders")
REMINDERS_FILE = os.path.join(REMINDERS_DIR, "reminders.json")

_lock = threading.Lock()
_reminders = []  # list of {"due_at": iso string, "text": "..."}


def load():
    global _reminders
    if os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE, "r") as f:
                _reminders = json.load(f)
        except (json.JSONDecodeError, OSError):
            _reminders = []
    else:
        _reminders = []


def save():
    os.makedirs(REMINDERS_DIR, exist_ok=True)
    with _lock:
        with open(REMINDERS_FILE, "w") as f:
            json.dump(_reminders, f, indent=2)


def _looks_like_reminder(text: str) -> bool:
    return "remind" in text.lower()


_UNIT_MAP = {
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "day": "days", "days": "days",
}


def _extract_reminder(model, text):
    now = datetime.now()
    prompt = (
        "Does this message ask to be reminded of something at a future "
        "time? If yes, reply with ONLY strict JSON extracting exactly "
        "what was said - do NOT calculate or convert anything, just "
        "pull out the number, the time unit, and the reminder text as "
        'stated, e.g. {"amount": 5, "unit": "hours", "text": "check the '
        'oven"}. unit must be one of: seconds, minutes, hours, days. '
        "If the message does not ask for a reminder, reply with exactly "
        "NONE.\n\n"
        f"Message: {text}"
    )
    try:
        response = requests.post(
            LM_URL,
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        result = response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return

    if result.upper() == "NONE":
        return

    try:
        cleaned = result.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        amount = float(data["amount"])
        unit = _UNIT_MAP.get(str(data["unit"]).strip().lower())
        reminder_text = str(data["text"]).strip()
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return  # model didn't return usable JSON - fail silently

    if unit is None or not reminder_text:
        return  # unrecognized unit or empty text - don't guess, just skip

    due_at = now + timedelta(**{unit: amount})  # the only math happens here, in code

    with _lock:
        _reminders.append({"due_at": due_at.isoformat(), "text": reminder_text})
    save()


def extract_in_background(model, text):
    if not REMINDERS_ENABLED or not _looks_like_reminder(text):
        return
    threading.Thread(target=_extract_reminder, args=(model, text), daemon=True).start()


def _pop_due():
    now = datetime.now()
    due = []
    remaining = []
    with _lock:
        for r in _reminders:
            try:
                due_at = datetime.fromisoformat(r["due_at"])
            except (KeyError, ValueError):
                continue
            if due_at <= now:
                due.append(r)
            else:
                remaining.append(r)
        _reminders[:] = remaining
    if due:
        save()
    return due


def run_scanner(model):
    """
    Background loop: on start, loads reminders.json (so anything saved
    from a previous session isn't lost/ignored) and immediately checks
    for reminders that are already due - e.g. one that came due while
    the app was closed. After that, it checks again every
    REMINDER_CHECK_INTERVAL_SECONDS.

    Each due reminder is processed inside its own try/except - a
    single failure (LM Studio error, context overflow, TTS hiccup,
    etc.) gets logged and skipped rather than raising out of the loop.
    Without this, one bad reminder would kill this entire background
    thread silently - no more reminders would ever fire for the rest
    of the session, with nothing on screen indicating why.
    """
    import ui
    from speech import speak
    import llm

    load()  # pick up whatever was saved from the last session
    if _reminders:
        print(f"[reminders] Loaded {len(_reminders)} pending reminder(s) from {REMINDERS_FILE}")
    else:
        print("[reminders] No pending reminders.")

    while True:
        if REMINDERS_ENABLED:
            for r in _pop_due():
                try:
                    ui.set_status("Reminder due...")
                    trigger_text = (
                        f'(This is a reminder you set earlier: "{r["text"]}". '
                        "Bring it up now, naturally, in your own voice.)"
                    )
                    answer = llm.ask(trigger_text, model)
                    ui.add_message(AGENT_NAME.lower(), answer)
                    ui.set_status("Speaking...")
                    speak(answer)
                    ui.set_status("Idle")
                except Exception as e:
                    print(f"[reminders] Failed to deliver reminder {r!r}: {e}")
                    ui.set_status(f"Reminder failed: {e}")

        time.sleep(REMINDER_CHECK_INTERVAL_SECONDS)
