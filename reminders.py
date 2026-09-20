"""Reminders.

The division of labour that made the original work is kept: the model
only *extracts* what you said - a number, a unit, a clock time - and all
date arithmetic happens here in Python. Models are bad at "what time is
it in 90 minutes" and fine at "the user said 90 minutes".

What changed:

  * Failures are visible. The old version returned silently on bad JSON,
    so a reminder you thought was set simply never existed.
  * Absolute times ("at 5pm", "tomorrow at 9") alongside durations.
  * Reminders are removed only after delivery succeeds, so an LM Studio
    hiccup can't swallow one.
  * The scanner sleeps until the next reminder is actually due instead
    of waking on a fixed interval, so "in one minute" means one minute.
  * Repeats.
"""
import json
import os
import re
import threading
import time
import uuid

import requests
from datetime import datetime, timedelta

import timeutil

from config import (
    LM_URL,
    AGENT_NAME,
    REMINDERS_ENABLED,
    REMINDER_CHECK_INTERVAL_SECONDS,
    BASE_DIR,
)

REMINDERS_DIR = os.path.join(BASE_DIR, "reminders")
REMINDERS_FILE = os.path.join(REMINDERS_DIR, "reminders.json")

MAX_DELIVERY_ATTEMPTS = 3

_lock = threading.Lock()
_reminders = []
# Set whenever the schedule changes, so the scanner recomputes its sleep
# instead of napping through a reminder added a second ago.
_wake = threading.Event()

_UNIT_MAP = {
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
}

# Deliberately wide. A false positive costs one cheap model call that
# answers NONE; a false negative means the reminder silently never
# existed, which is the failure people actually notice.
_TRIGGERS = re.compile(
    r"\b("
    r"remind|reminder|remember to|don'?t let me forget|forget to|"
    r"wake me|nudge me|ping me|tell me to|let me know|alarm|timer|"
    r"in \d+\s*(second|sec|minute|min|hour|hr|day|week)s?|"
    r"in (a|an|half)\s+(second|minute|hour|day|week)|"
    r"at \d{1,2}([:.]\d{2})?\s*(am|pm|o'?clock)|"
    r"every (morning|evening|night|day|hour|week|\d+)|"
    r"tomorrow|tonight|later"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load():
    global _reminders

    if not os.path.exists(REMINDERS_FILE):
        _reminders = []
        return

    try:
        with open(REMINDERS_FILE, "r") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        _reminders = []
        return

    # Tolerate records written by the older version, which had no ids.
    for item in loaded:
        item.setdefault("id", uuid.uuid4().hex[:8])
        item.setdefault("attempts", 0)
        item.setdefault("repeat", None)

    _reminders = loaded


def save():
    os.makedirs(REMINDERS_DIR, exist_ok=True)

    with _lock:
        snapshot = list(_reminders)

    try:
        with open(REMINDERS_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)
    except OSError:
        pass  # best effort; a failed save shouldn't kill the turn


def pending():
    """Everything scheduled, soonest first."""
    with _lock:
        items = list(_reminders)

    return sorted(items, key=lambda r: r.get("due_at", ""))


def count():
    with _lock:
        return len(_reminders)


def cancel(index):
    """Cancel by 1-based position as shown by pending(). Returns the
    removed reminder, or None."""
    items = pending()

    if index < 1 or index > len(items):
        return None

    target = items[index - 1]

    with _lock:
        _reminders[:] = [r for r in _reminders if r.get("id") != target.get("id")]

    save()
    _wake.set()

    return target


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def human_delta(seconds):
    """Kept for callers that only have a number of seconds."""
    return timeutil.relative(datetime.now() + timedelta(seconds=seconds))


def describe(reminder):
    """'tomorrow 09:00 - take the bins out (in 18 hours)'.

    Named days rather than ISO timestamps, because this string is read
    aloud as often as it is printed, and "two-oh-two-six dash zero
    nine" is not a time anybody wants spoken at them.
    """
    try:
        due = datetime.fromisoformat(reminder["due_at"])
    except (KeyError, ValueError):
        return reminder.get("text", "?")

    now = datetime.now()
    repeat = reminder.get("repeat")
    suffix = f" (repeats {timeutil.describe_repeat(repeat)})" if repeat else ""

    return "{} - {} ({}){}".format(
        timeutil.friendly(due, now), reminder.get("text", "?"),
        timeutil.relative(due, now), suffix,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def looks_like_reminder(text):
    """Does this sound like it wants scheduling?

    Deliberately wide. A false positive costs one cheap model call that
    answers NONE; a false negative means the reminder silently never
    existed, which is the failure people actually notice.
    """
    return bool(_TRIGGERS.search(text or ""))


# Older private name, still used inside this module.
_looks_like_reminder = looks_like_reminder


_PROMPT = """You extract reminder requests. Do NOT do any arithmetic and do
NOT convert times - only report what was said.

Reply with ONLY one of these:

NONE
  - the message does not ask to be reminded of anything later.

{"kind":"in","amount":<number>,"unit":"seconds|minutes|hours|days|weeks","text":"<what to remind about>"}
  - a delay was given, e.g. "in 10 minutes", "in an hour" -> amount 1, unit hours.

{"kind":"at","time":"HH:MM","day":"today|tomorrow","text":"<what to remind about>"}
  - a clock time was given. Use 24-hour time. "5pm" -> "17:00". "tonight"
    with no time -> "20:00" today. "tomorrow morning" -> "09:00" tomorrow.

{"kind":"every","amount":<number>,"unit":"minutes|hours|days","text":"<what>"}
  - a repeating interval, e.g. "every 30 minutes".

{"kind":"daily","time":"HH:MM","text":"<what>"}
  - repeats at the same clock time each day, e.g. "every morning at 8".

The text field is the thing to be reminded about, in plain words, without
"remind me to".

Message: %s"""


def _parse_json(raw):
    """Pull a JSON object out of whatever the model wrapped it in."""
    cleaned = raw.strip().strip("`").strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def schedule_from(data, now=None):
    """Turn the keyword extractor's fields into (due_at, repeat).

    Only the fallback path reaches this now - with tools available the
    model hands a phrase straight to timeutil.parse_when() instead of
    choosing a "kind". The fields are reassembled into a phrase here so
    there is exactly one implementation of the date arithmetic in the
    app, and it is the one with tests.
    """
    now = now or datetime.now()
    kind = str(data.get("kind", "")).strip().lower()
    text = str(data.get("text", "")).strip()

    if not text:
        return None, None

    if kind in ("in", "every"):
        prefix = "every" if kind == "every" else "in"
        phrase = f"{prefix} {data.get('amount')} {data.get('unit', '')}"
    elif kind == "at":
        phrase = "{} at {}".format(
            str(data.get("day", "today")).strip().lower() or "today",
            data.get("time"),
        )
    elif kind == "daily":
        phrase = f"daily at {data.get('time')}"
    else:
        return None, None

    return timeutil.parse_when(phrase, now)


def add(text, due, repeat):
    reminder = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "due_at": due.isoformat(timespec="seconds"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repeat": repeat,
        "attempts": 0,
    }

    with _lock:
        _reminders.append(reminder)

    save()
    _wake.set()

    return reminder


def _extract_reminder(model, text):
    import ui

    try:
        response = requests.post(
            LM_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT % text}],
            },
            timeout=60,
        )
        raw = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        ui.add_message("system", f"Couldn't check that for a reminder: {e}")
        return

    if raw.upper().startswith("NONE"):
        return  # not a reminder; nothing to report

    data = _parse_json(raw)

    if data is None:
        ui.add_message(
            "system",
            "That looked like a reminder but I couldn't read a time out of "
            "it - try something like \"remind me in 10 minutes to stretch\".",
        )
        return

    due, repeat = schedule_from(data)

    if due is None:
        ui.add_message(
            "system",
            "That looked like a reminder but the time didn't make sense - "
            "try \"in 10 minutes\" or \"at 17:30\".",
        )
        return

    reminder = add(str(data.get("text", "")).strip(), due, repeat)

    # The whole point of the rewrite: say so, out loud, every time.
    ui.add_message("system", f"Reminder set - {describe(reminder)}")


def extract_in_background(model, text):
    if not REMINDERS_ENABLED or not _looks_like_reminder(text):
        return

    threading.Thread(
        target=_extract_reminder, args=(model, text), daemon=True
    ).start()


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------
def _due_now():
    now = datetime.now()
    due = []

    with _lock:
        for reminder in _reminders:
            try:
                if datetime.fromisoformat(reminder["due_at"]) <= now:
                    due.append(reminder)
            except (KeyError, ValueError):
                continue

    return due


def _retire(reminder, delivered):
    """Reschedule a repeating reminder, or drop a one-off.

    Only called once delivery actually worked - the old code removed
    reminders before attempting delivery, so a failure lost them.

    The next time comes from timeutil.next_occurrence(), which rebuilds
    a daily reminder from the wall clock instead of adding 24 hours.
    That is the difference between "every morning at 8" staying at 8
    and quietly becoming 7 for the winter. It also understands weekly
    repeats, which the old arithmetic here could not express.

    It is measured from when this one was *due* rather than from now,
    so a reminder delivered a few minutes late doesn't drag the whole
    schedule later every time it fires.
    """
    repeat = reminder.get("repeat")

    if delivered and repeat:
        try:
            anchor = datetime.fromisoformat(reminder["due_at"])
        except (KeyError, ValueError):
            anchor = datetime.now()

        reminder["due_at"] = timeutil.next_occurrence(
            repeat, anchor
        ).isoformat(timespec="seconds")
        reminder["attempts"] = 0
        save()
        return

    with _lock:
        _reminders[:] = [r for r in _reminders if r.get("id") != reminder.get("id")]

    save()


def _seconds_until_next():
    items = pending()

    if not items:
        return REMINDER_CHECK_INTERVAL_SECONDS

    try:
        due = datetime.fromisoformat(items[0]["due_at"])
    except (KeyError, ValueError):
        return REMINDER_CHECK_INTERVAL_SECONDS

    remaining = (due - datetime.now()).total_seconds()

    return max(0.5, min(remaining, REMINDER_CHECK_INTERVAL_SECONDS))


def _deliver(model, texts, missed=False):
    """Hand the reminder(s) to Luna so she raises them in her own voice."""
    import llm
    import ui
    from speech import speak
    import state

    if len(texts) == 1:
        body = f'"{texts[0]}"'
    else:
        body = "; ".join(f'"{t}"' for t in texts)

    prefix = (
        "These reminders came due while the app was closed"
        if missed else
        "This is a reminder you set earlier"
    )

    trigger = (
        f"({prefix}: {body}. Bring "
        f"{'them' if len(texts) > 1 else 'it'} up now, naturally, in your "
        "own voice.)"
    )

    ui.set_status("Reminder due...")
    answer = llm.ask(trigger, model)
    ui.add_message(AGENT_NAME.lower(), answer)

    if not state.stop_speaking:
        ui.set_status("Speaking...")
        speak(answer)

    ui.set_status("Idle")


def run_scanner(model):
    import ui

    load()

    if _reminders:
        ui.add_message("system", f"Loaded {len(_reminders)} pending reminder(s).")

    first_pass = True

    while True:
        if REMINDERS_ENABLED:
            due = _due_now()

            if due:
                # Anything already overdue at startup gets delivered as one
                # message rather than N separate model calls in a row.
                batch = due if (first_pass and len(due) > 1) else due[:1]

                try:
                    _deliver(model, [r["text"] for r in batch], missed=first_pass)

                    for reminder in batch:
                        _retire(reminder, delivered=True)
                except Exception as e:
                    for reminder in batch:
                        reminder["attempts"] = reminder.get("attempts", 0) + 1

                        if reminder["attempts"] >= MAX_DELIVERY_ATTEMPTS:
                            ui.add_message(
                                "system",
                                f"Giving up on reminder \"{reminder['text']}\" "
                                f"after {MAX_DELIVERY_ATTEMPTS} tries: {e}",
                            )
                            _retire(reminder, delivered=False)

                    save()
                    ui.set_status("Idle")

            first_pass = False

        # Sleep until the next one is actually due. _wake fires early when
        # a reminder is added or cancelled.
        _wake.wait(timeout=_seconds_until_next())
        _wake.clear()
