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

import config
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


def is_alarm(reminder):
    return (reminder or {}).get("kind") == "alarm"


def describe(reminder, now=None):
    """'tomorrow 09:00 - take the bins out (in 18 hours)'.

    Named days rather than ISO timestamps, because this string is read
    aloud as often as it is printed, and "two-oh-two-six dash zero
    nine" is not a time anybody wants spoken at them.

    `now` is for describing a reminder as it read at some other moment -
    repairing history needs "in 5 minutes" as it was then, not as it
    would be now.
    """
    try:
        due = datetime.fromisoformat(reminder["due_at"])
    except (KeyError, ValueError):
        return reminder.get("text", "?")

    now = now or datetime.now()
    repeat = reminder.get("repeat")
    suffix = f" (repeats {timeutil.describe_repeat(repeat)})" if repeat else ""

    # The word "alarm" goes in the sentence rather than in a symbol or a
    # bracketed tag, because this string gets spoken as often as printed
    # and "left square bracket alarm" is not a thing anyone wants to hear.
    label = "alarm: " if is_alarm(reminder) else ""

    return "{} - {}{} ({}){}".format(
        timeutil.friendly(due, now), label, reminder.get("text", "?"),
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
    phrase = phrase_from(data)

    if phrase is None:
        return None, None

    return timeutil.parse_when(phrase, now or datetime.now())


def phrase_from(data):
    """The extractor's fields, back as the words a person would say.

    Split out of schedule_from() because the phrase is worth having on
    its own: it is what set_reminder's `when` argument would have been
    had the model called the tool, and recording it is what turns a
    rescued reminder into an example for next time.
    """
    kind = str(data.get("kind", "")).strip().lower()

    if not str(data.get("text", "")).strip():
        return None

    if kind in ("in", "every"):
        prefix = "every" if kind == "every" else "in"

        return f"{prefix} {data.get('amount')} {data.get('unit', '')}"

    if kind == "at":
        return "{} at {}".format(
            str(data.get("day", "today")).strip().lower() or "today",
            data.get("time"),
        )

    if kind == "daily":
        return f"daily at {data.get('time')}"

    return None


def add(text, due, repeat, kind="reminder"):
    """Schedule something. `kind` decides how it gets delivered, not how
    it gets scheduled - an alarm is a reminder that wakes you up, so it
    reuses all of this rather than growing a second scanner."""
    reminder = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "due_at": due.isoformat(timespec="seconds"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repeat": repeat,
        "attempts": 0,
    }

    if kind != "reminder":
        reminder["kind"] = kind

    with _lock:
        _reminders.append(reminder)

    save()
    _wake.set()

    return reminder


def _extract_reminder(model, text):
    import ui

    try:
        raw, data = _extract_fields(model, text)
    except Exception as e:
        ui.add_message("system", f"Couldn't check that for a reminder: {e}")
        return

    if raw.upper().startswith("NONE"):
        return  # not a reminder; nothing to report

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

    # And write it into history as the tool call it should have been.
    # This path only runs when the model *didn't* call set_reminder, so
    # left alone it teaches the opposite of what we want: another turn
    # of prose that happened to work out. Recorded, it becomes the
    # example that stops the next one needing rescuing.
    _record_as_tool_call(data, reminder)


def _record_as_tool_call(data, reminder):
    import history

    phrase = phrase_from(data)

    if not phrase:
        return

    try:
        history.record_tool_use(
            "set_reminder",
            json.dumps(
                {"text": str(data.get("text", "")).strip(), "when": phrase}
            ),
            f"Scheduled: {describe(reminder)}",
        )
    except Exception:
        pass  # bookkeeping; never worth losing a set reminder over


def _extract_fields(model, text, timeout=60):
    """The extractor's answer, as (raw_reply, fields_or_None).

    The raw reply comes back too because "NONE" and "that wasn't JSON"
    mean different things to the caller - one is a non-reminder and
    says nothing, the other is worth complaining about.
    """
    response = requests.post(
        LM_URL,
        json={"model": model,
              "messages": [{"role": "user", "content": _PROMPT % text}]},
        timeout=timeout,
    )
    raw = response.json()["choices"][0]["message"]["content"].strip()

    if raw.upper().startswith("NONE"):
        return raw, None

    return raw, _parse_json(raw)


# The app writes this itself when a reminder comes due, so a turn that
# starts with it is a delivery, not a request - repairing it would
# record a reminder that was never asked for.
_DELIVERY = ("(this is a reminder you set earlier",
             "(these reminders came due")


def repair_history(model, report=None):
    """Rewrite past reminder turns as the tool calls they really were.

    Every reminder request answered before tool use was recorded looks,
    in history, like a turn where the right thing to do was to say
    "Got it, setting that for you!" and call nothing. It is the most
    convincing possible demonstration of the exact failure, and with
    several of them in the window a single counter-example does not
    come close to outvoting them.

    They are not lies, though - the keyword fallback really did schedule
    those reminders, it just left no trace. So this reconstructs the
    trace: the user's own words back through the same extractor, the
    same date arithmetic, but anchored to when the turn happened rather
    than now, so "in 5 minutes" means five minutes after he said it.

    Nothing new is scheduled and nothing she said is altered. The only
    change is that a turn which used a tool now says so.
    """
    import history

    def say(line):
        if report:
            report(line)

    messages = history.get_messages_full()
    repaired = 0

    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or message.get("tools"):
            continue

        if index == 0 or messages[index - 1].get("role") != "user":
            continue

        asked = str(messages[index - 1].get("content", ""))

        if not _looks_like_reminder(asked):
            continue

        if asked.strip().lower().startswith(_DELIVERY):
            continue

        try:
            _raw, data = _extract_fields(model, asked)
        except Exception as e:
            say(f"  couldn't re-read {asked[:40]!r}: {e}")
            continue

        if not data:
            continue

        # The turn's own timestamp is the reference, so the reminder is
        # described as it was described then.
        try:
            then = datetime.fromisoformat(str(message.get("timestamp")))
        except (TypeError, ValueError):
            then = datetime.now()

        due, repeat = schedule_from(data, then)
        phrase = phrase_from(data)

        if due is None or not phrase:
            continue

        text = str(data.get("text", "")).strip()
        shown = describe(
            {"text": text, "due_at": due.isoformat(timespec="seconds"),
             "repeat": repeat},
            now=then,
        )

        if history.record_tool_use(
            "set_reminder",
            json.dumps({"text": text, "when": phrase}),
            f"Scheduled: {shown}",
            index=index,
        ):
            repaired += 1
            say(f"  {asked[:46]:48} -> set_reminder({phrase})")

    return repaired


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


# How late an alarm can be and still be worth ringing. Judged from the
# due time rather than from "is this the first scan", because starting
# the app at 07:29 with a 07:30 alarm should still wake you up, and the
# first scan is exactly when that happens.
ALARM_GRACE_MINUTES = 10


def _deliver_alarm(model, item, stale=False):
    """Ring, and honour a snooze by putting it back a few minutes."""
    import alarm
    import ui

    try:
        late = (datetime.now()
                - datetime.fromisoformat(item["due_at"])).total_seconds() / 60
    except (KeyError, ValueError):
        late = 0

    if not config.ALARMS_ENABLED:
        # Not "ignore it" - you still wanted telling, you just didn't
        # want the room woken up. Falls back to the reminder delivery,
        # which says it once at the normal volume.
        _deliver(model, [item.get("text", "wake up")])

        return

    if late > ALARM_GRACE_MINUTES:
        # It came due while the app was shut. Waking someone at 11am for
        # a 7am alarm is worse than missing it.
        ui.add_message(
            "system",
            "Alarm \"{}\" was due {} - too late to ring, skipping it.".format(
                item.get("text", "?"),
                timeutil.relative(datetime.fromisoformat(item["due_at"])),
            ),
        )

        return

    minutes = alarm.ring(model, item)

    if minutes:
        due = datetime.now() + timedelta(minutes=minutes)
        add(item.get("text", "wake up"), due, None, kind="alarm")


def run_scanner(model):
    import ui

    load()

    if _reminders:
        alarms = sum(1 for r in _reminders if is_alarm(r))
        plain = len(_reminders) - alarms
        parts = []

        if plain:
            parts.append(f"{plain} reminder(s)")

        if alarms:
            parts.append(f"{alarms} alarm(s)")

        ui.add_message("system", "Loaded " + " and ".join(parts) + ".")

    first_pass = True

    while True:
        if REMINDERS_ENABLED:
            due = _due_now()

            if due:
                # An alarm always goes on its own. Batching it with
                # reminders would bury the thing that is supposed to be
                # impossible to ignore inside a list of things that
                # aren't - and a missed alarm is not worth ringing about
                # hours later, so an overdue one at startup is retired
                # rather than fired.
                alarms = [r for r in due if is_alarm(r)]

                if alarms:
                    batch = alarms[:1]
                elif first_pass and len(due) > 1:
                    batch = due
                else:
                    batch = due[:1]

                try:
                    if alarms:
                        _deliver_alarm(model, batch[0])
                    else:
                        _deliver(model, [r["text"] for r in batch],
                                 missed=first_pass)

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
