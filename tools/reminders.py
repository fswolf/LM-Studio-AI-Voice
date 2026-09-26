"""Scheduling: reminders that speak once, alarms that keep going.

One free-text field for the time rather than a number, because
the model is good at repeating "next friday at 4" and bad at
turning it into minutes. timeutil.parse_when does the maths.
"""
import reminders
import timeutil

from . import tool


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------
# One tool, one free-text field. This used to be two tools - one taking
# minutes, one taking a clock time - because small models fill in a
# polymorphic schema badly. That is still true, but the fix was the
# wrong way round: the model was being asked to *translate* ("in two
# days" -> 2880 minutes), which is exactly the arithmetic it gets wrong.
#
# Now it repeats the user's own timing words and timeutil.parse_when()
# does the work, so there is nothing left to choose between.
@tool(
    "set_reminder",
    "Schedule a reminder. Pass the user's own timing words through "
    "unchanged - 'in ten minutes', 'tomorrow at 9', 'next friday', "
    "'every morning at 8', 'tonight'. Never convert them to a number "
    "and never calculate a date yourself.",
    {
        "text": {
            "type": "string",
            "description": "What to remind the user about, in plain words, "
                           "without 'remind me to'.",
        },
        "when": {
            "type": "string",
            "description": "When it should fire, exactly as the user said "
                           "it. Repeats are fine: 'every 30 minutes', "
                           "'every monday at 9', 'daily at 7:30'.",
        },
    },
    required=("text", "when"),
)
def _set_reminder(text, when):
    due, repeat = timeutil.parse_when(when)

    if due is None:
        return (
            f"Couldn't read {when!r} as a time, so nothing was scheduled. "
            "Ask the user when they want it - do not guess a time."
        )

    return "Scheduled: " + reminders.describe(reminders.add(text, due, repeat))

# An alarm is the same schedule with a different delivery, so it is the
# same parser and the same store - but it gets its own tool rather than a
# `kind` parameter on set_reminder. A boolean flag is a choice the model
# has to make *after* it has already decided which tool to call, and that
# is the point at which small models stop reading the description. Two
# names it can match against the user's own word ("remind me" / "wake me")
# is a lookup, not a judgement.
@tool(
    "set_alarm",
    "Set a wake-up alarm. Use this instead of set_reminder when the user "
    "wants waking up or getting out of bed - 'wake me at 7', 'alarm for "
    "6:30am', 'get me up in the morning'. An alarm rings repeatedly with "
    "a tone until dismissed; a reminder is said once. Pass the user's own "
    "timing words through unchanged and never calculate a date yourself.",
    {
        "text": {
            "type": "string",
            "description": "Why they are getting up, in plain words - "
                           "'work', 'the gym', or just 'wake up' if they "
                           "didn't say.",
        },
        "when": {
            "type": "string",
            "description": "When it should ring, exactly as the user said "
                           "it. Repeats are fine: 'every weekday at 7:30', "
                           "'daily at 6'.",
        },
    },
    required=("text", "when"),
)
def _set_alarm(text, when):
    due, repeat = timeutil.parse_when(when, morning=True)

    if due is None:
        return (
            f"Couldn't read {when!r} as a time, so no alarm was set. "
            "Ask the user when they want waking - do not guess a time."
        )

    return "Alarm set: " + reminders.describe(
        reminders.add(text, due, repeat, kind="alarm")
    )

@tool(
    "list_reminders",
    "List the user's pending reminders and alarms with their due times.",
    {},
)
def _list_reminders():
    items = reminders.pending()

    if not items:
        return "No reminders pending."

    return "\n".join(
        f"{i}. {reminders.describe(r)}" for i, r in enumerate(items, 1)
    )

@tool(
    "cancel_reminder",
    "Cancel a pending reminder or alarm by its number from "
    "list_reminders. Call list_reminders first to find the number.",
    {
        "number": {
            "type": "integer",
            "description": "Position in the list, starting at 1.",
        },
    },
    required=("number",),
)
def _cancel_reminder(number):
    removed = reminders.cancel(int(number))

    if removed is None:
        return "No reminder with that number."

    return f"Cancelled: {removed['text']}"
