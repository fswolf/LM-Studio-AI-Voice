"""Telling the time, and working out how far away something is.

Both exist because a 9B model will not admit it cannot do
arithmetic - it will give you a confident wrong date instead. The
rule the descriptions enforce is: never calculate, always ask.
"""
import timeutil

from . import tool


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
# The reply is a block rather than a bare clock reading on purpose. A
# model that only knows "14:32" still cannot answer "what's the date on
# Friday" - and a small one will not admit that, it will invent a date.
# Everything it might need is spelled out so it never has to calculate.
@tool(
    "get_datetime",
    "Get the current date, time, weekday and timezone. Call this before "
    "answering anything about what day or time it is, how long until "
    "something, or what the date will be - never work it out yourself.",
    {},
)
def _get_datetime():
    return timeutil.describe_now()

@tool(
    "time_until",
    "Work out how far away a date or time is, or what date it falls on. "
    "Pass the user's own words - 'christmas day', 'next friday', "
    "'december 25', '18:00'. Use this instead of counting days yourself.",
    {
        "when": {
            "type": "string",
            "description": "The date or time to measure to, in plain words.",
        },
    },
    required=("when",),
)
def _time_until(when):
    moment, _ = timeutil.parse_when(when)

    if moment is None:
        return (
            f"Couldn't read {when!r} as a date or time. Ask the user to "
            "say it another way - do not guess."
        )

    return "{} falls on {} - that is {}.".format(
        when, moment.strftime("%A %d %B %Y at %H:%M"), timeutil.relative(moment)
    )
