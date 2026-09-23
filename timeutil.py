"""Everything that knows what time it is.

Three different places used to answer "when": tools.get_datetime built
its own string, reminders.py did its own clock arithmetic, and llm.py
stamped history with isoformat(). They agreed with each other by
accident, and a small model saw three different formats for the same
concept.

Two ideas run through this whole file:

  * The model never does arithmetic. It repeats the words the user
    actually said - "next tuesday at 9", "in half an hour" - and
    parse_when() turns that into a datetime here, in Python. Asking a
    9B model for "2 days in minutes" is asking it to be wrong.

  * Time shown to the model is relative. "17 minutes ago" is something
    a small model reads correctly; "2026-09-19T14:32:07" is something
    it has to subtract, badly, before it can use it.

Everything here is naive local time, the same as datetime.now(). Making
it timezone-aware would mean every stored reminder written before the
change compares as a different type, for no benefit on a machine that
lives in one timezone. The timezone name is reported to the model as
information, not carried around as state - and daily repeats recompute
from the wall clock rather than adding 24 hours, which is where the
DST bug would actually have bitten.
"""
import contextvars
import re
import time as _time
from datetime import datetime, timedelta

# Whether a bare hour with no am/pm should read as morning. A context
# variable rather than a module global so two threads - the reminder
# scanner and whoever is typing - can't read each other's setting.
_MORNING = contextvars.ContextVar("morning", default=False)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "weds": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

# Named sets of days, because "every weekday at 7" is the single most
# common alarm anyone sets and it is not a weekly repeat - it fires five
# times a week, which {"weekday": n} cannot express.
DAY_SETS = {
    "weekday": (0, 1, 2, 3, 4),
    "weekdays": (0, 1, 2, 3, 4),
    "workday": (0, 1, 2, 3, 4),
    "workdays": (0, 1, 2, 3, 4),
    "weekend": (5, 6),
    "weekends": (5, 6),
}

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

UNITS = {
    "second": "seconds", "seconds": "seconds", "sec": "seconds",
    "secs": "seconds", "s": "seconds",
    "minute": "minutes", "minutes": "minutes", "min": "minutes",
    "mins": "minutes", "m": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "h": "hours",
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "wk": "weeks", "wks": "weeks",
    "month": "months", "months": "months",
    "year": "years", "years": "years",
}

# Spelled-out counts. People say "in a couple hours" far more often than
# they say "in 2 hours", and a model transcribing speech passes it
# through verbatim.
WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fortyfive": 45, "sixty": 60, "ninety": 90,
    "half": 0.5, "couple": 2, "few": 3, "several": 3, "dozen": 12,
}

# Vague times of day, resolved to something specific. Picking a concrete
# hour is better than refusing: "remind me tonight" should produce a
# reminder at 20:00, not an error message.
DAY_PARTS = {
    "morning": (9, 0),
    "noon": (12, 0),
    "midday": (12, 0),
    "lunch": (12, 30),
    "lunchtime": (12, 30),
    "afternoon": (14, 0),
    "evening": (18, 0),
    "dinner": (18, 30),
    "dinnertime": (18, 30),
    "tonight": (20, 0),
    "night": (20, 0),
    "bed": (22, 0),
    "bedtime": (22, 0),
    "midnight": (0, 0),
}


# ---------------------------------------------------------------------------
# Now
# ---------------------------------------------------------------------------
def now():
    return datetime.now()


def timezone_label():
    """'America/New_York (UTC-04:00)', or as close as the machine knows.

    time.tzname only gives the abbreviation ('EDT'), which is ambiguous
    worldwide, so the IANA name is read from the environment or
    /etc/localtime when it's there.
    """
    import os

    name = os.environ.get("TZ") or ""

    if not name:
        try:
            link = os.path.realpath("/etc/localtime")

            if "zoneinfo/" in link:
                name = link.split("zoneinfo/", 1)[1]
        except OSError:
            pass

    if not name:
        name = _time.tzname[_time.daylight and _time.localtime().tm_isdst > 0]

    offset = -(_time.altzone if _time.localtime().tm_isdst > 0 else _time.timezone)
    sign = "+" if offset >= 0 else "-"
    hours, minutes = divmod(abs(offset) // 60, 60)

    return f"{name} (UTC{sign}{hours:02d}:{minutes:02d})"


def day_part(moment=None):
    """Which part of the day it is, in the words a person would use."""
    hour = (moment or now()).hour

    if hour < 5:
        return "the middle of the night"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"

    return "night"


def describe_now(moment=None):
    """The block handed to the model by get_datetime().

    Deliberately more than the bare clock: a model that only has
    "14:32" still can't answer "what day is it next Friday", and if it
    tries it will guess. Everything it might need to reason about a
    date is spelled out so it never has to compute one.
    """
    moment = moment or now()
    tomorrow = moment + timedelta(days=1)

    return "\n".join([
        moment.strftime("%A %d %B %Y, %H:%M") + f" ({day_part(moment)})",
        f"Timezone: {timezone_label()}",
        f"Today is {moment.strftime('%A')}; tomorrow is "
        f"{tomorrow.strftime('%A %d %B')}.",
        f"ISO date {moment.strftime('%Y-%m-%d')}, week "
        f"{moment.isocalendar()[1]} of {moment.year}.",
    ])


# ---------------------------------------------------------------------------
# Formatting for humans (and for small models, which read the same way)
# ---------------------------------------------------------------------------
def relative(target, reference=None):
    """'in 20 minutes', '3 hours ago', 'just now'.

    Rounded and worded, not precise - this exists so nobody, model or
    person, has to subtract two timestamps in their head.
    """
    reference = reference or now()
    seconds = (target - reference).total_seconds()
    future = seconds >= 0
    seconds = abs(seconds)

    if seconds < 45:
        return "in a moment" if future else "just now"

    if seconds < 90:
        amount = "a minute"
    elif seconds < 3600:
        amount = f"{round(seconds / 60)} minutes"
    elif seconds < 5400:
        amount = "an hour"
    elif seconds < 21600:
        # Half hours are worth saying up to about six hours out, and
        # noise after that - nobody needs "in 12.5 hours".
        hours = round(seconds / 1800) / 2
        amount = f"{hours:g} hours"
    elif seconds < 86400:
        amount = f"{round(seconds / 3600)} hours"
    elif seconds < 129600:
        # Up to a day and a half. Not 48h: "in 2 days" stored to
        # second precision measures a hair under two days by the time
        # it's read back, and reading out as "a day" was just wrong.
        amount = "a day"
    elif seconds < 1209600:
        amount = f"{round(seconds / 86400)} days"
    elif seconds < 5184000:
        amount = f"{round(seconds / 604800)} weeks"
    else:
        amount = f"{round(seconds / 2592000)} months"

    return f"in {amount}" if future else f"{amount} ago"


def friendly(target, reference=None):
    """'today 14:32', 'tomorrow 09:00', 'Monday 09:00', 'Mon 22 Sep 09:00'.

    Named days inside the coming week, because "Monday 9am" is how the
    reminder was asked for and how it should be read back - and because
    it survives being spoken aloud, which '2026-09-22T09:00' does not.
    """
    reference = reference or now()
    days = (target.date() - reference.date()).days
    clock = target.strftime("%H:%M")

    if days == 0:
        return f"today {clock}"
    if days == 1:
        return f"tomorrow {clock}"
    if days == -1:
        return f"yesterday {clock}"
    if 2 <= days <= 6:
        return f"{target.strftime('%A')} {clock}"
    if -6 <= days <= -2:
        return f"last {target.strftime('%A')} {clock}"

    if target.year == reference.year:
        return target.strftime("%a %d %b ") + clock

    return target.strftime("%a %d %b %Y ") + clock


def stamp(target, reference=None):
    """The prefix on a history message: 'today 14:32, 17 minutes ago'.

    Both halves earn their place. The relative half is what the model
    actually reasons with; the absolute half is what lets it answer
    "what time did I say that" without another tool call.
    """
    reference = reference or now()

    return f"{friendly(target, reference)}, {relative(target, reference)}"


def describe_repeat(repeat):
    if not repeat:
        return ""

    if repeat.get("days"):
        days = sorted(set(repeat["days"]))
        at = repeat.get("at", "?")

        # "every weekday" rather than a five-name list, because this is
        # spoken aloud and nobody says Monday-Tuesday-Wednesday-Thursday-
        # Friday out loud.
        if days == [0, 1, 2, 3, 4]:
            return f"every weekday at {at}"

        if days == [5, 6]:
            return f"every weekend day at {at}"

        names = [WEEKDAY_NAMES[d] for d in days]

        return "every {} and {} at {}".format(
            ", ".join(names[:-1]), names[-1], at
        )

    if repeat.get("weekday") is not None:
        return f"every {WEEKDAY_NAMES[repeat['weekday']]} at {repeat.get('at', '?')}"

    if repeat.get("unit"):
        amount = repeat["amount"]
        amount = f"{amount:g}"
        unit = repeat["unit"]

        return f"every {amount} {unit}" if amount != "1" else f"every {unit[:-1]}"

    return f"every day at {repeat.get('at', '?')}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_FILLER = re.compile(
    r"^\s*(?:please\s+|remind\s+me\s+|set\s+(?:a\s+)?(?:reminder|alarm|timer)\s+"
    r"(?:for\s+|at\s+|in\s+)?|wake\s+me\s+(?:up\s+)?|ping\s+me\s+|"
    r"nudge\s+me\s+)+",
    re.IGNORECASE,
)


def _clean(phrase):
    text = str(phrase or "").strip().lower()
    text = _FILLER.sub("", text)
    text = re.sub(r"\bo'?clock\b", "", text)
    text = re.sub(r"[,.]+$", "", text)

    return re.sub(r"\s+", " ", text).strip()


def _number(token):
    """'3', 'three', 'a', 'half' -> a float, or None."""
    token = token.strip().lower()

    if token in WORD_NUMBERS:
        return float(WORD_NUMBERS[token])

    try:
        return float(token)
    except ValueError:
        return None


def _clock(text):
    """'5pm', '17:30', '9', '9.15am', 'half past 8' -> (hour, minute, vague).

    The third value is the interesting one: it says nobody specified am
    or pm, so the caller is free to shift the result by twelve hours if
    that lands somewhere more sensible. "Remind me at 8" said at two in
    the afternoon means tonight, not eighteen hours from now, and that
    single rule fixes most of the reminders that used to arrive half a
    day late.
    """
    text = text.strip().lower()

    match = re.match(r"^half\s+past\s+(\d{1,2})$", text)

    if match:
        return _bare(int(match.group(1)), 30)

    match = re.match(
        r"^(\d{1,2})(?:[:.h](\d{2}))?\s*(am|a\.m\.|pm|p\.m\.)?$", text
    )

    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").replace(".", "").rstrip("m") or None

    if minute > 59:
        return None

    if meridiem is None:
        return _bare(hour, minute)

    if meridiem == "p" and hour < 12:
        hour += 12
    elif meridiem == "a" and hour == 12:
        hour = 0

    return (hour, minute, False) if 0 <= hour <= 23 else None


def _bare(hour, minute):
    """A time with no am/pm attached. 13-23 can only be one thing; 1-12
    is a guess the caller may override."""
    if hour > 23:
        return None

    if hour >= 13:
        return hour, minute, False

    # An alarm inverts the default: "wake me at 6" has never once in the
    # history of alarm clocks meant six in the evening. Set for the whole
    # parse rather than passed down, because the hour is read six call
    # levels below parse_when and threading a flag through all of them to
    # reach one `if` is worse than this.
    if _MORNING.get():
        return hour, minute, True

    # The daytime reading is the better default on its own - "at 3"
    # means the afternoon - but it stays flagged as a guess.
    if 1 <= hour <= 6:
        return hour + 12, minute, True

    return hour, minute, True


def _hm(text):
    """_clock() for the callers that don't care whether it was a guess."""
    parsed = _clock(text)

    return parsed[:2] if parsed else None


def _time_or_part(text):
    """A clock time or a word like 'morning'. Returns (hour, minute, vague)."""
    if text in DAY_PARTS:
        hour, minute = DAY_PARTS[text]

        return hour, minute, False

    return _clock(text)


def _shift_pm(parsed, evening=False):
    """Push a guessed morning hour into the evening.

    Only applied where the surrounding words already say evening -
    "tonight at 11", "every night at 10" - because there 11 and 10
    cannot have meant the morning.
    """
    if not parsed or not evening:
        return parsed[:2] if parsed else None

    hour, minute, vague = parsed

    if vague and hour < 12:
        hour += 12

    return hour, minute


def _at_wall_clock(day, hour, minute):
    """A wall-clock time on a given date.

    Built by replacing fields rather than adding timedelta, so a daily
    08:00 reminder is still 08:00 the morning the clocks change instead
    of drifting to 07:00 or 09:00 for the rest of the year.
    """
    return datetime(day.year, day.month, day.day, hour, minute)


def next_weekday(reference, weekday, hour=9, minute=0, force_next=False):
    """The next <weekday> at that time, strictly in the future.

    "next friday" said on a Friday means the Friday after this one,
    which is what force_next is for.
    """
    ahead = (weekday - reference.weekday()) % 7
    candidate = _at_wall_clock(reference + timedelta(days=ahead), hour, minute)

    if ahead == 0 and (force_next or candidate <= reference):
        candidate += timedelta(days=7)
    elif force_next and ahead == 0:
        candidate += timedelta(days=7)

    return candidate


def next_in_days(reference, days, hour, minute, force_next=False):
    """The soonest of several weekdays at that time, strictly ahead.

    Just next_weekday() over a set and taking the nearest, which is all
    "every weekday at 7:30" needs - Friday's 7:30 rolls to Monday's on
    its own because each candidate is already pushed into the future.
    """
    candidates = [
        next_weekday(reference, day, hour, minute, force_next=force_next)
        for day in sorted(set(days))
    ]

    return min(candidates) if candidates else None


def _add_months(moment, count):
    """Calendar months, clamped - 31 January plus one month is 28/29
    February, not 3 March."""
    count = int(count)
    month = moment.month - 1 + count
    year = moment.year + month // 12
    month = month % 12 + 1

    for day in range(moment.day, 27, -1):
        try:
            return moment.replace(year=year, month=month, day=day)
        except ValueError:
            continue

    return moment.replace(year=year, month=month, day=min(moment.day, 28))


def _delta(amount, unit, reference):
    if unit == "months":
        return _add_months(reference, amount)

    if unit == "years":
        return _add_months(reference, amount * 12)

    return reference + timedelta(**{unit: amount})


# --- single points in time -------------------------------------------------
def _resolve_once(text, reference):
    """One datetime out of one phrase, or None."""
    if not text:
        return None

    # in 10 minutes / in an hour / in a couple of hours / 10 minutes from now
    match = re.match(
        r"^(?:in|after)?\s*(?:a\s+)?([\d.]+|[a-z-]+)\s*(?:of\s+)?"
        r"(?:an?\s+)?([a-z]+)\b(?:\s+from\s+now)?$",
        text,
    )

    if match:
        amount = _number(match.group(1))
        unit = UNITS.get(match.group(2))

        if amount is not None and unit and amount > 0:
            return _delta(amount, unit, reference)

    # "in half an hour" parses above; "half an hour" alone does too.
    # "in 1h30" / "in 90s" - shorthand people type.
    match = re.match(r"^(?:in\s+)?(\d+)\s*h(?:\s*(\d{1,2})\s*m?)?$", text)

    if match:
        return reference + timedelta(
            hours=int(match.group(1)), minutes=int(match.group(2) or 0)
        )

    # ISO date, optionally with a time
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[ t](.+))?$", text)

    if match:
        hour, minute = _hm(match.group(4) or "09:00") or (9, 0)

        try:
            return _at_wall_clock(
                datetime(int(match.group(1)), int(match.group(2)),
                         int(match.group(3))),
                hour, minute,
            )
        except ValueError:
            return None

    # today / tomorrow / tonight / this evening / in the morning, with an
    # optional time after them
    match = re.match(
        r"^(?:on\s+|this\s+|the\s+|in\s+the\s+)?"
        r"(today|tomorrow|tmrw|tonight|overnight|"
        + "|".join(DAY_PARTS) + r")"
        r"(?:\s+(?:at|around|about)?\s*(.+))?$",
        text,
    )

    if match:
        word, rest = match.group(1), match.group(2)
        day = reference
        evening = word in ("tonight", "overnight", "evening", "night",
                           "dinner", "dinnertime", "bed", "bedtime")

        if word in ("tomorrow", "tmrw"):
            day = reference + timedelta(days=1)
            hour, minute = 9, 0
        elif word in ("tonight", "overnight"):
            hour, minute = DAY_PARTS["tonight"]
        elif word == "today":
            hour, minute = DAY_PARTS.get(day_part(reference), (0, 0))
        else:
            hour, minute = DAY_PARTS[word]

        if rest:
            clock = _shift_pm(_time_or_part(rest), evening)

            if clock is None:
                return None

            hour, minute = clock

        resolved = _at_wall_clock(day, hour, minute)

        # "tonight" said at 23:00 means the small hours, not an hour ago.
        if resolved <= reference and word in ("today", "tonight", "overnight"):
            resolved += timedelta(days=1)

        return resolved

    # [next|this] <weekday> [at <time>|<daypart>]
    match = re.match(
        r"^(?:on\s+)?(next|this|coming)?\s*([a-z]+)(?:s)?"
        r"(?:\s+(?:at|in the|around)?\s*(.+))?$",
        text,
    )

    if match and match.group(2) in WEEKDAYS:
        weekday = WEEKDAYS[match.group(2)]
        rest = match.group(3)
        hour, minute = 9, 0

        if rest:
            clock = _shift_pm(_time_or_part(rest), rest in ("night", "evening"))

            if clock is None:
                return None

            hour, minute = clock

        return next_weekday(
            reference, weekday, hour, minute,
            force_next=match.group(1) == "next",
        )

    # 25 september / september 25 / 25th / the 3rd at 4pm
    resolved = _resolve_date(text, reference)

    if resolved:
        return resolved

    # a bare time: "5pm", "at 17:30", "09:00"
    clock = _clock(re.sub(r"^(?:at|around|about)\s+", "", text))

    if clock:
        hour, minute, vague = clock
        resolved = _at_wall_clock(reference, hour, minute)

        if resolved > reference:
            return resolved

        # Already gone. If nobody said am or pm, the other reading of
        # the same number is usually what they meant - "at 8" at two in
        # the afternoon is tonight, not tomorrow morning.
        if vague:
            other = resolved + timedelta(hours=12 if hour < 12 else -12)

            if other > reference:
                return other

        return resolved + timedelta(days=1)

    return None


def _resolve_date(text, reference):
    """Calendar dates without a year: '25 sep', 'september 25 at 3pm',
    'the 25th'. A date already gone this year means next year."""
    body, clock = text, None
    split = re.split(r"\s+(?:at|around|about)\s+", text, maxsplit=1)

    if len(split) == 2:
        body, rest = split
        parsed = _time_or_part(rest)

        if parsed is None:
            return None

        clock = parsed[:2]

    body = re.sub(r"^(?:on\s+|the\s+)+", "", body).strip()
    body = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", body)

    day = month = None

    match = re.match(r"^(\d{1,2})\s+([a-z]+)$", body)

    if match and match.group(2) in MONTHS:
        day, month = int(match.group(1)), MONTHS[match.group(2)]

    if day is None:
        match = re.match(r"^([a-z]+)\s+(\d{1,2})$", body)

        if match and match.group(1) in MONTHS:
            month, day = MONTHS[match.group(1)], int(match.group(2))

    if day is None:
        match = re.match(r"^(\d{1,2})/(\d{1,2})$", body)

        if match:
            first, second = int(match.group(1)), int(match.group(2))

            # Whichever number can't be a month is the day. When both
            # could be - "05/06" - month-first wins, matching the way
            # the date is written everywhere else in the app.
            if first > 12:
                day, month = first, second
            else:
                month, day = first, second

    if day is None and re.match(r"^\d{1,2}$", body) and clock is not None:
        day, month = int(body), reference.month

    if day is None or month is None:
        return None

    hour, minute = clock or (9, 0)

    for year in (reference.year, reference.year + 1):
        try:
            candidate = datetime(year, month, day, hour, minute)
        except ValueError:
            return None

        if candidate > reference:
            return candidate

    return None


# --- repeats ---------------------------------------------------------------
def _day_set(text):
    """'weekdays', 'monday and thursday', 'mon, wed, fri' -> (0, 3).

    Returns None unless *every* part is a day, so "green and blue" can't
    quietly become an empty schedule that never fires.
    """
    days = set()

    for part in re.split(r"\s*(?:,|/|and|&)\s*", text.strip().lower()):
        part = part.strip()

        if part in DAY_SETS:
            days.update(DAY_SETS[part])
        elif part in WEEKDAYS:
            days.add(WEEKDAYS[part])
        elif part:
            return None

    return tuple(sorted(days)) or None


def _day_set_repeat(body, reference):
    """'weekdays at 7:30', 'monday and thursday at 7', 'mon, wed, fri'.

    A set of days, not one - so it is not the weekly repeat below. Only
    fires for two days or more: a single day is already stored as
    {"weekday": n} in everybody's reminders.json and a second encoding
    of the same schedule is how the two of them drift apart.
    """
    match = re.match(
        r"^([a-z]+(?:\s*(?:,|/|and|&)\s*[a-z]+)*)"
        r"(?:\s+(?:at|around|in\s+the)?\s*(.+))?$",
        body,
    )

    if not match:
        return None

    days = _day_set(match.group(1))

    if not days or len(days) < 2:
        return None

    rest = match.group(2)
    hour, minute = 9, 0

    if rest:
        clock = _shift_pm(_time_or_part(rest), rest in ("night", "evening"))

        if clock is None:
            return None

        hour, minute = clock

    due = next_in_days(reference, days, hour, minute)

    return due, {"at": due.strftime("%H:%M"), "days": list(days)}


def _resolve_repeat(text, reference):
    """'every 30 minutes', 'every morning at 8', 'every monday at 9',
    'daily', 'hourly'. Returns (due, repeat) or None."""
    match = re.match(r"^(daily|nightly|hourly|weekly|monthly)\b\s*(.*)$", text)

    if match:
        word, rest = match.group(1), match.group(2).strip()
        rest = re.sub(r"^(?:at|around)\s+", "", rest)

        if word == "hourly":
            return reference + timedelta(hours=1), {"amount": 1, "unit": "hours"}

        if word == "weekly":
            due = _resolve_once(rest, reference) if rest else \
                reference + timedelta(days=7)

            return (due, {"amount": 1, "unit": "weeks"}) if due else None

        if word == "monthly":
            return _add_months(reference, 1), {"amount": 30, "unit": "days"}

        clock = _shift_pm(_clock(rest), word == "nightly") if rest else None

        if clock is None:
            clock = DAY_PARTS["night"] if word == "nightly" else (9, 0)

        due = _at_wall_clock(reference, *clock)

        if due <= reference:
            due += timedelta(days=1)

        return due, {"at": due.strftime("%H:%M")}

    match = re.match(r"^(?:every|each)\s+(.+)$", text)

    if not match:
        # No "every" in front. A plural day set is the only thing that
        # repeats on its own - "weekdays at 7" has no one-off reading,
        # whereas "30 minutes" very much does, so nothing else gets in.
        return _day_set_repeat(text.strip(), reference)

    body = match.group(1).strip()

    # every 30 minutes / every other day
    interval = re.match(r"^(other\s+)?(?:([\d.]+|[a-z-]+)\s+)?([a-z]+)$", body)

    if interval:
        raw_amount = interval.group(2)
        unit = UNITS.get(interval.group(3))

        if unit:
            amount = _number(raw_amount) if raw_amount else 1

            if interval.group(1):
                amount = (amount or 1) * 2

            if amount and amount > 0:
                return _delta(amount, unit, reference), \
                    {"amount": amount, "unit": unit}

    grouped = _day_set_repeat(body, reference)

    if grouped:
        return grouped

    # every monday [at 9] - weekly on a named day
    weekly = re.match(
        r"^([a-z]+?)s?(?:\s+(?:at|in the|around)?\s*(.+))?$", body
    )

    if weekly and weekly.group(1) in WEEKDAYS:
        weekday = WEEKDAYS[weekly.group(1)]
        rest = weekly.group(2)
        hour, minute = 9, 0

        if rest:
            clock = _shift_pm(_time_or_part(rest), rest in ("night", "evening"))

            if clock is None:
                return None

            hour, minute = clock

        due = next_weekday(reference, weekday, hour, minute)

        return due, {"at": due.strftime("%H:%M"), "weekday": weekday}

    # every day at 8 / every morning / every night at 10
    daily = re.match(
        r"^(?:day|morning|afternoon|evening|night|noon|midday)"
        r"(?:\s+(?:at|around)?\s*(.+))?$",
        body,
    )

    if daily:
        part = body.split()[0]
        hour, minute = DAY_PARTS.get(part, (9, 0))

        if daily.group(1):
            clock = _shift_pm(
                _clock(daily.group(1)), part in ("night", "evening")
            )

            if clock is None:
                return None

            hour, minute = clock

        due = _at_wall_clock(reference, hour, minute)

        if due <= reference:
            due += timedelta(days=1)

        return due, {"at": due.strftime("%H:%M")}

    return None


def parse_when(phrase, reference=None, morning=False):
    """Natural language in, (datetime, repeat) out.

    This is the whole point of the module. The model is asked for the
    words, not the maths: it passes through "next tuesday at 9" exactly
    as the user said it, and every calculation happens here where it
    can be tested.

    Returns (None, None) when the phrase doesn't describe a time, which
    the caller should treat as "ask the user", never as "guess".

    `morning=True` is for alarms: it makes a bare "6" mean six in the
    morning instead of six in the evening. An explicit "6pm" still wins.
    """
    reference = reference or now()
    text = _clean(phrase)

    if not text:
        return None, None

    token = _MORNING.set(bool(morning))

    try:
        repeated = _resolve_repeat(text, reference)

        if repeated:
            return repeated

        return _resolve_once(text, reference), None
    finally:
        _MORNING.reset(token)


def next_occurrence(repeat, reference=None, after=None):
    """When a repeating reminder should next fire.

    `reference` is what the schedule is measured from - pass the time
    it was *due*, not the time it was delivered, so a daily 08:00 that
    went out at 08:04 is still 08:00 tomorrow instead of creeping
    forward four minutes a day.

    `after` skips a backlog: if the app was closed for a week, the
    daily reminder is due tomorrow morning, not seven times at once.

    Wall-clock times are rebuilt rather than advanced by 24 hours, so
    08:00 stays 08:00 through a DST change; intervals are added as
    elapsed time, which is what "every 90 minutes" means.
    """
    reference = reference or now()
    after = after or now()

    result = _step(repeat, reference)

    # Guard against a pathological repeat (a zero interval would spin).
    for _ in range(4096):
        if result > after:
            break

        stepped = _step(repeat, result)

        if stepped <= result:
            return after + timedelta(minutes=1)

        result = stepped

    return result


def _step(repeat, reference):
    if repeat.get("days"):
        hour, minute = _hm(repeat.get("at", "09:00")) or (9, 0)

        return next_in_days(reference, repeat["days"], hour, minute,
                            force_next=True)

    if repeat.get("weekday") is not None:
        hour, minute = _hm(repeat.get("at", "09:00")) or (9, 0)

        return next_weekday(reference, repeat["weekday"], hour, minute,
                            force_next=True)

    if repeat.get("unit"):
        return _delta(float(repeat["amount"]), repeat["unit"], reference)

    hour, minute = _hm(repeat.get("at", "09:00")) or (9, 0)
    due = _at_wall_clock(reference, hour, minute)

    while due <= reference:
        due += timedelta(days=1)
        due = _at_wall_clock(due, hour, minute)

    return due
