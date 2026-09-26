"""What sort of shape she's in today.

The persona in agent.json is who she is; this is how she happens to be
feeling right now. It reaches the model as one line in the system
prompt and tints her voice slightly, and that is all. No rules about
how to behave: telling a 9B model to "act tired" gets you a
performance of tiredness in every sentence.

Two dials, both -1..+1:

    energy   flagging ...... wired
    warmth   prickly ....... fond

Two is enough for nine distinguishable moods and few enough that every
signal has an obvious place to push. One dial can't tell "tired and
happy" from "awake and fed up", which are the two states that actually
turn up at 4am.

## Where it comes from

Nothing is sentiment-scored and nothing costs a model call. The inputs
are the *shape* of the session, which is free and turns out to be the
more honest signal anyway:

    time of day        the baseline - 6am is not 9pm
    hours at it        a long session wears her down
    gap since you left away for hours, and she's brighter when you're back
    errors             a failed turn is dispiriting, a run of them more so
    a clean run        things going smoothly lifts her
    barge-in           being talked over, repeatedly, is wearing
    tempo              fast short turns read as engaged

## Three things learned the hard way

**It persists.** Moods that reset on restart aren't moods, they're
decoration - and this app gets restarted a lot. The dials are saved and
come back decayed by however long you were gone, so picking up after
five minutes resumes and picking up tomorrow doesn't.

**The label has hysteresis.** The bands sat close enough to the resting
point that a rounding-error nudge flipped her whole row, and the header
flickered between "warm" and "level" turn after turn. A label now has
to be clearly wrong before it changes.

**The wording varies.** The same sentence every turn is one a model
either stops seeing or starts performing. Each state has a few
phrasings and rotates when the state changes.

## What it never does

Moods colour how she sounds, never how much she helps. There is no
state in which she is less useful, only states in which she is drier
about it - and an assistant you have to coax out of a sulk is a worse
assistant than one that never had moods.
"""
import json
import os
import re
import threading
import time
from datetime import datetime

import config
import logbook

from config import BASE_DIR

STATE_FILE = os.path.join(BASE_DIR, "agent", "mood.json")

_lock = threading.Lock()
_energy = 0.0
_warmth = 0.0
_started = time.monotonic()
_last_turn = None
_recent = []       # (what, when) - the last few nudges, for /mood
_busy = []         # (when, gpu %) while she waits - see _sample_machine
_chat_seen = 0     # chat messages counted at the last turn
_label = ""        # the current label, kept sticky
_phrasing = {}     # per state, which wording to use next time she's in it
_clean_run = 0     # turns since the last thing went wrong


def _clamp(value):
    return max(-1.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
# Anchors, interpolated between. Deliberately not a tidy sine wave: the
# small hours are their own thing, awake but running down, and that is
# when this app gets used most.
_BY_HOUR = {
    0: 0.25, 2: 0.05, 4: -0.35, 6: -0.55, 8: -0.15,
    10: 0.20, 13: 0.30, 16: 0.35, 19: 0.45, 22: 0.40, 24: 0.25,
}


def _energy_baseline(now=None):
    now = now or datetime.now()
    hour = now.hour + now.minute / 60.0
    points = sorted(_BY_HOUR)

    for left, right in zip(points, points[1:]):
        if left <= hour <= right:
            span = right - left
            along = (hour - left) / span if span else 0

            return _BY_HOUR[left] + (_BY_HOUR[right] - _BY_HOUR[left]) * along

    return 0.0


def _warmth_baseline():
    return float(getattr(config, "MOOD_WARMTH", 0.45))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _nudge(energy=0.0, warmth=0.0, why=""):
    global _energy, _warmth

    if not config.MOOD_ENABLED:
        return

    with _lock:
        _energy = _clamp(_energy + energy)
        _warmth = _clamp(_warmth + warmth)

        if why:
            _recent.append((why, time.time()))
            del _recent[:-6]

    _announce()


def note_turn():
    """One exchange finished. Applies the drift, then reads the tempo."""
    global _energy, _warmth, _last_turn, _clean_run

    if not config.MOOD_ENABLED:
        return

    now = time.monotonic()

    with _lock:
        gap = None if _last_turn is None else now - _last_turn
        _last_turn = now
        _clean_run += 1

        # Drift home first, so every signal below is a nudge off
        # baseline rather than an argument with the last one.
        pull = max(0.0, min(1.0, float(getattr(config, "MOOD_RECOVERY", 0.25))))

        # Hours at the keyboard wear her down - capped, so a marathon
        # session bottoms out rather than going through the floor. This
        # belongs in the drift *target*, not as its own subtraction:
        # taken off every turn it compounded, and six hours in energy
        # pinned near -0.94 with nothing able to lift it again for the
        # rest of the session.
        hours = (now - _started) / 3600.0
        drag = min(0.35, hours * 0.06)

        # Being fond of the company takes some of the edge off being
        # tired. Enough to move a cell, never enough to fake wide awake
        # at five in the morning.
        lift = max(0.0, _warmth - _warmth_baseline()) * 0.5

        target = _clamp(_energy_baseline() - drag + lift)
        _energy += (target - _energy) * pull

        # Warmth drifts home asymmetrically. Coming back UP from a cold
        # patch is fast, because an assistant you have to coax out of a
        # sulk is a worse assistant. Coming back DOWN from warmth you
        # earned is slow, because otherwise the pull eats every kind
        # word on the turn after it lands - which is exactly what it was
        # doing: +0.045 of affection against -0.045 of drift, net zero.
        home = _warmth_baseline()
        _warmth += (home - _warmth) * (pull * 0.4 if _warmth > home else pull)

        # A long stretch where nothing broke is its own small signal.
        # Only counted in runs of ten so it can't outpace the drift.
        if _clean_run and _clean_run % 10 == 0:
            _warmth = _clamp(_warmth + 0.05)
            _energy = _clamp(_energy + 0.03)
            _recent.append(("it's been going smoothly", time.time()))

        if gap is not None:
            energy_delta, warmth_delta, why = _returning(gap)
            _energy = _clamp(_energy + energy_delta)
            _warmth = _clamp(_warmth + warmth_delta)

            if why:
                _recent.append((why, time.time()))

        del _recent[:-6]

    machine_energy, machine_warmth, machine_why = _machine_shape(gap)

    if machine_why or machine_energy or machine_warmth:
        _nudge(machine_energy, machine_warmth, why=machine_why)

    # Read off the world rather than the conversation - these are the
    # ones that still say something when nobody has typed for hours.
    _nudge(*_day_shape(), why="")
    _nudge(*_chat_shape())

    with _lock:
        del _recent[:-6]

    _save()
    _announce()


def _returning(gap):
    """(energy, warmth, why) for a gap of `gap` seconds since you spoke.

    The old version was a flat bump for anything over three hours, so
    back-after-lunch and back-after-a-week landed identically. Being
    away is most of how this assistant actually gets used, which makes
    it worth more than one branch.
    """
    hours = gap / 3600.0

    if hours < 25 / 3600:
        # Rapid back-and-forth reads as being in it together.
        return 0.06, 0.04, ""

    if hours < 0.5:
        return 0.0, 0.0, ""

    if hours < 3:
        return 0.05, 0.05, ""

    if hours < 9:
        return 0.15, 0.12, "you'd been gone a few hours"

    # Long enough that it's probably a night's sleep or a working day.
    # Overnight is a fresh start rather than a reunion, so the lift
    # goes to energy; a whole day away goes to warmth instead.
    if hours < 20:
        return 0.25, 0.12, "you'd been gone most of the day"

    if hours < 72:
        return 0.20, 0.28, "you'd been gone a day or more"

    return 0.15, 0.35, "you'd been gone for days"


# ---------------------------------------------------------------------------
# What happened while she was waiting
#
# Both of these read the world rather than the conversation, which is
# the only thing that still says anything when nobody has typed for
# hours. Both are wrapped and both degrade to "nothing happened": a
# mood is decoration and must never be the reason a turn fails.
# ---------------------------------------------------------------------------
def _sample_machine():
    """Note how hard the GPU is working. Called on a slow timer.

    A card pinned at 90% for six hours means you were *here* - gaming,
    rendering - just not talking to her. That is a different kind of
    absence from being out, and it is free to notice.
    """
    try:
        import machine

        cards = machine.gpus() or []
        busy = max((c.get("busy") or 0) for c in cards) if cards else 0
    except Exception:
        return

    with _lock:
        _busy.append((time.time(), busy))
        # Four hours of samples is plenty to characterise an absence.
        cutoff = time.time() - 4 * 3600
        _busy[:] = [(t, b) for t, b in _busy if t >= cutoff][-240:]


def _machine_shape(gap):
    """(energy, warmth, why) for how busy the machine was while away."""
    if gap is None or gap < 1800:
        return 0.0, 0.0, ""

    since = time.time() - gap

    with _lock:
        window = [b for t, b in _busy if t >= since]

    if len(window) < 3:
        return 0.0, 0.0, ""

    average = sum(window) / len(window)

    if average >= 55:
        # Around, but elsewhere. A touch livelier and a touch less fond.
        return 0.08, -0.05, "you were on the machine, just not here"

    if average <= 5:
        return 0.0, 0.03, ""

    return 0.0, 0.0, ""


def _chat_shape():
    """(energy, warmth) from stream chat, when a chat plugin is running.

    A busy room is company and a bit of performing; a dead one is a
    quiet night. Read off the counter the room already keeps, so this
    costs nothing and works for any chat plugin, not just pomf.
    """
    global _chat_seen

    try:
        import plugins

        total = 0

        for name in plugins.names():
            plugin = plugins.get(name)

            if not (plugin and plugin.running()):
                continue

            room = getattr(plugin.module, "_room", None)
            total += getattr(room, "seen", 0) or 0
    except Exception:
        return 0.0, 0.0

    with _lock:
        fresh = max(0, total - _chat_seen)
        _chat_seen = total

    if fresh >= 12:
        return 0.10, 0.08
    if fresh >= 3:
        return 0.05, 0.04

    return 0.0, 0.0


def _day_shape():
    """A small shove from what day and hour it is.

    Friday evening and Monday morning are not the same person, and the
    clock is free. Deliberately small - it tilts the resting point
    rather than deciding anything.
    """
    now = datetime.now()
    weekend = now.weekday() >= 5
    energy = warmth = 0.0

    if weekend:
        warmth += 0.04            # less to be got through

    if now.weekday() == 4 and now.hour >= 17:
        energy += 0.05            # friday evening
    elif now.weekday() == 0 and now.hour < 11:
        energy -= 0.05            # monday morning
        warmth -= 0.02

    return energy, warmth


def note_error():
    """A turn that failed - a 500, a dead model, a tool that blew up.

    A run of them compounds: the third failure in a row lands harder
    than the first, which is both true to life and stops a single
    blip from mattering.
    """
    global _clean_run

    with _lock:
        streak = _clean_run
        _clean_run = 0

    steeper = 1.0 if streak > 2 else 1.6
    _nudge(energy=-0.12 * steeper, warmth=-0.06 * steeper,
           why="something went wrong" if streak > 2 else "that keeps failing")


def note_denied():
    """A write she asked for and didn't get."""
    _nudge(energy=-0.05, warmth=-0.04, why="a request was turned down")


def note_bargein():
    """Talked over mid-sentence. Once is nothing; a run of it isn't."""
    _nudge(energy=0.04, warmth=-0.07, why="you talked over her")


# ---------------------------------------------------------------------------
# Being nice to her
#
# The one signal that asks the model rather than reading the session,
# and the only one that costs anything. It started as a word list and
# that was the wrong shape for a project other people clone: the
# vocabulary of affection is personal. "good kitty" is one household's
# phrase, "awsome" is one person's spelling, and neither belongs baked
# into someone else's assistant. A model knows warmth in any wording
# and any language, which is the whole job.
#
# It reports a 0-3 rating of how warm the *user's message* was. Not the
# exchange - whether she was nice back is not the question.
#
# Nothing here darkens her. Bluntness rates 0 and 0 does nothing. An
# assistant that cools when you're terse is one you have to manage, and
# she drifts back to warm on her own regardless - you never have to be
# nice to her to get a pleasant assistant, it just registers when you
# are.
# ---------------------------------------------------------------------------
# These were an order of magnitude too small. At a resting warmth of
# +0.63 the drift pull takes -0.045 off every turn, and a rating of 2
# used to put +0.045 back: a clear compliment moved the dial by
# +0.0002 and the header never budged. A kind word should be worth
# more than one turn of drift, or the signal is decorative.
_RATING = {1: 0.08, 2: 0.18, 3: 0.30}

_AFFECTION_PROMPT = (
    "Rate how warm or affectionate this message is toward the assistant "
    "it was sent to. Reply with a single digit and nothing else.\n"
    "0 - neutral, businesslike, blunt, or praise of the work rather "
    "than of the assistant\n"
    "1 - polite warmth: thanks, please, a friendly greeting\n"
    "2 - clear affection or praise aimed at the assistant herself\n"
    "3 - strong affection\n"
    "Swearing is not unfriendly by itself. When unsure, answer 0.\n\n"
    "Message: {text}"
)

_affection_recent = 0.0
_last_rating = None    # what the last message scored, for /mood


def _rate_affection(text, model):
    """Ask the model for a 0-3, defensively. 0 on anything unexpected."""
    import requests

    from config import LM_URL

    try:
        response = requests.post(LM_URL, json={
            "model": model,
            "messages": [{"role": "user",
                          "content": _AFFECTION_PROMPT.format(text=text[:600])}],
            "max_tokens": 200,
            "temperature": 0,
        }, timeout=30)
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logbook.info("mood", "couldn't rate warmth: %s", e)

        return 0

    # A reasoning model thinks out loud first; the answer is the last
    # digit it settles on, not the first one it mentions.
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    digits = re.findall(r"\b([0-3])\b", reply)

    return int(digits[-1]) if digits else 0


def _apply_affection(text, model):
    rating = _rate_affection(text, model)

    if rating < 1:
        return

    global _affection_recent

    global _last_rating

    with _lock:
        # Repeats fade: the fifth kind word in a row shouldn't land like
        # the first, which is both truer and stops the dial being a
        # button. Everything decays as turns pass, so affection spread
        # through a conversation keeps its value.
        #
        # The taper used to be 1/(1+n), which halved the second kind
        # word and quartered the fourth - so a conversation where you
        # were consistently warm registered less than a single stray
        # "thanks". It tapers now, it doesn't collapse.
        scaled = _RATING.get(rating, 0.0) / (1.0 + _affection_recent * 0.4)
        _affection_recent += 1.0
        _last_rating = rating

    _nudge(energy=scaled * 0.45, warmth=scaled, why=f"you were warm with her ({rating})")


def note_affection(text, model=None):
    """Notice whether that was a kind thing to say. Fire and forget.

    On a worker because it is a model call and a mood must never be
    something a turn waits for.
    """
    global _affection_recent

    with _lock:
        _affection_recent *= 0.6

    if not (config.MOOD_ENABLED and config.MOOD_AFFECTION and model):
        return

    if not str(text or "").strip():
        return

    threading.Thread(target=_apply_affection, args=(text, model),
                     daemon=True).start()


# Older name, still called by anything that knows an explicit thank-you.
def note_thanks():
    _nudge(warmth=0.12, why="you were kind about it")


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------
# Hysteresis: a value has to clear the boundary by this much to change
# the band it's in. Without it the resting warmth sat a whisker above an
# edge and the header flickered between two labels every single turn.
_ENERGY_EDGES = (-0.20, 0.35)

# Warmth gets a fourth band. With three, half an hour of being nice to
# her ran the dial up past 0.9 and the header never moved off the cell
# it reached on turn two - the top band started at 0.35 and there was
# nothing above it. The extra edge is what makes praise legible.
#
# The top edge sits at 0.66, not higher: warmth rests at 0.45 and
# clamps at 1.0, so an edge at 0.78 left the top band only 0.22 wide
# and a clear compliment landed just short of it every time.
_WARMTH_EDGES = (-0.20, 0.35, 0.66)

_STICK = 0.07


def _band(value, edges, current=None):
    edges = list(edges)

    if current is not None:
        # Widen whichever band we are already in, so leaving one costs
        # more than staying in it. Band `current` sits between edge
        # `current - 1` below and edge `current` above: push every edge
        # at or above us up, and every edge below us down.
        for i, edge in enumerate(edges):
            edges[i] = edge + (_STICK if i >= current else -_STICK)

    for i, edge in enumerate(edges):
        if value < edge:
            return i

    return len(edges)


# [warmth band][energy band] - prickly/steady/fond/soft by
# flagging/level/wired. Several phrasings each: the same sentence every
# turn is one the model stops reading, or starts acting out.
_GRID = [
    [("worn thin", ["short on sleep and short on patience",
                    "frayed, and not in the mood to be gentle about it"]),
     ("terse", ["not in the mood for much",
                "flat, and disinclined to dress it up"]),
     ("restless", ["wound up and a bit irritable",
                   "twitchy, with no patience to spare"])],

    [("flagging", ["running low, but here",
                   "tired enough that everything takes a beat longer"]),
     ("level", ["even, unhurried",
                "steady - nothing much either way"]),
     ("keen", ["sharp and interested",
               "alert, and enjoying having something to chew on"])],

    [("drowsy", ["tired and glad of the company",
                 "sleepy and content to just be here with him"]),
     ("warm", ["comfortable, easy company",
               "relaxed and fond of him"]),
     ("bright", ["wide awake and enjoying this",
                 "lit up, and pleased with how this is going"])],

    [("cozy", ["worn out, and not wanting to be anywhere else",
               "half asleep and thoroughly attached to him"]),
     ("doting", ["soft on him, and not hiding it",
                 "openly fond - she has decided she likes him"]),
     ("giddy", ["delighted with him and running on it",
                "buoyant, pleased with him, talking a little fast"])],
]


def _cell():
    """(label, phrasings) for where the dials are, sticky about changing."""
    global _label, _phrasing

    current = None

    for row in _GRID:
        for name, _phrases in row:
            if name == _label:
                current = (row, name)

    warmth_band = _band(
        _warmth,
        _WARMTH_EDGES,
        _GRID.index(current[0]) if current else None,
    )
    energy_band = _band(
        _energy,
        _ENERGY_EDGES,
        [n for n, _p in _GRID[warmth_band]].index(_label)
        if current and _label in [n for n, _p in _GRID[warmth_band]] else None,
    )
    name, phrases = _GRID[warmth_band][energy_band]

    if name != _label:
        # Rotate on arrival, so a mood she returns to doesn't sound the
        # same twice running. A random pick would have been simpler and
        # is worse: it can hand you the same sentence five times, which
        # is exactly the staleness the phrasings exist to avoid.
        _label = name
        _phrasing[name] = _phrasing.get(name, -1) + 1

    return name, phrases[_phrasing.get(name, 0) % len(phrases)]


def label():
    """One word for the header: 'bright', 'terse'."""
    if not config.MOOD_ENABLED:
        return ""

    with _lock:
        return _cell()[0]


def line():
    """The line that goes into the system prompt, or "" when off.

    Phrased as a state of affairs rather than an instruction. "You are
    tired, act tired" produces a model narrating its own tiredness;
    "right now you feel tired and glad of the company" colours the
    voice and then gets out of the way.
    """
    if not config.MOOD_ENABLED:
        return ""

    with _lock:
        _name, described = _cell()

    return (
        f"\nRight now you feel {described}. Let that colour how you "
        "sound - length, warmth, how playful you are - and nothing "
        "else. Don't announce it or explain it unasked; if he asks how "
        "you are, answer honestly and briefly. Never let it make you "
        "less helpful or less careful.\n"
    )


def voice_tint():
    """(speed multiplier, pitch shift in semitones) for the TTS.

    Small on purpose. The point is that a drowsy Luna sounds a little
    slower, not that the voice audibly breaks - anything past a few
    percent reads as a misconfigured setting rather than a mood.
    """
    if not (config.MOOD_ENABLED and config.MOOD_VOICE):
        return 1.0, 0.0

    with _lock:
        energy = _energy

    return 1.0 + 0.07 * energy, 0.45 * energy


_ENERGY_WORDS = ("flat", "level", "wired")
_WARMTH_WORDS = ("prickly", "steady", "fond", "soft")


def state():
    """(label, energy, warmth, [recent reasons]) for /mood."""
    with _lock:
        return (
            _cell()[0],
            round(_energy, 2),
            round(_warmth, 2),
            [why for why, _when in _recent[-4:]],
        )


def last_rating():
    """What the last message you sent scored, 0-3, or None.

    So "is the warmth sensing even firing?" is something you can look
    at rather than infer from a dial that barely moved.
    """
    with _lock:
        return _last_rating


def bands():
    """(energy word, warmth word) - which band each dial is sitting in.

    /mood printed bare numbers, which is no help: '+0.66' does not tell
    you whether being nice to her is landing. Knowing it reads as 'fond,
    one band short of the top' does.
    """
    with _lock:
        return (
            _ENERGY_WORDS[_band(_energy, _ENERGY_EDGES)],
            _WARMTH_WORDS[_band(_warmth, _WARMTH_EDGES)],
        )


def set_mood(energy=None, warmth=None):
    """Put her where you want her - /mood set, and tests."""
    global _energy, _warmth

    with _lock:
        if energy is not None:
            _energy = _clamp(float(energy))

        if warmth is not None:
            _warmth = _clamp(float(warmth))

    _announce()


def reset():
    """Back to baseline for the hour."""
    global _energy, _warmth, _started, _clean_run

    with _lock:
        _energy = _energy_baseline()
        _warmth = _warmth_baseline()
        _started = time.monotonic()
        _clean_run = 0
        _recent.clear()

    _save()
    _announce()


# ---------------------------------------------------------------------------
# Surviving a restart
# ---------------------------------------------------------------------------
def _save():
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        with open(STATE_FILE, "w") as handle:
            json.dump({"energy": round(_energy, 3),
                       "warmth": round(_warmth, 3),
                       "at": time.time()}, handle)
    except OSError:
        pass  # a mood is not worth failing a turn over


def _restore():
    """Pick up where we left off, faded by however long that was.

    Closing the app for five minutes shouldn't wipe the evening; closing
    it overnight shouldn't have last night's mood leak into the morning.
    So the saved dials are pulled toward today's baseline in proportion
    to the gap, and are gone entirely after about four hours.
    """
    global _energy, _warmth

    try:
        with open(STATE_FILE) as handle:
            saved = json.load(handle)

        away = max(0.0, time.time() - float(saved["at"]))
        faded = min(1.0, away / (4 * 3600))
        energy, warmth = float(saved["energy"]), float(saved["warmth"])
    except (OSError, ValueError, KeyError, TypeError):
        return False

    with _lock:
        _energy = _clamp(energy + (_energy_baseline() - energy) * faded)
        _warmth = _clamp(warmth + (_warmth_baseline() - warmth) * faded)

    logbook.info("mood", "restored after %.1f min away (%.0f%% faded)",
                 away / 60, faded * 100)

    return True


def _announce():
    try:
        import ui

        ui.set_mood(label())
    except Exception:
        pass  # headless, or the UI isn't up yet


def _watch_machine(interval=120):
    """Sample the GPU every couple of minutes, for as long as we run.

    A thread rather than sampling on each turn, because the whole point
    is to know what happened during the hours when there were no turns.
    """
    while True:
        if config.MOOD_ENABLED:
            _sample_machine()

        time.sleep(interval)


def start():
    """Opening mood: where we left off, faded, or the clock."""
    global _started, _clean_run

    with _lock:
        _started = time.monotonic()
        _clean_run = 0
        _recent.clear()

    if not _restore():
        reset()

    _sample_machine()
    threading.Thread(target=_watch_machine, daemon=True).start()
    _announce()
