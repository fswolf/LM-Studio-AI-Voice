"""Full-screen TUI built on prompt_toolkit.

Replaces the old "clear the screen and print()" dashboard. Three things
that bought us:

  * No flicker. prompt_toolkit diffs the screen and redraws only what
    changed, instead of wiping and repainting every frame.
  * Keys work while this window is focused - HOME for push-to-talk, ESC
    to quit - because the app reads keys as they arrive rather than
    blocking in input() waiting for a newline. No evdev, no compositor
    bind, no permissions, same behaviour on Linux/macOS/Windows.
  * Real panels, and a conversation you can scroll back through.

Public API is unchanged for callers: set_status, add_message, init,
set_controls, set_voice_server. Instead of ui.prompt() in a loop, main
calls ui.run(on_submit=...) once and the app owns the main thread.
"""
import re
import threading
import time

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

import logbook
from config import THEME

try:
    from wcwidth import wcswidth
except ImportError:  # pragma: no cover - fallback if wcwidth isn't installed
    wcswidth = None

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Shared UI state (mutated by assistant/keyboard threads, read on render)
# ---------------------------------------------------------------------------
state = {
    "agent_name": "Luna",
    "model": "unknown",
    "voice": "af_bella",
    "voice_server": "unknown",
    "status": "Idle",
    "memory": "Loaded",
    "controls": "",
    "mode": "auto",
}
conversation = []  # list of (speaker, text) tuples

_app = None
_on_submit = None
_on_hotkey = None
_conv_window = None
_scrollback = 0  # lines scrolled up from the bottom; 0 = pinned to latest
_help_visible = False
_help_scroll = 0  # lines scrolled down from the top of the help text
_help_window = None
_approval = None  # the pending permission request, or None
_approval_scroll = 0
_approval_window = None
_approval_lock = threading.Lock()  # one question at a time

# Mouse capture is off by default, and that is a deliberate trade.
# prompt_toolkit's mouse support gives you wheel scrolling, but it turns
# on terminal mouse reporting - which means the terminal hands drags to
# the application instead of doing its own text selection. You lose
# copy and paste entirely, which matters far more in a window full of
# log lines and error messages than scrolling does, especially when
# PgUp/PgDn/End already scroll.
_mouse = False


# ---------------------------------------------------------------------------
# State updates - all thread-safe, all just repaint
# ---------------------------------------------------------------------------
def _refresh():
    if _app is not None:
        try:
            _app.invalidate()
        except Exception:
            pass  # app not running yet, or already torn down


def set_status(status: str):
    with _lock:
        state["status"] = status
    _refresh()


def set_voice_server(label: str):
    """Update the server shown beside the TTS voice name."""
    with _lock:
        if state["voice_server"] == label:
            return
        state["voice_server"] = label
    _refresh()


def set_mode(mode: str):
    """Recording mode - shown on the Voice row."""
    with _lock:
        if state["mode"] == mode:
            return
        state["mode"] = mode
    _refresh()


def set_controls(text: str):
    """A note shown at the bottom of the help window (Tab).

    Was the footer line; the footer is gone, but the keyboard listener
    still uses this to report when no global hotkey is available.
    """
    with _lock:
        if state["controls"] == text:
            return
        state["controls"] = text
    _refresh()


def add_message(speaker: str, text: str):
    """Add a finished message.

    If a streamed reply is open, this goes *above* it rather than at
    the end. Tool logs arrive while she's mid-sentence, and they
    describe something that already happened, so they belong before
    her answer - and putting them after would leave the open message
    no longer last, which is what extend_message relies on.
    """
    global _scrollback, _streaming

    with _lock:
        if _streaming is not None and 0 <= _streaming < len(conversation):
            conversation.insert(_streaming, (speaker, text))
            _streaming += 1
        else:
            conversation.append((speaker, text))

        _scrollback = 0  # a new message pulls you back to the bottom
    _refresh()


# A streamed reply is one message that grows while other messages may
# arrive around it. Its position is tracked explicitly rather than
# assumed to be last, because a tool call logs a system message in the
# middle of her sentence - and appending to conversation[-1] then put
# her whole reply inside the system row, labelled "sys".
_streaming = None


def begin_message(speaker: str):
    """Open an empty message for extend_message() to fill in."""
    global _scrollback, _streaming

    with _lock:
        conversation.append((speaker, ""))
        _streaming = len(conversation) - 1
        _scrollback = 0
    _refresh()


def extend_message(text: str):
    """Append to the open message as tokens arrive."""
    global _scrollback

    if not text:
        return

    with _lock:
        if _streaming is None or not 0 <= _streaming < len(conversation):
            return

        speaker, existing = conversation[_streaming]
        conversation[_streaming] = (speaker, existing + text)
        _scrollback = 0
    _refresh()


def replace_message(text: str):
    """Swap the streamed text for the final cleaned-up version.

    Worth doing even though they're usually identical: the streamed
    text is raw, and the final one has had stray timestamps and
    leftover think-tags stripped out of it.
    """
    global _scrollback

    with _lock:
        if _streaming is None or not 0 <= _streaming < len(conversation):
            return

        speaker, existing = conversation[_streaming]

        if existing == text:
            return

        conversation[_streaming] = (speaker, text)
        _scrollback = 0
    _refresh()


def end_message():
    """Close the open message, dropping it if nothing arrived in it."""
    global _streaming

    with _lock:
        index, _streaming = _streaming, None

        if index is None or not 0 <= index < len(conversation):
            return

        if not conversation[index][1].strip():
            conversation.pop(index)
    _refresh()


# Older name, kept so nothing breaks if it's still called somewhere.
drop_empty_message = end_message


def mouse_enabled():
    return _mouse


def toggle_mouse(on=None):
    """Swap between wheel scrolling and being able to select text."""
    global _mouse

    _mouse = (not _mouse) if on is None else bool(on)
    _refresh()

    return _mouse


def set_voice(name: str):
    """Update the Voice row. /voice can change this mid-session, and a
    header still showing the old name is worse than no header."""
    with _lock:
        state["voice"] = name
    _refresh()


def set_model(label: str):
    """Update the Model row - it now shows health, not just a name."""
    with _lock:
        state["model"] = label
    _refresh()


def set_memory(text: str):
    """The Memory row: which backend, how many facts. longterm.status()
    writes it whenever the table changes, so it's never a stale label."""
    with _lock:
        if state["memory"] == text:
            return
        state["memory"] = text
    _refresh()


def init(agent_name, model, voice, memory_status="Loaded", voice_server="unknown"):
    with _lock:
        state["agent_name"] = agent_name
        state["model"] = model
        state["voice"] = voice
        state["voice_server"] = voice_server
        state["memory"] = memory_status
    _refresh()


def render():
    """Kept so older call sites don't break - repainting is automatic."""
    _refresh()


# ---------------------------------------------------------------------------
# Text measuring / wrapping
# ---------------------------------------------------------------------------
def vwidth(s: str) -> int:
    """Visible width in terminal cells.

    Emoji and other wide characters break len()-based wrapping - an emoji
    is usually 2 cells but len() counts 1 - and Luna uses them, so
    measure properly when wcwidth is available.
    """
    if wcswidth is None:
        return len(s)

    w = wcswidth(s)

    return w if w is not None and w >= 0 else len(s)


def _wrap_text(text: str, width: int):
    """Word-wrap on visible width rather than character count."""
    words = text.split()

    if not words:
        return [""]

    lines = []
    current = []
    current_width = 0

    for word in words:
        w = vwidth(word)
        space = 1 if current else 0

        if current_width + space + w > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_width = w
        else:
            current.append(word)
            current_width += space + w

    if current:
        lines.append(" ".join(current))

    return lines or [""]


def _width():
    if _app is None:
        return 80

    # minus the frame's two borders and two padding columns
    return max(20, _app.output.get_size().columns - 4)


# ---------------------------------------------------------------------------
# Panel contents
# ---------------------------------------------------------------------------
def _status_lines():
    voice_server = state["voice_server"]
    offline = "offline" in voice_server
    server_style = "class:warn" if offline else "class:value"

    # Voice = how you talk to her (the recording mode); TTS = how she
    # talks back, which voice and where it's synthesized. The TTS voice
    # and its server are one fact, not two.
    rows = [
        ("Status", [
            ("class:ok" if state["status"] == "Idle" else "class:value",
             state["status"]),
        ]),
        ("Voice", [("class:value", state["mode"])]),
        ("Memory", [("class:value", state["memory"])]),
        ("TTS", [
            ("class:value", state["voice"]),
            ("class:dim", " - "),
            (server_style, voice_server),
        ]),
        ("Model", [("class:value", state["model"])]),
    ]

    fragments = []

    for index, (label, parts) in enumerate(rows):
        if index:
            fragments.append(("", "\n"))

        fragments.append(("class:label", f" {label:<8}"))
        fragments.append(("class:dim", "│ "))
        fragments.extend(parts)

    return fragments


_line_cache = {"key": None, "lines": []}


def _conversation_lines():
    """The conversation as a list of rendered lines, pre-wrapped so
    continuation lines line up under the first one.

    Cached on (message count, width, last message) because it's needed
    twice per frame - once for the text, once to work out where the
    cursor goes - and re-wrapping a long history twice a frame is waste.
    """
    width = _width()
    key = (
        len(conversation), width,
        conversation[-1] if conversation else None,
        conversation[_streaming] if _streaming is not None
        and 0 <= _streaming < len(conversation) else None,
    )

    if _line_cache["key"] == key:
        return _line_cache["lines"]

    lines = []

    for index, (speaker, text) in enumerate(conversation):
        if speaker == "user":
            label, name_style = "You", "class:user"
        elif speaker == "system":
            label, name_style = "sys", "class:system"
        elif speaker.startswith("chat:"):
            # A stream viewer, not the person sitting here. The column
            # names the source rather than the person, because viewer
            # names are arbitrary length and this one is five wide -
            # truncating a stranger's name is worse than putting it at
            # the front of what they said, where it reads naturally
            # and can't collide.
            label, name_style = speaker[5:][:5], "class:guest"
        else:
            label, name_style = state["agent_name"][:5], "class:agent"

        # Only the name is coloured. Bodies stay in one readable colour -
        # a wall of pink is pretty for one line and tiring for twenty.
        body_style = "class:system" if speaker == "system" else "class:text"

        prefix = f" {label:<5} "
        indent = " " * (len(prefix) + 2)
        limit = max(10, width - len(prefix) - 2)

        # Wrap each line separately so explicit newlines survive - a
        # numbered list from /reminders would otherwise collapse into one
        # run-on paragraph, since _wrap_text splits on all whitespace.
        wrapped = []

        for paragraph in text.split("\n"):
            wrapped.extend(_wrap_text(paragraph, limit))

        if index:
            lines.append([("", "")])  # breathing room between turns

        lines.append([
            (name_style + " bold", prefix),
            ("class:dim", "│ "),
            (body_style, wrapped[0]),
        ])

        for line in wrapped[1:]:
            lines.append([("", indent), (body_style, line)])

    if not lines:
        lines = [[("class:dim", " Say something, or press HOME to talk.")]]

    _line_cache["key"] = key
    _line_cache["lines"] = lines

    return lines


def _conversation_fragments():
    fragments = []

    for line in _conversation_lines():
        fragments.extend(line)
        fragments.append(("", "\n"))

    return fragments


def _window_height():
    """Visible height of the conversation pane, from the last paint."""
    if _conv_window is not None and _conv_window.render_info is not None:
        return max(1, _conv_window.render_info.window_height)

    return 10


def _page_size():
    """One PgUp/PgDn step: a screenful, less two lines of overlap."""
    return max(1, _window_height() - 2)


def _desired_top():
    """Which line should sit at the top of the pane.

    _scrollback is measured from the bottom, so 0 means "show the newest
    screenful" and paging just walks this backwards.
    """
    return max(0, len(_conversation_lines()) - _window_height() - _scrollback)


def _scroll_position(window):
    return _desired_top()


# One notch of the wheel. Three lines is what terminals and editors
# have settled on; a full page per notch overshoots badly on a trackpad,
# which sends a flurry of these.
WHEEL_LINES = 3


def _scroll_by(lines):
    """Move the view. Positive is backwards in time.

    Clamped at both ends: 0 is pinned to the newest line, and the top
    stop is the oldest line that can still fill the pane. Without the
    upper clamp the wheel happily winds _scrollback into the thousands
    and then needs the same number of notches back before anything
    moves again.
    """
    global _scrollback

    _scrollback = max(0, min(
        _scrollback + lines,
        max(0, len(_conversation_lines()) - _window_height()),
    ))
    _refresh()


def _conversation_cursor():
    """Report the cursor on the same line we're scrolling to.

    Both halves are needed. get_vertical_scroll alone gets overridden:
    Window applies it, then runs a cursor-following pass that drags the
    view back to wherever the cursor is - line 0 by default. And the
    cursor alone isn't enough either, because that pass scrolls
    *minimally*, which makes paging lopsided (the first PgUp does
    nothing, PgDn sticks). Pointing both at the same line leaves the
    library with nothing to correct.
    """
    return Point(x=0, y=_desired_top())


# Counters for /scroll, so "the wheel doesn't work" can be split into
# "the terminal never sent it" and "it arrived and nothing moved".
_events = {"conv_wheel": 0, "help_wheel": 0, "other_mouse": 0,
           "page_keys": 0, "arrow_keys": 0}


class _ConversationControl(FormattedTextControl):
    """The conversation pane, with a wheel that actually scrolls it.

    Window handles wheel events on its own by nudging `vertical_scroll`
    - which does nothing here, because `get_vertical_scroll` recomputes
    that from `_scrollback` on every render and overwrites it. So the
    wheel was silently dead whenever mouse capture was on.

    Moving `_scrollback` instead puts the wheel and PgUp/PgDn on the
    same mechanism, with the same clamps, so they can't disagree about
    where the view is.
    """

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            _events["conv_wheel"] += 1
            _scroll_by(WHEEL_LINES)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            _events["conv_wheel"] += 1
            _scroll_by(-WHEEL_LINES)
            return None

        _events["other_mouse"] += 1

        return super().mouse_handler(mouse_event)


# ---------------------------------------------------------------------------
# Scrolling the help window
#
# Same machinery as the conversation, measured from the other end: the
# conversation counts back from the newest line because that is where
# you live, help counts down from the top because it's a document. Both
# route the wheel through their own counter for the reason documented on
# _ConversationControl - Window's own wheel handling is overwritten by
# get_vertical_scroll on every render.
# ---------------------------------------------------------------------------
def _help_line_count():
    return sum(text.count("\n") for _style, text in _help_fragments())


def _help_height():
    if _help_window is not None and _help_window.render_info is not None:
        return max(1, _help_window.render_info.window_height)

    return 10


def _help_scroll_by(lines):
    """Positive is further down the page."""
    global _help_scroll

    _help_scroll = max(0, min(
        _help_scroll + lines,
        max(0, _help_line_count() - _help_height()),
    ))
    _refresh()


def _help_scroll_position(window):
    return _help_scroll


def _help_cursor():
    return Point(x=0, y=_help_scroll)


class _HelpControl(FormattedTextControl):
    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            _events["help_wheel"] += 1
            _help_scroll_by(-WHEEL_LINES)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            _events["help_wheel"] += 1
            _help_scroll_by(WHEEL_LINES)
            return None

        _events["other_mouse"] += 1

        return super().mouse_handler(mouse_event)


def scroll_report():
    """Everything /scroll needs to say where the wheel is getting lost."""
    import os
    import prompt_toolkit

    conv_lines = len(_conversation_lines())

    return (
        f"prompt_toolkit {prompt_toolkit.__version__} | TERM={os.environ.get('TERM', '?')} "
        f"| mouse capture: {'ON' if _mouse else 'OFF (F2)'}\n"
        f"conversation: {conv_lines} lines, pane {_window_height()} tall, "
        f"scrollback {_scrollback} (max {max(0, conv_lines - _window_height())})\n"
        f"help: {_help_line_count()} lines, pane {_help_height()} tall, "
        f"scroll {_help_scroll}, visible={_help_visible}\n"
        f"events since start: conversation wheel {_events['conv_wheel']}, "
        f"help wheel {_events['help_wheel']}, other mouse {_events['other_mouse']}, "
        f"PgUp/PgDn {_events['page_keys']}, arrows {_events['arrow_keys']}"
    )


# ---------------------------------------------------------------------------
# Permission popups
#
# A tool that wants to write a file calls ask_approval() from the worker
# thread and blocks. The TUI floats a box over the conversation with the
# path, what's about to happen, and the content or diff; Y or N on the
# keyboard answers it; no answer before the timeout is a no. It's the one
# place the model's turn waits on you rather than the other way round -
# and the reason it can be trusted with a write tool at all.
# ---------------------------------------------------------------------------
def ask_approval(title, body, note="", timeout=120):
    """Show the request and wait. Returns True only on an explicit yes.

    Safe to call from any thread. Headless (no app running) is a no:
    nothing gets written on the strength of a question nobody saw.
    """
    global _approval, _approval_scroll

    if _app is None:
        return False

    with _approval_lock:
        answered = threading.Event()
        request = {
            "title": title,
            "note": note,
            "body": [str(line) for line in body],
            "answer": None,
            "event": answered,
            "deadline": time.monotonic() + max(5, float(timeout)),
        }

        with _lock:
            _approval = request
            _approval_scroll = 0
            previous_status = state["status"]
            state["status"] = "Waiting for you"

        _refresh()

        # A ticker so the countdown in the title moves without keypresses.
        def tick():
            while not answered.is_set():
                _refresh()
                answered.wait(1.0)

        threading.Thread(target=tick, daemon=True).start()
        answered.wait(max(5, float(timeout)))

        with _lock:
            answer = request["answer"]
            _approval = None
            state["status"] = previous_status

        if answer is None:
            logbook.warn("ui", "approval timed out: %s", title)

        _refresh()

        return answer is True


def approving():
    return _approval is not None


def _answer_approval(yes):
    with _lock:
        request = _approval

    if request is None:
        return

    request["answer"] = bool(yes)
    request["event"].set()


def _approval_title():
    request = _approval

    if request is None:
        return "Permission"

    left = max(0, int(request["deadline"] - time.monotonic()))

    return f"Permission - Y to allow, N to deny - {left}s"


def _approval_header():
    request = _approval

    if request is None:
        return []

    fragments = [("class:approval.title", f" {request['title']}\n")]

    if request["note"]:
        fragments.append(("class:footer", f" {request['note']}\n"))

    return fragments


def _approval_body():
    """The content or diff, coloured the way a diff reads."""
    request = _approval

    if request is None:
        return []

    fragments = []

    for line in request["body"]:
        if line.startswith(("+++", "---")):
            style = "class:footer"
        elif line.startswith("@@"):
            style = "class:diff.hunk"
        elif line.startswith("+"):
            style = "class:diff.add"
        elif line.startswith("-"):
            style = "class:diff.del"
        else:
            style = "class:text"

        fragments.append((style, f" {line}\n"))

    return fragments or [("class:footer", " (nothing to show)\n")]


def _approval_lines():
    return len(_approval["body"]) if _approval else 0


def _approval_height():
    if _approval_window is not None and _approval_window.render_info is not None:
        return max(1, _approval_window.render_info.window_height)

    return 10


def _approval_scroll_by(lines):
    global _approval_scroll

    _approval_scroll = max(0, min(
        _approval_scroll + lines,
        max(0, _approval_lines() - _approval_height()),
    ))
    _refresh()


def _approval_scroll_position(window):
    return _approval_scroll


def _approval_cursor():
    return Point(x=0, y=_approval_scroll)


class _ApprovalControl(FormattedTextControl):
    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            _approval_scroll_by(-WHEEL_LINES)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            _approval_scroll_by(WHEEL_LINES)
            return None

        return super().mouse_handler(mouse_event)


def _approval_size():
    """(width, height) for the float: most of the screen, not all of
    it, so it reads as a window over the conversation."""
    try:
        size = _app.output.get_size()
        columns, rows = size.columns, size.rows
    except Exception:
        columns, rows = 100, 30

    return max(40, int(columns * 0.8)), max(10, int(rows * 0.7))


def _keyed(text, label_style="class:footer"):
    """Colour the (key) parts like sfav does, leave the labels muted."""
    fragments = []

    for piece in re.split(r"(\([^)]*\))", text):
        if not piece:
            continue

        style = "class:key" if piece.startswith("(") else label_style
        fragments.append((style, piece))

    return fragments


# What HOME does depends on the recording mode, so the help says what
# it will actually do right now rather than something generic.
_HOME_BY_MODE = {
    "auto": "start listening - silence ends the turn",
    "manual": "start recording - press again to stop",
    "open": "turn hands-free listening on or off",
}

# A row is (command, description); a bare string renders as a dim note
# line, which is how sections get sentences instead of staying terse.
_HELP_SECTIONS = [
    ("Just say it", [
        "Most of what she can do has no command - asking is the command.",
        "These all trigger real tool calls:",
        "",
        '  "remind me in 20 minutes to stretch"',
        '  "wake me up at 7:30" - or "every weekday at 6"',
        '  "remember that I stream on tuesdays"',
        '  "what do you remember about my gpu?"',
        '  "search for the LM Studio 0.4 changelog"',
        '  "what time is it?" / "how long until friday?"',
        '  "turn the music down" / "what\'s on my clipboard?"',
        '  "how much vram have I got free?"',
        '  "what\'s on my screen?" - needs vision on and a vision model',
        '  "write me a bash script in ~/scripts that backs up my configs"',
        '  "open ~/notes.txt and fix the typo in the second line"',
    ]),
    ("Keys", [
        ("(HOME)", None),  # filled in from the current mode
        ("(Enter)", "send message"),
        ("(PgUp/PgDn)", "scroll the pane - conversation or this help"),
        ("(wheel)", "same - arrow keys too"),
        ("(F2)", "mouse capture: wheel scroll vs selecting text"),
        ("(End)", "back to newest / top of help"),
        ("(Tab)", "close this window"),
        ("(ESC)", "quit"),
    ]),
    ("Voice modes", [
        ("/mode", None),  # gets the (now: ...) suffix
        "auto:   HOME arms the mic, silence ends the turn",
        "manual: HOME starts recording, HOME again stops it",
        "open:   hands-free - she listens continuously",
        "",
        ("/mic", "levels from the last recording - is she hearing you?"),
        ("/barge", "talk-over-her diagnostics, with the thresholds"),
        ("/wake", "wake word status and live scores (openwakeword)"),
    ]),
    ("Her voice", [
        ("/voice", "list what the TTS server offers, current first"),
        ("/voice bella", "switch - partial names match, saved to config"),
        ("/voice luna", "blends work too, if the server defines them"),
        "pitch is a setting, not a voice: /set tts.pitch 3 reads",
        "younger, -3 older, and it applies to the next sentence.",
    ]),
    ("Reminders", [
        ("/reminders", "list what's scheduled, numbered, with countdowns"),
        ("/cancel N", "cancel by number - alarms live in the same list"),
        ("/when ...", "dry-run the time parser, schedules nothing"),
        "repeats all work: \"every monday at 9\", \"daily at 7:30\",",
        "\"every 30 minutes\", \"weekdays at 8\". Said or typed.",
    ]),
    ("Alarms", [
        ("/alarm 7:30am", "set one - any time phrase works"),
        ("/alarm every weekday at 6", ""),
        ("/alarms", "list them (cancel with /cancel)"),
        ("/snooze [n]", "ring again in n minutes - bare /snooze uses the default"),
        ("/alarm off", "stop one that's ringing - talking over it also works"),
        ("/alarm test", "hear the tone at alarm volume"),
        ("/when alarm 6", "how an alarm reads a time - bare hours mean morning"),
        "she speaks a line written for the alarm, plays the tone, and",
        "repeats until dismissed - louder and less charming each round.",
        "one due while the app was closed is skipped, not rung at noon.",
    ]),
    ("Memory", [
        ("/facts", "what she remembers, and which backend holds it"),
        ("/facts all", "the whole table, not just the newest"),
        ("/facts retired", "superseded facts - kept, hidden from her"),
        "json backend: the original capped list in memory.json.",
        "sqlite backend (experimental): permanent, searched, and a new",
        "fact can retire the old one it contradicts. Switch live with",
        "/set long_term_memory.backend sqlite|json - loses nothing.",
        "memory-manager/start.sh opens the editor in your browser.",
    ]),
    ("Tools", [
        ("/tools", "what the model can call - and what's off, and why"),
        ("/tooltest", "canned requests, two ways each: does this model"),
        "                actually pick the right tool? Run it after changing models.",
        ("/repair", "rewrite old keyword-rescued reminders as the tool"),
        "                calls they should have been - teaches by example.",
    ]),
    ("Files", [
        "she can list and read anything under ~ (minus keys, credentials",
        "and shell startup files), and write or edit with your say-so:",
        "every write pops a permission window with the content or the",
        "diff. Y or Enter allows, N or Esc denies, PgUp/PgDn scrolls.",
        "No answer in files.approval_timeout seconds is a no.",
        "/set files.enabled false turns the whole thing off.",
    ]),
    ("Plugins", [
        ("/plugins", "list add-ons and whether each is running"),
        ("/pomf on", "stream chat - /<name> on|off|status for any plugin"),
        "chat-triggered turns get a restricted tool set and never",
        "touch conversation history.",
    ]),
    ("Session", [
        ("/look", "list windows, or test a screenshot"),
        ("/log", "tail the debug log without leaving the app"),
        ("/mouse", "same as F2, and saves the choice"),
        ("/set", "alone lists every setting with its value; /set a.b x"),
        "                changes one, saved to config.json, most apply live.",
        ("/keys", "hotkey + socket diagnostics"),
        ("/clear", "wipe conversation and saved history - not memory"),
        ("/quit", "exit"),
    ]),
]


def _help_fragments():
    mode = state.get("mode", "auto")
    fragments = [("", "\n")]  # breathing room under the title

    for section, rows in _HELP_SECTIONS:
        fragments.append(("class:label bold", f" {section}\n"))

        for entry in rows:
            # A bare string is a note line - dim, indented, no column.
            if isinstance(entry, str):
                fragments.append(
                    ("class:footer", f"   {entry}\n" if entry else "\n")
                )
                continue

            key, description = entry

            if key == "(HOME)":
                description = _HOME_BY_MODE.get(mode, _HOME_BY_MODE["auto"])
            elif key == "/mode":
                description = f"auto | manual | open   (now: {mode})"

            style = "class:key" if key.startswith("(") else "class:agent"
            # Pad to the column, but never glue a long command straight
            # onto its description.
            fragments.append((style, f"   {key:<15}"))
            fragments.append(
                ("class:text", f" {description}\n" if description else "\n")
            )

        fragments.append(("", "\n"))

    note = state.get("controls")

    if note:
        fragments.extend(_keyed(f" {note}", "class:footer"))
        fragments.append(("", "\n"))

    return fragments


# ---------------------------------------------------------------------------
# Rounded frame
#
# prompt_toolkit's Frame draws square corners and centres the title
# between two rules, which renders as "───|  Title  |───". This draws
# rounded corners with the title sitting on the top rule, left-aligned,
# the way sfav does it.
# ---------------------------------------------------------------------------
_TOP_LEFT, _TOP_RIGHT = "╭", "╮"
_BOTTOM_LEFT, _BOTTOM_RIGHT = "╰", "╯"
_HORIZONTAL, _VERTICAL = "─", "│"


def _rule(char, width=None):
    return Window(char=char, style="class:frame.border", width=width, height=1)


def _framed(body, title=None):
    if title is None:
        top = VSplit([
            _rule(_TOP_LEFT, 1), _rule(_HORIZONTAL), _rule(_TOP_RIGHT, 1),
        ], height=1)
    else:
        def label():
            text = title() if callable(title) else title
            return [("class:frame.label", f" {text} ")]

        top = VSplit([
            _rule(_TOP_LEFT, 1),
            _rule(_HORIZONTAL, 1),
            Window(
                content=FormattedTextControl(label),
                height=1,
                dont_extend_width=True,
            ),
            _rule(_HORIZONTAL),
            _rule(_TOP_RIGHT, 1),
        ], height=1)

    return HSplit([
        top,
        VSplit([
            Window(char=_VERTICAL, style="class:frame.border", width=1),
            body,
            Window(char=_VERTICAL, style="class:frame.border", width=1),
        ]),
        VSplit([
            _rule(_BOTTOM_LEFT, 1), _rule(_HORIZONTAL), _rule(_BOTTOM_RIGHT, 1),
        ], height=1),
    ])


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def _build_style():
    return Style.from_dict({
        "frame.border": THEME["border"],
        "frame.label": f"{THEME['title']} bold",
        "label": THEME["label"],
        "text": THEME["text"],
        "value": THEME["value"],
        "agent": THEME["agent"],
        "user": THEME["user"],
        # Viewers get the link colour rather than the user colour: at a
        # glance it should be obvious which lines came from the room
        # and which came from you.
        "guest": f"{THEME['link']} bold",
        "system": THEME["system"],
        "ok": THEME["ok"],
        "warn": f"{THEME['warn']} bold",
        "footer": THEME["footer"],
        "key": f"{THEME['accent']} bold",
        "dim": THEME["dim"],
        "prompt": f"{THEME['prompt']} bold",
        "link": f"{THEME['link']} underline",
        # The help scrollbar: track in the theme's dim, thumb in the
        # border purple, so it reads as part of the frame.
        "scrollbar.background": f"bg:{THEME['dim']}",
        "scrollbar.button": f"bg:{THEME['border']}",
        # The permission popup: title in the warning colour so it can't
        # be mistaken for a chat message, diff lines the way diffs read.
        "approval.title": f"{THEME['warn']} bold",
        "diff.add": THEME["ok"],
        "diff.del": THEME["warn"],
        "diff.hunk": THEME["dim"],
    })


def _build_keys():
    keys = KeyBindings()

    @keys.add("home")
    def _(event):
        # Push-to-talk. Overrides "cursor to start of line" in the input
        # box, which is what we want - Ctrl+A still does that.
        if _on_hotkey:
            _on_hotkey()

    @keys.add("f2")
    def _(event):
        # Mouse capture on means the wheel scrolls; off means the
        # terminal can select text again. You can't have both, because
        # the terminal hands drags to whoever asked for them.
        on = toggle_mouse()
        add_message(
            "system",
            "Mouse capture on - the wheel scrolls the conversation. Most "
            "terminals still let you select with Shift held down. F2 "
            "again to swap back."
            if on else
            "Mouse capture off - select and copy normally. PgUp/PgDn "
            "scroll; F2 to get the wheel back.",
        )

    # While a permission popup is up, the keyboard belongs to it. Y/N
    # (and Enter/Esc) answer; the scroll keys scroll it; everything else
    # is swallowed so a half-typed message can't leak into the input
    # box while you're reading a diff. Exact keys beat Keys.Any in
    # prompt_toolkit, so the swallow doesn't eat the answers.
    asking = Condition(approving)

    @keys.add("y", filter=asking)
    @keys.add("Y", filter=asking)
    @keys.add("enter", filter=asking)
    def _(event):
        _answer_approval(True)

    @keys.add("n", filter=asking)
    @keys.add("N", filter=asking)
    def _(event):
        _answer_approval(False)

    @keys.add(Keys.Any, filter=asking)
    def _(event):
        pass

    @keys.add("tab")
    def _(event):
        global _help_visible, _help_scroll

        if approving():
            return

        _help_visible = not _help_visible

        if _help_visible:
            _help_scroll = 0  # always open at the top

        _refresh()

    @keys.add("escape", eager=True)
    def _(event):
        # Esc answers a permission popup with no, closes the help window
        # if it's open, quits otherwise - same as sfav's notes popup.
        global _help_visible

        if approving():
            _answer_approval(False)
            return

        if _help_visible:
            _help_visible = False
            _refresh()
            return

        event.app.exit()

    @keys.add("c-c")
    @keys.add("c-q")
    def _(event):
        event.app.exit()

    @keys.add("pageup")
    def _(event):
        # A page, not a few lines: the view only moves once the cursor
        # leaves the viewport, so nudging it 5 lines looks like nothing
        # happened. The keys drive whichever pane is showing.
        _events["page_keys"] += 1

        if approving():
            _approval_scroll_by(-max(1, _approval_height() - 2))
        elif _help_visible:
            _help_scroll_by(-max(1, _help_height() - 2))
        else:
            _scroll_by(_page_size())

    @keys.add("pagedown")
    def _(event):
        _events["page_keys"] += 1

        if approving():
            _approval_scroll_by(max(1, _approval_height() - 2))
        elif _help_visible:
            _help_scroll_by(max(1, _help_height() - 2))
        else:
            _scroll_by(-_page_size())

    # With mouse capture off, terminals don't drop wheel events on the
    # floor - in a full-screen app they translate each notch into a few
    # Up/Down arrow presses ("alternate scroll" mode). Those landed in
    # the input box, which read them as "recall the last thing I typed".
    # Routing the arrows to the visible pane means the wheel scrolls
    # with capture OFF as well as on, which is the better default: you
    # keep text selection and get the wheel for free. One line per key
    # because a notch is already several keys.
    @keys.add("up")
    def _(event):
        _events["arrow_keys"] += 1

        if approving():
            _approval_scroll_by(-1)
        elif _help_visible:
            _help_scroll_by(-1)
        else:
            _scroll_by(1)

    @keys.add("down")
    def _(event):
        _events["arrow_keys"] += 1

        if approving():
            _approval_scroll_by(1)
        elif _help_visible:
            _help_scroll_by(1)
        else:
            _scroll_by(-1)

    @keys.add("end")
    def _(event):
        # Back to the resting position of whichever pane is showing:
        # the top of the help, the newest line of the conversation.
        global _scrollback, _help_scroll

        if _help_visible:
            _help_scroll = 0
        else:
            _scrollback = 0

        _refresh()

    return keys


def run(on_submit, on_hotkey=None):
    """Build and run the TUI. Blocks until the user quits.

    on_submit(text) is called on the UI thread - it must not block, so
    anything slow belongs on a worker thread.
    """
    global _app, _on_submit, _on_hotkey, _conv_window, _help_window
    global _approval_window

    _on_submit = on_submit
    _on_hotkey = on_hotkey

    def accept(buffer):
        text = buffer.text.strip()

        if text and _on_submit:
            _on_submit(text)

        return False  # clear the input box

    input_area = TextArea(
        height=1,
        prompt=[("class:prompt", "> ")],
        multiline=False,
        wrap_lines=False,
        accept_handler=accept,
    )

    # The explicit Dimension matters: without it, Window asks the control
    # how tall it wants to be, FormattedTextControl answers "as tall as
    # the whole conversation", and the frame shoves the header off the
    # top of the screen and the input box off the bottom. Giving it a
    # weight instead makes it take the leftover space and scroll.
    conversation_window = Window(
        content=_ConversationControl(
            _conversation_fragments,
            get_cursor_position=_conversation_cursor,
        ),
        height=Dimension(min=3, weight=1),
        wrap_lines=False,
        get_vertical_scroll=_scroll_position,
        always_hide_cursor=True,
    )

    _conv_window = conversation_window

    # Same shape as the conversation window, plus a scrollbar - the help
    # outgrew one screen the day alarms landed, and a pane that cuts off
    # silently reads as "that's all there is".
    help_window = Window(
        content=_HelpControl(
            _help_fragments,
            get_cursor_position=_help_cursor,
        ),
        height=Dimension(min=3, weight=1),
        wrap_lines=False,
        get_vertical_scroll=_help_scroll_position,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin()],
    )

    _help_window = help_window

    # Help replaces the conversation panel rather than floating over it.
    # A float has to be full width anyway - a 2-cell emoji whose first
    # half falls outside the float still gets drawn in full and shunts
    # the border a column right, and Luna uses emoji constantly - and a
    # full-width float left the conversation's leftover rows and bottom
    # border poking out underneath. Swapping is exact and artifact-free.
    showing_help = Condition(lambda: _help_visible)

    body = HSplit([
        _framed(
            Window(
                content=FormattedTextControl(_status_lines),
                height=Dimension.exact(5),
            ),
            title=lambda: f"{state['agent_name']} AI Assistant",
        ),
        ConditionalContainer(
            _framed(conversation_window, title="Conversation"),
            filter=~showing_help,
        ),
        ConditionalContainer(
            _framed(help_window, title="Help"),
            filter=showing_help,
        ),
        _framed(input_area, title="tab for help"),
    ])

    # The permission popup floats over everything, centred, most of the
    # screen but not all of it. Its body scrolls the same way the other
    # panes do.
    approval_body = Window(
        content=_ApprovalControl(
            _approval_body,
            get_cursor_position=_approval_cursor,
        ),
        wrap_lines=False,
        get_vertical_scroll=_approval_scroll_position,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin()],
    )
    _approval_window = approval_body

    approval_box = ConditionalContainer(
        _framed(
            HSplit([
                Window(
                    content=FormattedTextControl(_approval_header),
                    height=Dimension(min=1, max=3),
                    dont_extend_height=True,
                ),
                _rule(_HORIZONTAL),
                approval_body,
                _rule(_HORIZONTAL),
                Window(
                    content=FormattedTextControl(lambda: _keyed(
                        " (Y) allow  (N) deny  (PgUp/PgDn) scroll  (Esc) deny"
                    )),
                    height=1,
                ),
            ]),
            title=_approval_title,
        ),
        filter=Condition(approving),
    )

    root = FloatContainer(
        content=body,
        floats=[
            Float(
                content=approval_box,
                width=lambda: _approval_size()[0],
                height=lambda: _approval_size()[1],
            ),
        ],
    )

    layout = Layout(root, focused_element=input_area)

    _app = Application(
        layout=layout,
        key_bindings=_build_keys(),
        style=_build_style(),
        full_screen=True,
        # A filter, not a flag, so it can be toggled without a restart.
        mouse_support=Condition(lambda: _mouse),
    )

    # First paint has no render_info, so the scroll lands at line one and
    # nothing would move it until the next keystroke - which is how a
    # freshly loaded history ends up showing its oldest messages. Repaint
    # once more, then get out of the way.
    primed = []

    def _prime(_sender):
        if not primed:
            primed.append(True)
            _refresh()

    _app.after_render += _prime

    _app.run()


def stop():
    if _app is not None:
        try:
            _app.exit()
        except Exception:
            pass
