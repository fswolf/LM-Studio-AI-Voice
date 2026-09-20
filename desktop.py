"""The desktop she is sitting on - audio, clipboard, windows.

vision.py already taught her to *see* the desktop. This is the other
half: acting on it. "pause the music", "what did I just copy", "bring
firefox up" are the things you say to something listening in the room,
and until now the honest answer to all three was a screenshot.

Everything here shells out to whatever is installed rather than binding
a library, for two reasons. The tools differ per setup - wpctl on
PipeWire, pactl on PulseAudio, wl-copy on Wayland - and shelling out
means a missing one is a tool that isn't offered rather than an import
that kills startup. And none of these are hot paths; a 5-second timeout
on a program that normally answers in 20ms is free insurance against a
hung player wedging a turn.

Nothing here is destructive. Volume, playback and focus are all states
you can put back by saying the opposite, and the clipboard is the one
thing that can't be - so writing to it is a separate switch.
"""
import shutil
import subprocess

from config import DESKTOP_ENABLED, DESKTOP_CLIPBOARD

# Long enough for a player that's paging in from disk, short enough
# that a wedged one costs a moment rather than the turn.
TIMEOUT = 5

# How much of the clipboard reaches the model. Enough for a URL, an
# error message or a paragraph; not enough for a pasted file to eat
# the context window.
MAX_CLIPBOARD_CHARS = 2000


def _run(command, stdin=None):
    """Run a command, returning (ok, output). Never raises."""
    try:
        result = subprocess.run(
            command,
            capture_output=True, text=True, timeout=TIMEOUT,
            input=stdin,
        )
    except FileNotFoundError:
        return False, f"{command[0]} isn't installed"
    except subprocess.TimeoutExpired:
        return False, f"{command[0]} didn't answer in {TIMEOUT}s"
    except OSError as e:
        return False, str(e)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()

        return False, detail.splitlines()[0] if detail else "it failed"

    return True, (result.stdout or "").strip()


def _has(*names):
    """The first of these that exists, or None."""
    for name in names:
        if shutil.which(name):
            return name

    return None


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------
def media_available():
    return DESKTOP_ENABLED and shutil.which("playerctl") is not None


def now_playing():
    """'Artist - Title (playing)', or why not."""
    ok, status = _run(["playerctl", "status"])

    if not ok:
        # playerctl says this on stderr and exits non-zero, which is
        # not an error worth reporting as one.
        return "Nothing is playing."

    ok, meta = _run(
        ["playerctl", "metadata", "--format",
         "{{artist}} - {{title}}"]
    )

    if not ok or not meta or meta == " - ":
        return f"Something is {status.lower()}, but it reports no track name."

    return f"{meta} ({status.lower()})"


def playback(action):
    """play / pause / play-pause / next / previous / stop."""
    if action == "status":
        return now_playing()

    ok, detail = _run(["playerctl", action])

    if not ok:
        return f"Couldn't {action}: {detail}"

    # playerctl is silent on success, so say what happened rather than
    # returning an empty string the model has to guess about.
    return {
        "play": "Playing.",
        "pause": "Paused.",
        "play-pause": "Toggled playback.",
        "next": "Skipped to the next track.",
        "previous": "Went back a track.",
        "stop": "Stopped.",
    }.get(action, "Done.") + " " + now_playing()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
# wpctl is PipeWire's, pactl works on both it and PulseAudio. Preferring
# wpctl because on a PipeWire box it's the one that's definitely there.
_SINK = "@DEFAULT_AUDIO_SINK@"


def volume_available():
    return DESKTOP_ENABLED and _has("wpctl", "pactl") is not None


def _wpctl_volume():
    ok, output = _run(["wpctl", "get-volume", _SINK])

    if not ok:
        return None, None

    # "Volume: 0.45" or "Volume: 0.45 [MUTED]"
    parts = output.replace("Volume:", "").strip().split()

    try:
        level = round(float(parts[0]) * 100)
    except (IndexError, ValueError):
        return None, None

    return level, "MUTED" in output.upper()


def _pactl_volume():
    ok, output = _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])

    if not ok:
        return None, None

    percentages = [
        int(chunk.rstrip("%")) for chunk in output.split()
        if chunk.endswith("%") and chunk.rstrip("%").isdigit()
    ]

    if not percentages:
        return None, None

    muted_ok, muted = _run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])

    return percentages[0], bool(muted_ok and "yes" in muted.lower())


def volume_state():
    """(percent, muted), or (None, None) if it can't be read."""
    if shutil.which("wpctl"):
        level, muted = _wpctl_volume()

        if level is not None:
            return level, muted

    if shutil.which("pactl"):
        return _pactl_volume()

    return None, None


def describe_volume():
    level, muted = volume_state()

    if level is None:
        return "Couldn't read the volume."

    return f"Volume is {level}%{' (muted)' if muted else ''}."


def set_volume(level):
    """Set the output volume to a percentage."""
    level = max(0, min(100, int(level)))

    if shutil.which("wpctl"):
        ok, detail = _run(["wpctl", "set-volume", _SINK, f"{level / 100:.2f}"])
    elif shutil.which("pactl"):
        ok, detail = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"])
    else:
        return "No volume control is installed (wpctl or pactl)."

    return f"Volume set to {level}%." if ok else f"Couldn't set the volume: {detail}"


def nudge_volume(step):
    """Up or down by a step, from wherever it is now.

    Read-then-set rather than wpctl's own `5%+`, so the reply can say
    what it landed on. "Turn it down" answered with "done" is not an
    answer when you wanted to know how far down.
    """
    level, _muted = volume_state()

    if level is None:
        return "Couldn't read the volume to change it."

    return set_volume(level + step)


def set_muted(muted):
    if shutil.which("wpctl"):
        ok, detail = _run(["wpctl", "set-mute", _SINK, "1" if muted else "0"])
    elif shutil.which("pactl"):
        ok, detail = _run(
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if muted else "0"]
        )
    else:
        return "No volume control is installed (wpctl or pactl)."

    if not ok:
        return f"Couldn't {'mute' if muted else 'unmute'}: {detail}"

    return "Muted." if muted else "Unmuted. " + describe_volume()


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------
def clipboard_available():
    return (DESKTOP_ENABLED and DESKTOP_CLIPBOARD
            and _has("wl-paste", "xclip", "xsel") is not None)


def why_no_clipboard():
    if not DESKTOP_ENABLED:
        return 'the desktop tools are off - set "desktop": {"enabled": true}'

    if not DESKTOP_CLIPBOARD:
        return 'clipboard access is off - set "desktop": {"clipboard": true}'

    return ("no clipboard tool is installed - wl-clipboard on Wayland "
            "(dnf install wl-clipboard)")


def read_clipboard():
    if shutil.which("wl-paste"):
        # --no-newline stops it inventing a trailing one; -t text asks
        # for the text flavour, so copying an image doesn't come back
        # as a wall of binary.
        ok, text = _run(["wl-paste", "--no-newline", "-t", "text"])
    elif shutil.which("xclip"):
        ok, text = _run(["xclip", "-selection", "clipboard", "-o"])
    elif shutil.which("xsel"):
        ok, text = _run(["xsel", "--clipboard", "--output"])
    else:
        return "No clipboard tool is installed."

    if not ok:
        return "The clipboard is empty, or holds something that isn't text."

    if not text.strip():
        return "The clipboard is empty."

    if len(text) > MAX_CLIPBOARD_CHARS:
        text = text[:MAX_CLIPBOARD_CHARS].rsplit(" ", 1)[0] + "..."

        return (f"The clipboard holds (first part only):\n{text}")

    return f"The clipboard holds:\n{text}"


def write_clipboard(text):
    text = str(text or "")

    if not text:
        return "Nothing to copy."

    if shutil.which("wl-copy"):
        ok, detail = _run(["wl-copy"], stdin=text)
    elif shutil.which("xclip"):
        ok, detail = _run(["xclip", "-selection", "clipboard"], stdin=text)
    elif shutil.which("xsel"):
        ok, detail = _run(["xsel", "--clipboard", "--input"], stdin=text)
    else:
        return "No clipboard tool is installed."

    if not ok:
        return f"Couldn't copy that: {detail}"

    shown = text if len(text) <= 60 else text[:60] + "..."

    return f"Copied to the clipboard: {shown}"


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def windows_available():
    return DESKTOP_ENABLED and shutil.which("hyprctl") is not None


def focus(phrase):
    """Bring a window to the front, matched the way the user named it.

    The matching is vision.find_window's, deliberately - "my browser"
    should mean the same window whether she is looking at it or
    switching to it, and two matchers would drift apart within a week.
    """
    import vision

    client = vision.find_window(phrase)

    if client is None:
        open_now = vision.windows()

        if not open_now:
            return "No windows are open to switch to."

        names = ", ".join(
            sorted({(c.get("class") or "?") for c in open_now})[:8]
        )

        return f"No window matches {phrase!r}. Open right now: {names}"

    address = client.get("address")

    if not address:
        return "Found the window but it has no address to focus."

    ok, detail = _run(["hyprctl", "dispatch", "focuswindow", f"address:{address}"])

    if not ok:
        return f"Couldn't focus it: {detail}"

    title = (client.get("title") or client.get("class") or "it").strip()

    if len(title) > 70:
        title = title[:70] + "..."

    return f"Switched to {title}."
