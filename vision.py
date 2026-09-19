"""Letting Luna see the screen.

The awkward part is that a tool result is a string. There is nowhere in
the OpenAI tool-calling shape to hand back an image, so the tool here
captures a screenshot, stashes it, and returns a sentence saying so;
llm._tool_rounds then picks it up and appends it as a user message with
an image block attached. From the model's point of view it asked to
look at something and the next thing it saw was a picture.

Capture is grim, which is the Wayland screenshot tool Hyprland expects.
The active window's geometry comes from hyprctl. Nothing here assumes
the model can actually read images - if it can't, LM Studio says so and
that error is surfaced as-is rather than dressed up.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile

from config import VISION_ENABLED, VISION_SCALE, VISION_MAX_BYTES

# Set by capture(), consumed once by llm._tool_rounds.
_pending = None
last_error = ""


def available():
    """grim is the hard requirement; hyprctl only narrows it to a window."""
    return VISION_ENABLED and shutil.which("grim") is not None


def why_unavailable():
    if not VISION_ENABLED:
        return 'vision is off - set "vision": {"enabled": true} in agent.json'

    if shutil.which("grim") is None:
        return "grim isn't installed - it's the Wayland screenshot tool (dnf install grim)"

    return ""


def _own_pids():
    """Every pid from this process up to the session leader.

    Used to recognise our own terminal. Walking /proc is Linux-only,
    which is the same thing hyprctl already assumes.
    """
    pids = set()
    pid = os.getpid()

    for _ in range(12):
        if pid <= 1 or pid in pids:
            break

        pids.add(pid)

        try:
            with open(f"/proc/{pid}/stat") as stat:
                # comm can contain spaces and brackets, so ppid is read
                # from after the closing one, not by splitting the lot.
                fields = stat.read().rsplit(")", 1)[1].split()

            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break

    return pids


def windows():
    """Every window she could look at, most recently focused first.

    Excludes her own terminal - photographing herself is never the
    answer, and on a tiling setup it's an easy mistake to make.
    """
    if shutil.which("hyprctl") is None:
        return []

    try:
        output = subprocess.run(
            ["hyprctl", "clients", "-j"],
            capture_output=True, text=True, timeout=5,
        )

        if output.returncode != 0:
            return []

        clients = json.loads(output.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []

    if not isinstance(clients, list):
        return []

    ours = _own_pids()
    candidates = [c for c in clients if _usable(c, ours)]
    candidates.sort(key=lambda c: c.get("focusHistoryID", 9999))

    return candidates


# What people call a thing versus what its window class is called.
# "Look at my browser" is the normal way to ask and matches nothing at
# all without this.
GENERIC_NAMES = {
    "browser": ("firefox", "chromium", "chrome", "brave", "zen", "librewolf",
                "vivaldi", "qutebrowser", "epiphany"),
    "editor": ("code", "codium", "vscodium", "subl", "sublime", "nvim",
               "vim", "emacs", "zed", "kate", "gedit", "helix"),
    "terminal": ("kitty", "alacritty", "foot", "wezterm", "konsole",
                 "ghostty", "st", "urxvt"),
    "music": ("ncmpcpp", "spotify", "cmus", "mpd", "rhythmbox", "audacious",
              "strawberry", "tauon"),
    "chat": ("discord", "element", "telegram", "signal", "slack", "vesktop"),
    "files": ("nautilus", "thunar", "dolphin", "nemo", "pcmanfm", "yazi",
              "ranger"),
    "video": ("mpv", "vlc", "celluloid", "obs"),
    "game": ("steam", "lutris", "heroic"),
}

# Reverse it once, so "firefox" also answers to "browser".
_ALIASES = {}

for _generic, _classes in GENERIC_NAMES.items():
    for _cls in _classes:
        _ALIASES.setdefault(_cls, set()).add(_generic)


def find_window(phrase):
    """Match a window by how the user would refer to it.

    "my browser", "the music player", "firefox" - the model passes
    their words straight through and the matching happens here, for the
    same reason set_reminder takes a phrase instead of a timestamp.
    """
    phrase = str(phrase or "").strip().lower()
    candidates = windows()

    if not phrase or not candidates:
        return None

    terms = [t for t in re.split(r"[^a-z0-9]+", phrase) if len(t) > 1]
    # Words people use for a window that never appear in its own name.
    terms = [t for t in terms if t not in
             ("the", "window", "app", "my", "one", "that", "this", "on")]

    if not terms:
        return None

    best, best_score = None, 0.0

    for index, client in enumerate(candidates):
        name = (client.get("class") or "").lower()
        title = (client.get("title") or "").lower()

        # A terminal running ncmpcpp answers to "music" even though its
        # class is kitty, so the title feeds the aliases too.
        generics = set(_ALIASES.get(name, ()))

        for word in re.split(r"[^a-z0-9]+", title):
            generics |= _ALIASES.get(word, set())

        score = 0.0

        for term in terms:
            if term == name:
                score += 3.0
            elif term in name:
                score += 2.0
            elif term in generics:
                score += 2.0
            elif term in title:
                score += 1.0

        if score <= 0:
            continue

        # Nudge towards the window they were in most recently, so a tie
        # between two terminals goes to the one they just left.
        score += (len(candidates) - index) / (len(candidates) * 10.0)

        if score > best_score:
            best, best_score = client, score

    return best


def _usable(client, ours):
    if not client.get("mapped", True) or client.get("hidden", False):
        return False

    if client.get("pid") in ours:
        return False

    size = client.get("size") or [0, 0]

    return size[0] > 1 and size[1] > 1


def _last_window_you_looked_at():
    """The most recently focused window that isn't this assistant.

    This exists because of how you actually use her. Typing to her
    means *her terminal* is the focused window, so "what does this say"
    was photographing herself - a genuinely baffling failure, because
    the screenshot succeeded and the answer was still nonsense.

    Hyprland keeps a focus history, so the window you were reading
    right before you turned to ask about it is one query away.
    """
    if shutil.which("hyprctl") is None:
        return None, ""

    try:
        output = subprocess.run(
            ["hyprctl", "clients", "-j"],
            capture_output=True, text=True, timeout=5,
        )

        if output.returncode != 0:
            return None, ""

        clients = json.loads(output.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, ""

    if not isinstance(clients, list):
        return None, ""

    ours = _own_pids()
    candidates = [c for c in clients if _usable(c, ours)]

    if not candidates:
        return None, ""

    # focusHistoryID 0 is the focused window, 1 is the one before it.
    candidates.sort(key=lambda c: c.get("focusHistoryID", 9999))

    return _describe(candidates[0])


def _describe(window):
    try:
        x, y = window["at"]
        width, height = window["size"]
    except (KeyError, TypeError, ValueError):
        return None, ""

    if width < 1 or height < 1:
        return None, ""

    name = (window.get("class") or "").strip()
    title = (window.get("title") or "").strip()

    if title and len(title) > 60:
        title = title[:57] + "..."

    described = " - ".join(part for part in (name, title) if part) or "a window"

    return f"{x},{y} {width}x{height}", f"{described} ({width}x{height})"


def _active_window():
    """('x,y wxh', 'firefox - Hyprland Wiki') for the focused window.

    The description is the point. "Screenshot taken" tells you nothing
    when the thing she photographed might be her own terminal; naming
    the window makes a wrong capture obvious at a glance instead of
    after a confusing exchange about what she can see.
    """
    if shutil.which("hyprctl") is None:
        return None, ""

    try:
        output = subprocess.run(
            ["hyprctl", "activewindow", "-j"],
            capture_output=True, text=True, timeout=5,
        )

        if output.returncode != 0:
            return None, ""

        window = json.loads(output.stdout)
        x, y = window["at"]
        width, height = window["size"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None, ""

    if width < 1 or height < 1:
        return None, ""

    if window.get("pid") in _own_pids():
        # She's looking at herself. Fall back to whatever you were
        # reading before you turned to type at her.
        geometry, described = _last_window_you_looked_at()

        if geometry:
            return geometry, described

        return None, ""

    return _describe(window)


def _selected_region():
    """Let the user drag a box with slurp. Blocks until they do."""
    if shutil.which("slurp") is None:
        return None, "slurp isn't installed - it's the region picker (dnf install slurp)"

    try:
        output = subprocess.run(
            ["slurp"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)

    if output.returncode != 0:
        return None, "nothing was selected"

    geometry = output.stdout.strip()

    return (geometry, f"the region you selected ({geometry})") if geometry else (
        None, "nothing was selected"
    )


def default_region():
    """What "look at my screen" should mean, given how she was asked.

    Typed at her, the focused window is her own terminal and the
    window behind it is whatever you last touched - which on a tiling
    setup is close to arbitrary. The whole screen is the honest answer
    there, because on Hyprland everything is visible at once anyway.

    Asked by voice through a compositor bind, you deliberately focused
    something before you spoke, so the focused window is exactly right.
    """
    import state

    return "active" if getattr(state, "turn_source", "typed") == "voice" else "full"


def capture(region="auto", window=None):
    """Take a screenshot. Returns (data_url, description) or (None, error).

    Scaled down on the way out: a 4K screenshot is several megabytes of
    base64, which is slow to move and mostly wasted on a vision model
    that resizes it anyway.
    """
    global _pending, last_error

    if not available():
        last_error = why_unavailable()

        return None, last_error

    command = ["grim", "-t", "png"]

    if VISION_SCALE and VISION_SCALE != 1.0:
        command += ["-s", str(VISION_SCALE)]

    if region == "auto":
        region = default_region()

    what = "the whole screen"

    if window:
        match = find_window(window)

        if match is None:
            open_now = ", ".join(
                (c.get("class") or "?") for c in windows()[:8]
            )
            last_error = (
                f"no window matching {window!r}"
                + (f" - open windows are: {open_now}" if open_now else "")
            )

            return None, last_error

        geometry, described = _describe(match)

        if geometry:
            command += ["-g", geometry]
            what = described
    elif region == "select":
        geometry, described = _selected_region()

        if geometry is None:
            last_error = described

            return None, described

        command += ["-g", geometry]
        what = described
    elif region != "full":
        geometry, described = _active_window()

        if geometry:
            command += ["-g", geometry]
            what = described

    handle, path = tempfile.mkstemp(suffix=".png")
    os.close(handle)

    try:
        result = subprocess.run(
            command + [path], capture_output=True, text=True, timeout=15
        )

        if result.returncode != 0:
            last_error = (result.stderr or "grim failed").strip()

            return None, last_error

        size = os.path.getsize(path)

        if size > VISION_MAX_BYTES:
            last_error = (
                f"screenshot is {size // 1024}KB, over the "
                f"{VISION_MAX_BYTES // 1024}KB limit - lower vision.scale "
                "in agent.json"
            )

            return None, last_error

        with open(path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()

        what = f"{what}, {size // 1024}KB"
    except (OSError, subprocess.SubprocessError) as e:
        last_error = str(e)

        return None, last_error
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    _pending = f"data:image/png;base64,{encoded}"
    last_error = ""

    return _pending, what


def take():
    """Hand over the captured image, once."""
    global _pending

    image, _pending = _pending, None

    return image


def pending():
    return _pending is not None
