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


def capture(region="active"):
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

    what = "the whole screen"

    if region == "select":
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
