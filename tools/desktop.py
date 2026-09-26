"""Her hands and eyes on the machine - screen, audio, clipboard,
windows, and what the hardware is doing.

Every one of these is gated on the thing it needs actually being
installed. A tool that cannot work is never offered, because a
model told it can see will describe a screen it never looked at.
"""
import config
import desktop
import machine
import vision

from . import tool


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
@tool(
    "look_at_screen",
    "Take a screenshot of what the user is looking at and see it. Use "
    "this whenever they refer to something on their screen - an error, "
    "a window, a design, 'this', 'what does that say'. Do not ask them "
    "to show you; you can look by yourself. The image arrives in the "
    "next message.",
    {
        "window": {
            "type": "string",
            "description": "Which window, in the user's own words - "
                           "'firefox', 'my editor', 'the music "
                           "player'. Leave it out if they didn't say, "
                           "and the whole screen is captured instead.",
        },
    },
    available=vision.available,
    why=vision.why_unavailable,
)
def _look_at_screen(window=None, whole_screen=False, region=None):
    # `region` and `whole_screen` are accepted but no longer advertised -
    # a model that saw the older schemas in its own recent turns will
    # keep sending them for a while.
    if whole_screen:
        region = "full"
    elif region == "select":
        # Never on the model's say-so: it blocks the whole conversation
        # on a crosshair, and "what does this say" is a request to look
        # at what's already there, not to go hunting with the mouse.
        # /look select exists for when you do want to point.
        region = "active"
    elif region not in ("active", "full"):
        region = "auto"

    image, detail = vision.capture(region, window=window)

    if image is None:
        return (
            f"Couldn't take a screenshot: {detail}. Tell the user this - "
            "do not describe a screen you have not seen."
        )

    return (
        f"Screenshot taken of {detail} - it is attached to the next "
        "message. Describe what you actually see in it; do not guess, "
        "and say so if it isn't what they meant."
    )


# ---------------------------------------------------------------------------
# The desktop
# ---------------------------------------------------------------------------
# One tool rather than six. "Turn it down", "skip this", "what's
# playing" and "mute" are the same request to a person - do something
# to the sound - and splitting them into six schemas costs selection
# accuracy on a small model for no gain the user can feel.
_AUDIO_ACTIONS = ("status", "play", "pause", "next", "previous", "stop",
                  "louder", "quieter", "mute", "unmute", "set_volume")

@tool(
    "control_audio",
    "Control playback and volume on the user's computer: pause or resume "
    "music, skip a track, change or mute the volume, or report what is "
    "playing. Use it whenever they ask about or ask you to change sound.",
    {
        "action": {
            "type": "string",
            "enum": list(_AUDIO_ACTIONS),
            "description": "What to do. 'status' reports what is playing "
                           "and how loud without changing anything.",
        },
        "level": {
            "type": "integer",
            "description": "Volume percent, 0-100. Only for set_volume.",
        },
    },
    required=("action",),
    available=lambda: desktop.media_available() or desktop.volume_available(),
    why=lambda: (
        'the desktop tools are off - set "desktop": {"enabled": true}'
        if not config.DESKTOP_ENABLED else
        "neither playerctl nor wpctl/pactl is installed"
    ),
)
def _control_audio(action, level=None):
    action = str(action or "").strip().lower()

    if action == "set_volume":
        if level is None:
            return "Ask the user what volume they want - no level was given."

        return desktop.set_volume(level)

    if action in ("louder", "quieter"):
        return desktop.nudge_volume(10 if action == "louder" else -10)

    if action in ("mute", "unmute"):
        return desktop.set_muted(action == "mute")

    if action == "status":
        return f"{desktop.now_playing()} {desktop.describe_volume()}"

    if action not in _AUDIO_ACTIONS:
        return f"No audio action called {action!r}. Try: {', '.join(_AUDIO_ACTIONS)}"

    if not desktop.media_available():
        return "There's no media player control installed (playerctl)."

    return desktop.playback(action)

@tool(
    "clipboard",
    "Read what the user last copied, or put text on their clipboard so "
    "they can paste it. Use 'read' when they refer to something they "
    "copied without saying what it is.",
    {
        "action": {
            "type": "string",
            "enum": ["read", "write"],
            "description": "'read' returns the clipboard's contents; "
                           "'write' replaces them.",
        },
        "text": {
            "type": "string",
            "description": "What to copy. Only for 'write'.",
        },
    },
    required=("action",),
    available=desktop.clipboard_available,
    why=desktop.why_no_clipboard,
)
def _clipboard(action, text=None):
    if str(action or "").strip().lower() == "write":
        return desktop.write_clipboard(text)

    return desktop.read_clipboard()

@tool(
    "focus_window",
    "Bring one of the user's windows to the front - switch to it. This "
    "changes what is on screen; to look at a window without switching "
    "to it, use look_at_screen instead.",
    {
        "name": {
            "type": "string",
            "description": "The window as the user named it - 'firefox', "
                           "'my browser', 'the code editor'.",
        },
    },
    required=("name",),
    available=desktop.windows_available,
    why=lambda: (
        'the desktop tools are off - set "desktop": {"enabled": true}'
        if not config.DESKTOP_ENABLED else
        "hyprctl isn't there - window switching needs Hyprland"
    ),
)
def _focus_window(name):
    return desktop.focus(name)

@tool(
    "system_status",
    "Check the computer's own state: free VRAM, GPU temperature and "
    "load, free RAM and disk, and which model LM Studio has loaded. Use "
    "it before answering whether something will fit or why things are "
    "slow - never guess these numbers.",
    {},
)
def _system_status():
    return machine.describe()
