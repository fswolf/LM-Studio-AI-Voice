"""A worked example: copy this to write a chat plugin.

It connects to nothing. Feed it messages by hand to watch the whole
path run:

    >>> import plugins
    >>> plugins.load()
    >>> p = plugins.get("example")
    >>> p.start("your-model")
    >>> from plugins import example
    >>> example.pretend("someone", "luna are you there?")

Everything about *who gets answered* - her name, rate limits, the
untrusted framing, the tool ceiling, waiting for her turn - is in
chatroom.py and comes for free. What a real plugin adds is the two
marked spots: connecting, and turning whatever arrives into
(name, message).

## Writing the YouTube one

YouTube live chat is polled, not pushed: with an API key, GET
`liveChat/messages` with the `liveChatId`, and the response carries
`pollingIntervalMillis` telling you when to come back. So `_run()`
becomes a poll loop instead of a socket read, `_room.saw()` is called
with `authorDetails.displayName` and `snippet.displayMessage`, and
nothing else in this file changes. Keep the API key out of
config.json the way plugins/pomf.py does - config.json is in the repo.
"""
import threading
import time

import chatroom
import config

# Required. This is what /<name> on|off calls it, and the key its
# settings live under in config.json.
NAME = "example"
SUMMARY = "a template - connects to nothing, answers nothing"

_room = None
_thread = None
_stop = threading.Event()
_running = False


def settings():
    """This plugin's block from config.json, merged over the shared
    chat defaults. The core never learns what these keys mean."""
    return config.plugin_settings(NAME)


def available():
    """Can it run right now? Check for the library and the credentials
    here, not at import time - a plugin that raises on import is a
    plugin that doesn't load at all."""
    return True


def why_unavailable():
    """If available() is False, a reason someone can act on. "not
    configured" helps nobody; name the file or the pip install."""
    return ""


# ---------------------------------------------------------------------------
# 1. Connecting - the part that differs per service
# ---------------------------------------------------------------------------
def _run():
    """Receive forever, handing each message to _room.saw().

    Reconnect in here rather than letting the thread die: chat drops
    and comes back, and a plugin that needs restarting by hand after
    every blip isn't one you'd leave on during a stream.
    """
    while not _stop.is_set():
        # A socket read, or an HTTP poll, or a subprocess - whatever
        # the service speaks. For the shape of a real one, see
        # plugins/pomf.py.
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# 2. Handing messages over - one line, and the only thing that matters
# ---------------------------------------------------------------------------
def pretend(who, message):
    """Stand-in for a real message arriving. saw() does the rest:
    ignores anything not addressed to her, applies the rate limits,
    queues it, and answers it out loud when she's free."""
    if _room is not None:
        _room.saw(who, message)


# ---------------------------------------------------------------------------
# The plugin interface
# ---------------------------------------------------------------------------
def start(model):
    global _room, _thread, _running

    if _running:
        return False, f"{NAME} is already running."

    problem = why_unavailable()

    if problem:
        return False, problem

    _room = chatroom.ChatRoom(
        NAME,
        owner=str(settings().get("channel", "")) or None,
        settings=settings(),
    )
    _room.start(model)

    _stop.clear()
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    _running = True

    return True, f"{NAME} is listening."


def stop():
    global _running

    if not _running:
        return False, f"{NAME} isn't running."

    _stop.set()

    if _room is not None:
        _room.stop()

    _running = False

    return True, f"{NAME} stopped."


def running():
    return _running


def status():
    if not _running:
        return f"{NAME}: off"

    return "\n".join([f"{NAME}: running"] + _room.summary())
