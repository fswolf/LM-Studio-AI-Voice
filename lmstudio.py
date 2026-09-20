"""Whether LM Studio is actually there, and which model it's holding.

The TTS server has had reachability tracking since the kokoro-reader
migration: a failed request flips a flag, the header shows [offline],
and it recovers on its own when the server comes back. LM Studio had
none of that. It was checked once at startup and never again, so if it
crashed, or you unloaded the model to free VRAM, or swapped to a
different one, every turn from then on failed with a raw error and no
indication of why.

That is the same problem twice, so this is the same solution twice.
"""
import threading

import requests

from config import LM_URL

MODELS_URL = LM_URL.replace("/chat/completions", "/models")

_lock = threading.Lock()

ok = False
error = ""
model = ""
# Set when the loaded model changes underneath us, so the turn that
# notices can say so rather than silently answering as someone else.
changed_to = ""


def label():
    """The Model row: what's loaded, or why nothing is."""
    with _lock:
        if ok:
            return model or "unknown"

        if not model:
            return f"offline ({error})" if error else "offline"

        return f"{model} [offline]"


def probe(timeout=5):
    """Ask what's loaded. Returns the model id, or "" if unreachable."""
    global ok, error, model, changed_to

    try:
        response = requests.get(MODELS_URL, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        loaded = (data.get("data") or [{}])[0].get("id", "")
    except Exception as e:
        _set(False, str(e))

        return ""

    with _lock:
        previous = model

    _set(True, "", loaded)

    if previous and loaded and loaded != previous:
        with _lock:
            changed_to = loaded

    return loaded


def _set(reachable, why="", loaded=None):
    global ok, error, model

    with _lock:
        changed = reachable != ok
        ok = reachable
        error = why

        if loaded is not None:
            model = loaded

    if changed:
        try:
            import ui

            ui.set_model(label())
        except Exception:
            pass


def mark_failed(why):
    """Called when a request fails mid-conversation."""
    _set(False, str(why)[:120])


def mark_worked():
    if not ok:
        _set(True, "")


def take_change():
    """The new model id, once, if it changed since we last looked."""
    global changed_to

    with _lock:
        was, changed_to = changed_to, ""

    return was


def watch(interval=20):
    """Poll in the background so the header is right even between turns.

    Cheap - one GET every twenty seconds against a local server - and
    it means an unloaded model shows up in the header immediately
    instead of on the next thing you say.
    """
    def loop():
        while True:
            probe(timeout=4)
            threading.Event().wait(interval)

    threading.Thread(target=loop, daemon=True).start()
