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
# LM Studio's own (non-OpenAI) endpoint: the only place it reports the
# context length a model was actually loaded with.
NATIVE_MODELS_URL = LM_URL.split("/v1/")[0] + "/api/v0/models"

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


def context_length(timeout=3):
    """Tokens the loaded model was given, or 0 if LM Studio won't say.

    This is the number every "HTTP 500" out of nowhere is really about:
    the prompt - persona, facts, summary, fifteen turns and the tool
    schemas - has to fit in it, and the schemas alone are a few
    thousand tokens now.
    """
    try:
        response = requests.get(NATIVE_MODELS_URL, timeout=timeout)
        response.raise_for_status()

        for entry in response.json().get("data") or []:
            if entry.get("state") == "loaded" or entry.get("loaded_context_length"):
                return int(entry.get("loaded_context_length")
                           or entry.get("max_context_length") or 0)
    except Exception:
        pass

    return 0


def estimate_tokens(payload):
    """Rough size of a chat payload. 3.5 chars/token is close enough for
    English plus JSON to say "this is 90% of the window"."""
    try:
        import json

        return int(len(json.dumps(payload)) / 3.5)
    except Exception:
        return 0
