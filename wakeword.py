"""Wake word detection, so open mode doesn't need a key at all.

openWakeWord runs a small ONNX classifier over 80ms frames. It is
cheap enough to leave running indefinitely - a few percent of one core
- which is the whole point: the mic can be armed all day without
Whisper or the model ever waking up.

There is no pretrained "hey Luna", and there won't be unless you train
one. Two options, both handled here:

  * Use a stock phrase. openWakeWord ships hey_jarvis, alexa,
    hey_mycroft and hey_rhasspy, and they work out of the box.

  * Train "hey Luna" yourself. openWakeWord's training notebook takes
    synthetic samples and produces an .onnx you point `model` at. It
    takes about an hour and is the only way to get her own name.

Either way this module degrades quietly: if the package isn't
installed or the model file is missing, open mode falls back to what it
did before - any speech starts a turn.
"""
import os

import numpy as np

from config import (
    SAMPLE_RATE,
    WAKE_WORD_ENABLED,
    WAKE_WORD_MODEL,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_COOLDOWN,
)

# openWakeWord is trained on 80ms frames of 16kHz int16.
FRAME_SIZE = 1280

_model = None
_error = ""
_name = ""

# Highest score seen since the last reset, for /wake.
scores = {"best": 0.0, "last": 0.0, "detections": 0}

# Stock models that ship with the package, by the name you'd pass.
BUILT_IN = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "timer",
            "weather")


def enabled():
    return WAKE_WORD_ENABLED


def available():
    return _model is not None


def label():
    """What the header and /wake should call it."""
    if not WAKE_WORD_ENABLED:
        return "off"

    if _model is None:
        return f"unavailable ({_error})" if _error else "unavailable"

    return _name


def load():
    """Import and load the model. Safe to call when disabled."""
    global _model, _error, _name

    if not WAKE_WORD_ENABLED:
        return False

    try:
        from openwakeword.model import Model
    except ImportError:
        _error = "pip install openwakeword"
        return False

    target = os.path.expanduser(str(WAKE_WORD_MODEL or "").strip())

    if not target:
        _error = "no model set"
        return False

    # A path means a model you trained; a bare name means one of theirs.
    if os.path.sep in target or target.endswith((".onnx", ".tflite")):
        if not os.path.exists(target):
            _error = f"no such file: {target}"
            return False

        models = [target]
        _name = os.path.splitext(os.path.basename(target))[0]
    else:
        if target not in BUILT_IN:
            _error = (
                f"{target!r} isn't one of the built-in models "
                f"({', '.join(BUILT_IN)}) and isn't a path to one you trained"
            )
            return False

        models = [target]
        _name = target

    try:
        # The feature extractors are downloaded once on first use; if
        # they aren't there yet this is what fetches them.
        try:
            import openwakeword.utils

            openwakeword.utils.download_models()
        except Exception:
            pass  # already present, or offline - Model() will say

        _model = Model(wakeword_models=models, inference_framework="onnx")
    except Exception as e:
        _error = str(e)
        _model = None
        return False

    return True


def reset():
    _model.reset() if _model is not None and hasattr(_model, "reset") else None
    scores["best"] = 0.0


def heard(block):
    """Feed one frame. True when the wake word just fired.

    `block` is float32 in [-1, 1], as sounddevice hands it over;
    openWakeWord wants int16.
    """
    if _model is None:
        return False

    frame = np.clip(block, -1.0, 1.0)
    frame = (frame * 32767).astype(np.int16)

    try:
        prediction = _model.predict(frame)
    except Exception:
        return False

    best = max(prediction.values()) if prediction else 0.0

    scores["last"] = best
    scores["best"] = max(scores["best"], best)

    if best < WAKE_WORD_THRESHOLD:
        return False

    scores["detections"] += 1

    # Without clearing the buffers the same utterance keeps scoring
    # above threshold for the next second and fires repeatedly.
    reset()

    return True


def cooldown_seconds():
    return WAKE_WORD_COOLDOWN
