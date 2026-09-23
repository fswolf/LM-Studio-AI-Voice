"""Waking someone up, which is not the same job as reminding them.

A reminder is polite. It says its piece once, at whatever volume you
left the assistant on, and if you were out of the room it is gone. That
is correct for "take the bins out" and useless at seven in the morning.

So an alarm is a reminder with a different delivery:

  * It repeats until dismissed rather than firing once.
  * It has its own volume, because the whole point is to be louder than
    the setting you chose for a conversation at midnight.
  * It plays an actual tone after the voice. Speech alone doesn't wake
    anybody - it is the thing your brain has spent years learning to
    incorporate into a dream.
  * It escalates. The first line is hers, generated for the occasion;
    by the fourth she has stopped being charming about it.
  * It snoozes.

Scheduling, persistence and repeats are all reminders.py's - an alarm
is stored as a reminder with kind="alarm", so "every weekday at 7:30"
already works and there is one scanner rather than two.
"""
import os
import threading
import time

import numpy as np

import config
import logbook
import state

from config import AGENT_NAME, BASE_DIR

TONE_RATE = 24000

# Nagging lines for when the first one didn't work. Deliberately not
# model-generated: by the third repeat you want it fast and identical,
# not a fresh round trip to LM Studio while the alarm sits silent.
NAGS = (
    "Still in bed.",
    "I can keep doing this.",
    "Seriously. Up.",
    "This is your last warning before I start singing.",
)

_ringing = threading.Event()
_dismissed = threading.Event()
_snoozed_for = [0]


# ---------------------------------------------------------------------------
# The sound
# ---------------------------------------------------------------------------
def _synthesize():
    """The built-in tone, generated rather than loaded.

    Exists so a missing or unreadable assets/alarm.wav degrades to a
    working alarm instead of a silent one. An alarm that fails quietly
    is worse than no alarm, because you were relying on it.
    """
    def beep(freq, seconds):
        t = np.arange(int(TONE_RATE * seconds)) / TONE_RATE
        tone = (np.sin(2 * np.pi * freq * t)
                + 0.35 * np.sin(4 * np.pi * freq * t)
                + 0.18 * np.sin(6 * np.pi * freq * t))
        envelope = np.ones_like(t)
        edge = int(TONE_RATE * 0.008)
        envelope[:edge] = np.linspace(0, 1, edge)
        envelope[-edge:] = np.linspace(1, 0, edge)

        return tone * envelope

    parts = []

    for index in range(4):
        parts.append(beep(880 if index % 2 == 0 else 1175, 0.12))
        parts.append(np.zeros(int(TONE_RATE * 0.09)))

    parts.append(np.zeros(int(TONE_RATE * 0.55)))
    audio = np.concatenate(parts).astype(np.float32)

    return audio / max(float(np.abs(audio).max()), 1e-6) * 0.85, TONE_RATE


_cached_tone = None
_cached_for = None


def tone():
    """(samples, rate) for the alarm sound, loaded once per path.

    Keyed on the path rather than just "have we loaded anything", so
    `/set alarms.tone something.wav` takes effect on the next ring
    instead of at the next restart - which is exactly when you would
    have stopped trusting the setting.
    """
    global _cached_tone, _cached_for

    path = config.ALARM_TONE or os.path.join(BASE_DIR, "assets", "alarm.wav")

    if _cached_tone is not None and _cached_for == path:
        return _cached_tone

    _cached_for = path

    try:
        import wave

        with wave.open(path) as handle:
            if handle.getsampwidth() != 2:
                raise ValueError(f"{path} isn't 16-bit")

            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        _cached_tone = (samples, rate)
        logbook.info("alarm", "tone: %s (%.1fs)", path, len(samples) / rate)
    except Exception as e:
        logbook.info("alarm", "using the built-in tone (%s: %s)", path, e)
        _cached_tone = _synthesize()

    return _cached_tone


def play_tone():
    """Play it once, at the alarm's own volume. Interruptible."""
    import sounddevice as sd

    samples, rate = tone()

    try:
        sd.play(np.clip(samples * config.ALARM_VOLUME, -1.0, 1.0), rate)

        while True:
            stream = sd.get_stream()

            if stream is None or not stream.active:
                return

            if _dismissed.is_set():
                sd.stop()

                return

            sd.sleep(50)
    except Exception as e:
        logbook.warn("alarm", "couldn't play the tone: %s", e)


# ---------------------------------------------------------------------------
# Ringing
# ---------------------------------------------------------------------------
def ringing():
    return _ringing.is_set()


def dismiss(snooze_minutes=0):
    """Stop the current alarm. Returns True if one was actually ringing."""
    if not _ringing.is_set():
        return False

    _snoozed_for[0] = max(0, int(snooze_minutes))
    _dismissed.set()

    return True


def _wake_line(model, text):
    """Her own words for this specific alarm, once.

    Falls back to something canned rather than letting a dead LM Studio
    turn the alarm into silence - the tone still has to fire, and it is
    the part that does the work anyway.
    """
    try:
        import llm

        return llm.ask(
            f'(It is time to wake up - this alarm was set for "{text}". '
            "Say one short, daft, affectionate line to wake them up. "
            "One sentence. No preamble.)",
            model,
        )
    except Exception as e:
        logbook.warn("alarm", "couldn't get a wake-up line: %s", e)

        return f"Wake up! {text}."


def ring(model, alarm):
    """The whole wake-up: her line, the tone, and again until told to stop.

    Returns the snooze in minutes, or 0 if it was dismissed properly.
    """
    import ui
    from speech import speak

    text = alarm.get("text") or "wake up"

    _dismissed.clear()
    _snoozed_for[0] = 0
    _ringing.set()

    ui.set_status("ALARM")
    ui.add_message(
        "system",
        f"Alarm: {text} - /snooze, or /alarm off to stop it.",
    )

    try:
        for round_number in range(max(1, config.ALARM_REPEATS)):
            if _dismissed.is_set():
                break

            if round_number == 0:
                line = _wake_line(model, text)
            else:
                line = NAGS[min(round_number - 1, len(NAGS) - 1)]

            ui.add_message(AGENT_NAME.lower(), line)

            # stop_speaking is barge-in: talking over an alarm is the
            # most natural way there is to tell it to shut up.
            if not _dismissed.is_set() and not state.stop_speaking:
                try:
                    speak(line)
                except Exception as e:
                    logbook.warn("alarm", "couldn't speak: %s", e)

            if state.stop_speaking:
                _dismissed.set()
                break

            play_tone()

            if _dismissed.is_set():
                break

            # Between rounds. Checked often so a dismissal lands
            # immediately rather than after another 25 seconds.
            waited = 0.0

            while waited < config.ALARM_GAP_SECONDS:
                if _dismissed.is_set() or state.stop_speaking:
                    break

                time.sleep(0.25)
                waited += 0.25
    finally:
        _ringing.clear()
        ui.set_status("Idle")

    snooze = _snoozed_for[0]
    _snoozed_for[0] = 0

    if snooze:
        ui.add_message("system", f"Snoozed for {snooze} minutes.")

    return snooze
