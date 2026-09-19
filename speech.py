import io
import re
import wave

import requests
import sounddevice as sd
import numpy as np
import state
import ui

from concurrent.futures import ThreadPoolExecutor
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from collections import deque

from config import (
    SAMPLE_RATE,
    VOICE,
    KOKORO_URL,
    KOKORO_SPEED,
    KOKORO_VOLUME,
    KOKORO_ADDRESS,
    STT_MODEL,
    STT_LANGUAGE,
    STT_SENSITIVITY,
    STT_SILENCE_SECONDS,
    STT_MAX_SECONDS,
    STT_NO_SPEECH_TIMEOUT,
    STT_MANUAL_MAX_SECONDS,
    STT_MODE,
    STT_VAD,
    STT_VAD_THRESHOLD,
)

whisper = None

# Set by load_models(); main.py surfaces it in the UI so a dead TTS
# server is obvious at startup instead of on the first reply.
kokoro_ok = False
kokoro_error = ""


def server_label():
    """The VServer line: address, plus whether it answered."""
    return KOKORO_ADDRESS if kokoro_ok else f"{KOKORO_ADDRESS} [offline]"


def _set_reachable(ok, error=""):
    """Track reachability so the header reflects reality mid-session -
    if the server dies (or comes back) you see it on the next reply."""
    global kokoro_ok, kokoro_error

    changed = ok != kokoro_ok
    kokoro_ok = ok
    kokoro_error = error

    if changed:
        try:
            ui.set_voice_server(server_label())
        except Exception:
            pass

# The server truncates at 1200 chars, so split below that and stitch
# the pieces back together on playback.
MAX_TTS_CHARS = 1000

# One worker: synthesize the next chunk while the current one plays.
_pool = ThreadPoolExecutor(max_workers=1)


def load_models():
    global whisper

    whisper = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")

    _load_vad()

    try:
        response = requests.get(f"{KOKORO_URL}/health", timeout=5)
        response.raise_for_status()
        _set_reachable(bool(response.json().get("ok")))
    except Exception as e:
        _set_reachable(False, str(e))

    return kokoro_ok


# ---------------------------------------------------------------------------
# Recording / voice activity detection
#
# The old version compared RMS against a hardcoded 0.035 and nothing else.
# That's roughly ten times the level a normal desktop mic produces, so on
# most inputs `started` never flipped true, the silence timer never began,
# and every recording ran to the 60-second cap instead of stopping when
# you did. Thresholds are now derived from the actual noise floor, with
# separate levels to start and to keep going.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 1024                 # 64ms at 16kHz, energy path
VAD_BLOCK_SIZE = 512              # 32ms - what the Silero model expects
CALIBRATION_SECONDS = 0.4         # listen to the room before arming
SMOOTHING = 3                     # blocks averaged, so one quiet frame
                                  # doesn't look like you stopped talking
PRE_ROLL_SECONDS = 0.8            # audio kept from before speech was detected
MIN_SPEECH_SECONDS = 0.25         # shorter than this is a cough, not a sentence
ABSOLUTE_FLOOR = 0.0022           # don't trust a silent-room calibration
NOISE_CEILING = 0.02              # don't trust a loud one either

# Energy-path multipliers. Lowered from 3.0/1.5: with only loudness to go
# on, a high bar is late to trigger and a high floor treats quiet
# syllables as silence. Silero replaces the guesswork entirely.
START_MULTIPLIER = 2.2
CONTINUE_MULTIPLIER = 1.15

# Set by load_models() when silero-vad imports.
_vad_model = None
vad_backend = "energy"


def _load_vad():
    """Load Silero if we're allowed to and it's installed.

    It is a genuine speech classifier rather than a loudness meter, which
    is the whole difference: it can trigger on the first syllable without
    also triggering on a fan, and it won't end a turn because you went
    quiet for a moment.
    """
    global _vad_model, vad_backend

    if STT_VAD == "energy":
        vad_backend = "energy"
        return

    try:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
        vad_backend = "silero"
    except Exception as e:
        _vad_model = None
        vad_backend = "energy"

        if STT_VAD == "silero":
            # Explicitly asked for it, so say why it isn't happening.
            try:
                ui.add_message(
                    "system",
                    f"silero-vad unavailable ({e}) - using the energy detector. "
                    "pip install silero-vad",
                )
            except Exception:
                pass

# Last measurement, surfaced by /mic so a bad mic is diagnosable.
levels = {
    "noise_floor": 0.0,
    "start": 0.0,
    "continue": 0.0,
    "peak": 0.0,
    "speech_seconds": 0.0,
    "triggered": False,
    "mode": STT_MODE,
    "backend": "energy",
}


def _rms(block):
    return float(np.sqrt(np.mean(block ** 2)))


def _calibrate(stream):
    """Measure the room for a moment to place the thresholds.

    Uses the lower quartile rather than the mean, so starting to talk
    immediately after pressing HOME doesn't drag the noise floor up and
    deafen the detector.
    """
    readings = []

    for _ in range(max(1, int(CALIBRATION_SECONDS * SAMPLE_RATE / BLOCK_SIZE))):
        audio, _overflow = stream.read(BLOCK_SIZE)
        readings.append(_rms(audio[:, 0]))

    readings.sort()

    return readings[len(readings) // 4] if readings else 0.0


def _record_energy(mode):
    """RMS fallback. Only loudness to work with, so it is late to start
    and prone to cutting off - see _record_silero for the good path."""
    record_immediately = mode == "manual"
    stop_on_silence = mode != "manual"
    give_up_after = STT_NO_SPEECH_TIMEOUT if mode == "auto" else None
    max_seconds = STT_MANUAL_MAX_SECONDS if mode == "manual" else STT_MAX_SECONDS

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, dtype="float32"
    )
    stream.start()

    try:
        noise = min(_calibrate(stream), NOISE_CEILING)

        sensitivity = max(0.1, STT_SENSITIVITY)
        start_threshold = max(ABSOLUTE_FLOOR, noise * START_MULTIPLIER) / sensitivity
        continue_threshold = max(
            ABSOLUTE_FLOOR / 2, noise * CONTINUE_MULTIPLIER
        ) / sensitivity

        levels.update({
            "noise_floor": noise,
            "start": start_threshold,
            "continue": continue_threshold,
            "peak": 0.0,
            "speech_seconds": 0.0,
            "triggered": record_immediately,
            "mode": mode,
            "backend": "energy",
        })

        block_seconds = BLOCK_SIZE / SAMPLE_RATE
        pre_roll = deque(maxlen=max(1, int(PRE_ROLL_SECONDS / block_seconds)))
        recent = deque(maxlen=SMOOTHING)

        chunks = []
        started = record_immediately
        silence = 0.0
        elapsed = 0.0

        if record_immediately:
            ui.set_status("Recording...")

        while True:
            audio, _overflow = stream.read(BLOCK_SIZE)
            block = audio[:, 0].copy()

            level = _rms(block)
            recent.append(level)
            smoothed = sum(recent) / len(recent)

            elapsed += block_seconds
            levels["peak"] = max(levels["peak"], level)

            if not started:
                pre_roll.append(block)

                # Raw level to trigger: waiting for the smoothed average
                # costs you the first syllable.
                if level > start_threshold:
                    started = True
                    levels["triggered"] = True
                    chunks.extend(pre_roll)
                    chunks.append(block)
                    ui.set_status("Recording...")
                elif give_up_after is not None and elapsed >= give_up_after:
                    return None  # you never said anything
            else:
                chunks.append(block)

                if stop_on_silence:
                    # Smoothed level to *stop*: a pause between words
                    # shouldn't end the recording.
                    if smoothed > continue_threshold:
                        silence = 0.0
                    else:
                        silence += block_seconds

                    if silence >= STT_SILENCE_SECONDS:
                        break

            if state.stop_listening:
                break

            if elapsed >= max_seconds:
                break
    finally:
        stream.stop()
        stream.close()

    if not started or not chunks:
        return None

    recorded = np.concatenate(chunks)

    # Drop the trailing silence that ended the recording - Whisper
    # hallucinates filler over long silent tails.
    keep = len(recorded) - int(max(0.0, silence - 0.3) * SAMPLE_RATE)
    recorded = recorded[:max(0, keep)]

    speech_seconds = len(recorded) / SAMPLE_RATE
    levels["speech_seconds"] = speech_seconds

    if speech_seconds < MIN_SPEECH_SECONDS:
        return None

    filename = "/tmp/input.wav"
    write(filename, SAMPLE_RATE, recorded)

    return filename


def _record_silero(mode):
    """Silero VAD path.

    VADIterator hands back {'start': ...} the first frame that crosses
    the threshold and {'end': ...} only after min_silence of genuine
    non-speech - and it ends at (threshold - 0.15), not the threshold, so
    trailing off quietly doesn't terminate the turn. That built-in
    hysteresis is what the energy detector could never do.
    """
    from silero_vad import VADIterator

    iterator = VADIterator(
        _vad_model,
        threshold=min(0.9, max(0.1, STT_VAD_THRESHOLD)),
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=int(STT_SILENCE_SECONDS * 1000),
        speech_pad_ms=200,
    )
    iterator.reset_states()

    give_up_after = STT_NO_SPEECH_TIMEOUT if mode == "auto" else None
    block_seconds = VAD_BLOCK_SIZE / SAMPLE_RATE

    levels.update({
        "noise_floor": 0.0,
        "start": STT_VAD_THRESHOLD,
        "continue": max(0.0, STT_VAD_THRESHOLD - 0.15),
        "peak": 0.0,
        "speech_seconds": 0.0,
        "triggered": False,
        "mode": mode,
        "backend": "silero",
    })

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=VAD_BLOCK_SIZE,
        dtype="float32",
    )
    stream.start()

    pre_roll = deque(maxlen=max(1, int(PRE_ROLL_SECONDS / block_seconds)))
    chunks = []
    started = False
    elapsed = 0.0

    try:
        while True:
            audio, _overflow = stream.read(VAD_BLOCK_SIZE)
            block = audio[:, 0].copy()

            elapsed += block_seconds
            levels["peak"] = max(levels["peak"], _rms(block))

            try:
                event = iterator(block)
            except Exception:
                # Model unhappy with a frame - don't let it kill the turn.
                event = None

            if not started:
                pre_roll.append(block)

                if event and "start" in event:
                    started = True
                    levels["triggered"] = True
                    chunks.extend(pre_roll)
                    ui.set_status("Recording...")
                elif give_up_after is not None and elapsed >= give_up_after:
                    return None
            else:
                chunks.append(block)

                if event and "end" in event:
                    break

            if state.stop_listening:
                break

            if elapsed >= STT_MAX_SECONDS:
                break
    finally:
        stream.stop()
        stream.close()

    if not started or not chunks:
        return None

    recorded = np.concatenate(chunks)
    speech_seconds = len(recorded) / SAMPLE_RATE
    levels["speech_seconds"] = speech_seconds

    if speech_seconds < MIN_SPEECH_SECONDS:
        return None

    filename = "/tmp/input.wav"
    write(filename, SAMPLE_RATE, recorded)

    return filename


def record_audio(mode=None):
    """Record a turn. Returns a wav path, or None if there was nothing
    worth transcribing.

    Three shapes, because one size genuinely doesn't fit:

      auto   - wait for speech, stop on silence. Good for a quick
               question.
      manual - record from the moment you press, stop when you press
               again. Silence ignored entirely, so you can gather your
               thoughts mid-sentence.
      open   - wait for speech however long it takes, stop on silence.
               Same as auto minus the give-up timeout, because hands
               free there's nothing to give up on.
    """
    mode = mode or STT_MODE

    # Manual has nothing to detect - you decide both ends.
    if mode != "manual" and _vad_model is not None:
        try:
            return _record_silero(mode)
        except Exception as e:
            ui.add_message(
                "system", f"Silero VAD failed ({e}) - falling back to levels."
            )

    return _record_energy(mode)


# Whisper emits these over near-silence. Harmless when you press a key
# to talk; in open mode the mic is always armed, so without this Luna
# would answer a room tone every few seconds.
_HALLUCINATIONS = {
    "you", "thank you.", "thanks for watching!", "thank you for watching!",
    "bye.", "bye bye.", ".", "..", "...", "okay.", "oh.", "uh", "um",
    "please subscribe", "subtitles by the amara.org community",
}


def transcribe(filename):
    if not filename:
        return ""

    segments, _info = whisper.transcribe(
        filename,
        language=STT_LANGUAGE,
        vad_filter=True,
        # Without this, Whisper carries its previous output forward as
        # context and can loop the same phrase on a noisy clip.
        condition_on_previous_text=False,
    )

    text = "".join(segment.text for segment in segments).strip()

    if text.lower().strip(" .!?,") in {
        h.strip(" .!?,") for h in _HALLUCINATIONS
    }:
        return ""

    return text


# ---------------------------------------------------------------------------
# Text-to-speech via the kokoro-reader HTTP server
# ---------------------------------------------------------------------------
def _split_chunks(text):
    """Break text into <= MAX_TTS_CHARS pieces on sentence boundaries."""
    pieces = []

    for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
        sentence = sentence.strip()

        if not sentence:
            continue

        # A single monster sentence still has to be cut somewhere.
        while len(sentence) > MAX_TTS_CHARS:
            cut = sentence.rfind(" ", 0, MAX_TTS_CHARS)

            if cut <= 0:  # rfind returns -1 when there's no space to break on
                cut = MAX_TTS_CHARS

            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if not pieces or len(pieces[-1]) + len(sentence) + 1 > MAX_TTS_CHARS:
            pieces.append(sentence)
        else:
            pieces[-1] = f"{pieces[-1]} {sentence}"

    return [p for p in pieces if p]


def _synthesize(text):
    """POST one chunk to /tts and decode the WAV it returns."""
    try:
        response = requests.post(
            f"{KOKORO_URL}/tts",
            json={"text": text, "voice": VOICE, "speed": KOKORO_SPEED},
            timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        _set_reachable(False, str(e))
        raise RuntimeError(
            f"Kokoro server unreachable at {KOKORO_URL} "
            f"(start kokoro_server.py from kokoro-reader): {e}"
        ) from e

    _set_reachable(True)

    with wave.open(io.BytesIO(response.content), "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    return samples * KOKORO_VOLUME, rate


def _play(samples, rate):
    """Play one chunk; return False if HOME interrupted it."""
    sd.play(samples, rate)

    while True:
        stream = sd.get_stream()

        if stream is None or not stream.active:
            return True

        if state.stop_speaking:
            sd.stop()
            return False

        sd.sleep(50)


def speak(text):
    state.stop_speaking = False

    chunks = _split_chunks(text or "")

    if not chunks:
        return

    pending = _pool.submit(_synthesize, chunks[0])

    for index in range(len(chunks)):
        samples, rate = pending.result()

        # Queue the next chunk's synthesis now so it renders while this
        # one is still playing - long replies stop stuttering between
        # sentences. Don't bother if we've already been interrupted.
        has_next = index + 1 < len(chunks) and not state.stop_speaking
        pending = _pool.submit(_synthesize, chunks[index + 1]) if has_next else None

        if state.stop_speaking or not _play(samples, rate):
            break

        if pending is None:
            break
