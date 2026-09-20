import io
import queue
import re
import threading
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

import config

from config import (
    SAMPLE_RATE,
    VOICE,
    KOKORO_URL,
    KOKORO_ADDRESS,
    STT_MODEL,
    STT_LANGUAGE,
    STT_MODE,
    STT_VAD,
)

# The rest are deliberately NOT imported by value. /set can change them
# mid-session, and `from config import X` takes a copy that never hears
# about it - so the values you actually tune by ear are read off the
# config module each time they're used.

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
    # Barge-in, reported by /barge
    "barge_baseline": 0.0,
    "barge_peak": 0.0,
    "barge_probability": 0.0,
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
    give_up_after = config.STT_NO_SPEECH_TIMEOUT if mode == "auto" else None
    max_seconds = config.STT_MANUAL_MAX_SECONDS if mode == "manual" else config.STT_MAX_SECONDS

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, dtype="float32"
    )
    stream.start()

    try:
        noise = min(_calibrate(stream), NOISE_CEILING)

        sensitivity = max(0.1, config.STT_SENSITIVITY)
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

                    if silence >= config.STT_SILENCE_SECONDS:
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
        threshold=min(0.9, max(0.1, config.STT_VAD_THRESHOLD)),
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=int(config.STT_SILENCE_SECONDS * 1000),
        speech_pad_ms=200,
    )
    iterator.reset_states()

    give_up_after = config.STT_NO_SPEECH_TIMEOUT if mode == "auto" else None
    block_seconds = VAD_BLOCK_SIZE / SAMPLE_RATE

    levels.update({
        "noise_floor": 0.0,
        "start": config.STT_VAD_THRESHOLD,
        "continue": max(0.0, config.STT_VAD_THRESHOLD - 0.15),
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

            if elapsed >= config.STT_MAX_SECONDS:
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
            json={"text": text, "voice": VOICE, "speed": config.KOKORO_SPEED},
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

    return samples * config.KOKORO_VOLUME, rate


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
    """Speak a complete string. Used where the whole reply is already
    in hand - reminders, startup notices."""
    pieces = _split_chunks(text or "")

    if not pieces:
        return

    state.stop_speaking = False

    inbox = queue.Queue()

    for piece in pieces:
        inbox.put(piece)

    inbox.put(None)

    speak_queue(inbox, reset=False)


def speak_queue(inbox, reset=True):
    """Speak sentences as they arrive on a queue, terminated by None.

    This is the streaming path. The producer is the model itself, still
    generating, so the first sentence usually reaches the speaker while
    the rest of the reply is still being written - which is the whole
    difference between a voice assistant that answers and one that
    pauses first.

    Synthesis runs on the pool while the previous chunk is still
    playing, so the gap between sentences stays inaudible.
    """
    if reset:
        state.stop_speaking = False

    pending = deque()
    done = False

    try:
        while True:
            # Keep one synthesis in flight ahead of playback. More than
            # one gains nothing: the server renders them serially.
            while not done and len(pending) < 2:
                try:
                    item = inbox.get(timeout=0.05)
                except queue.Empty:
                    break

                if item is None:
                    done = True
                    break

                for piece in _split_chunks(item):
                    pending.append(_pool.submit(_synthesize, piece))

            if state.stop_speaking:
                return

            if not pending:
                if done:
                    return

                continue

            future = pending.popleft()

            try:
                samples, rate = future.result()
            except RuntimeError as e:
                # Server down mid-reply. Say so once and stop; the text
                # is already on screen, so nothing is actually lost.
                ui.add_message("system", str(e))
                return
            except Exception:
                continue

            if state.stop_speaking or not _play(samples, rate):
                return
    finally:
        for future in pending:
            future.cancel()


# ---------------------------------------------------------------------------
# Barge-in
#
# The problem with listening while she talks is that the microphone
# hears her too, and Silero is quite right to call that speech. With no
# echo cancellation available the only honest discriminator left is
# loudness: her voice arrives at the mic attenuated by the room, yours
# doesn't.
#
# So the first fraction of a second of playback is used to measure how
# loud she is *at the microphone*, and after that it takes both a
# confident speech classification and a level well above that baseline,
# sustained, to count as you interrupting. On headphones the baseline is
# near silence and this is trivially reliable; on speakers it depends on
# your volume, which is why the margin is configurable.
# ---------------------------------------------------------------------------
BARGE_IN_CALIBRATION_SECONDS = 0.5
BARGE_IN_FLOOR = 0.004      # below this it's room tone, not a person
BARGE_IN_ARM_TIMEOUT = 20.0 # give up waiting for playback to start


def _playing():
    """Is audio actually coming out of the speakers right now?

    The distinction matters more than it sounds. The watcher starts
    when the first sentence is handed to the TTS server, which is a
    second or so before any sound exists - and calibrating against that
    silence sets the baseline to the noise floor, after which her own
    first word clears the bar and she interrupts herself. Every time.
    """
    try:
        stream = sd.get_stream()
    except Exception:
        return False

    return stream is not None and stream.active


def _speech_probability(block):
    """Raw Silero probability for one 512-sample frame."""
    import torch

    with torch.no_grad():
        return float(_vad_model(torch.from_numpy(block), SAMPLE_RATE).item())


def _watch_for_barge_in(stop):
    """Set state.stop_speaking if the user starts talking over Luna.

    Runs for as long as playback does. Bails out quietly on any audio
    error - failing to offer barge-in is a missing nicety, but crashing
    the speaker thread would lose the reply.
    """
    threshold = min(0.95, config.STT_VAD_THRESHOLD + config.STT_BARGE_IN_BOOST)
    block_seconds = VAD_BLOCK_SIZE / SAMPLE_RATE
    needed = max(1, int(config.STT_BARGE_IN_SECONDS / block_seconds))

    bleed = []
    calibrating = max(1, int(BARGE_IN_CALIBRATION_SECONDS / block_seconds))
    baseline = None
    streak = 0

    levels["barge_peak"] = 0.0
    levels["barge_probability"] = 0.0

    # The model carries LSTM state between frames, and it was last used
    # on your voice, not hers.
    try:
        _vad_model.reset_states()
    except Exception:
        pass

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            blocksize=VAD_BLOCK_SIZE, dtype="float32",
        )
        stream.start()
    except Exception:
        return

    waited = 0.0

    try:
        while not stop.is_set() and not state.stop_speaking:
            audio, _overflow = stream.read(VAD_BLOCK_SIZE)
            block = audio[:, 0].copy()
            level = _rms(block)

            # Nothing to talk over yet. Keep draining the mic so the
            # buffer doesn't go stale, but don't measure anything.
            if not _playing():
                if baseline is None:
                    waited += block_seconds

                    if waited > BARGE_IN_ARM_TIMEOUT:
                        return

                continue

            if baseline is None:
                bleed.append(level)

                if len(bleed) >= calibrating:
                    # Upper quartile: her loudest moments are the ones
                    # that would otherwise trigger a false interrupt.
                    bleed.sort()
                    baseline = max(
                        BARGE_IN_FLOOR, bleed[int(len(bleed) * 0.75)]
                    )
                    levels["barge_baseline"] = baseline

                continue

            levels["barge_peak"] = max(levels["barge_peak"], level)

            if level < baseline * config.STT_BARGE_IN_MARGIN:
                streak = 0
                continue

            try:
                probability = _speech_probability(block)
            except Exception:
                return

            levels["barge_probability"] = max(
                levels["barge_probability"], probability
            )

            if probability < threshold:
                streak = 0
                continue

            streak += 1

            if streak >= needed:
                state.stop_speaking = True
                state.barged_in = True
                ui.set_status("Stopped - go ahead")
                return
    except Exception:
        return
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def barge_in_available():
    return config.STT_BARGE_IN and _vad_model is not None


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
def wait_for_wake_word(should_stop):
    """Block until the wake word is heard.

    Returns True if it fired, False if `should_stop()` asked us to give
    up. Nothing else runs while this does - no Whisper, no model, just
    a 1.5MB classifier over 80ms frames - so it can sit here all day.
    """
    import wakeword

    if not wakeword.available():
        return True  # nothing to wait for; behave as before

    ui.set_status(f"Waiting for \"{wakeword.label()}\"")
    wakeword.reset()

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            blocksize=wakeword.FRAME_SIZE, dtype="float32",
        )
        stream.start()
    except Exception as e:
        ui.add_message("system", f"Wake word listener couldn't open the mic: {e}")
        return True

    try:
        while not should_stop():
            audio, _overflow = stream.read(wakeword.FRAME_SIZE)

            if wakeword.heard(audio[:, 0].copy()):
                return True

            if state.stop_listening:
                return False
    except Exception as e:
        ui.add_message("system", f"Wake word listener stopped: {e}")
        return True
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    return False


class Player:
    """A speaker you can hand sentences to while they're still being
    written, and stop mid-word.

    main.py holds one of these for the length of a turn. Playback runs
    on its own thread so the model can keep generating into it.
    """

    def __init__(self, barge_in=True):
        self._inbox = queue.Queue()
        self._thread = None
        self._listener = None
        self._stop_listener = threading.Event()
        self._barge_in = barge_in

    def say(self, sentence):
        if state.stop_speaking:
            return

        if self._thread is None:
            ui.set_status("Speaking...")
            self._thread = threading.Thread(
                target=speak_queue, args=(self._inbox,),
                kwargs={"reset": False}, daemon=True,
            )
            self._thread.start()

            if self._barge_in and barge_in_available():
                self._listener = threading.Thread(
                    target=_watch_for_barge_in, args=(self._stop_listener,),
                    daemon=True,
                )
                self._listener.start()

        self._inbox.put(sentence)

    def wait(self):
        """Block until everything queued has been spoken."""
        if self._thread is None:
            return

        self._inbox.put(None)
        self._thread.join()
        self._thread = None

        self._stop_listener.set()

        if self._listener is not None:
            self._listener.join(timeout=1.0)
            self._listener = None

    @property
    def started(self):
        return self._thread is not None
