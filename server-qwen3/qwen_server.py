"""Qwen3-TTS, behind the same three endpoints kokoro-reader answers.

    POST /tts   {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes
    GET  /health                                           ->  {"ok": true}
    GET  /voices                                           ->  the voice list

## What this actually is

qwentts.cpp ships its own HTTP server - OpenAI-shaped, `/v1/audio/speech`
- and it is good. This is not a reimplementation of it. It starts that
server as a child process, keeps it alive, and translates between its
API and the one the assistant speaks.

That is the whole design, and it is deliberate. The alternative was
shelling out to the `qwen-tts` binary per sentence, which reloads a
gigabyte of weights every time somebody says hello. Running their
server means the model is loaded once and every sentence is just an
HTTP call - which is also how it manages to answer in under a second.

## Three ways to have a voice

Qwen3-TTS is really three models, and which one is loaded decides what
"voice" means. Set `mode` and the right talker is used:

  customvoice   named speakers - serena, vivian, ono_anna, sohee...
                Nothing to set up. This is the default.
  base          zero-shot cloning. Any .wav in voices/ is registered
                with the engine at startup and selectable by filename.
  voicedesign   the voice is a sentence: "female, young adult, bright,
                slightly breathy". 1.7B only.

They need different weights, so one server is one mode. Switching is a
config change and a restart, not a per-request flag.
"""
import base64
import io
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_PORT", "8899"))

# The child. Kept on a different port, and not one anything else here
# defaults to, so a stray tts-server can't be mistaken for ours.
ENGINE_PORT = int(os.environ.get("QWEN_ENGINE_PORT", "8971"))
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"

MODE = os.environ.get("QWEN_MODE", "customvoice").strip().lower()
SIZE = os.environ.get("QWEN_SIZE", "0.6b").strip().lower()
QUANT = os.environ.get("QWEN_QUANT", "Q8_0").strip()
LANGUAGE = os.environ.get("QWEN_LANGUAGE", "English")

REPO = os.environ.get("QWEN_REPO", os.path.join(HERE, "qwentts.cpp"))
MODELS = os.environ.get("QWEN_MODELS", os.path.join(REPO, "models"))
VOICE_DIR = os.path.join(HERE, "voices")

# The named speakers baked into the customvoice checkpoints.
SPEAKERS = ("serena", "vivian", "ono_anna", "sohee", "uncle_fu",
            "ryan", "aiden", "eric", "dylan")

# Passed straight through to tts-server. These exist because the
# CodePredictor stage can dominate the total - it is a 5-layer model run
# once per frame, so at 12 Hz it is hundreds of tiny GPU submissions a
# second, and on some drivers the per-dispatch overhead costs more than
# the arithmetic. None of that shows up as high GPU load; it shows up as
# a frame time thirty times the talker's.
#
#   QWEN_NO_FA=1          disable flash attention (often the culprit)
#   QWEN_MAX_BATCH=8      more work per submission
#   QWEN_CHUNK_DUR=2.0    codec decode chunk, seconds
#   QWEN_EXTRA="--foo 1"  anything else
NO_FLASH_ATTENTION = os.environ.get("QWEN_NO_FA", "").strip() not in ("", "0")
MAX_BATCH = os.environ.get("QWEN_MAX_BATCH", "").strip()
CHUNK_DUR = os.environ.get("QWEN_CHUNK_DUR", "").strip()
EXTRA = os.environ.get("QWEN_EXTRA", "").strip()

MAX_CHARS = int(os.environ.get("QWEN_MAX_CHARS", "1200"))
START_TIMEOUT = float(os.environ.get("QWEN_START_TIMEOUT", "180"))

_engine = None
_ready = threading.Event()
_problem = ""
_registered = []
_generating = threading.Lock()
_stats = {"requests": 0, "seconds": 0.0, "audio_seconds": 0.0}


def log(message):
    print(f"[qwen3] {message}", flush=True)


# ---------------------------------------------------------------------------
# Finding the pieces
# ---------------------------------------------------------------------------
def talker_path():
    # voicedesign was only ever trained at 1.7B; asking for 0.6B would
    # look up a file that does not exist, so correct it rather than
    # failing with a confusing "not found".
    size = "1.7b" if MODE == "voicedesign" else SIZE

    return os.path.join(MODELS, f"qwen-talker-{size}-{MODE}-{QUANT}.gguf")


def codec_path():
    return os.path.join(MODELS, f"qwen-tokenizer-12hz-{QUANT}.gguf")


def binary(name):
    for candidate in (os.path.join(REPO, "build", name),
                      os.path.join(REPO, "build", "bin", name)):
        if os.path.exists(candidate):
            return candidate

    return shutil.which(name)


def why_not_ready():
    """Everything that has to be true before this can work, in the order
    somebody would fix them."""
    if not os.path.isdir(REPO):
        return f"qwentts.cpp isn't at {REPO} - run ./setup.sh"

    if not binary("tts-server"):
        return f"tts-server isn't built in {REPO}/build - run ./setup.sh"

    for path in (talker_path(), codec_path()):
        if not os.path.exists(path):
            return f"missing {os.path.basename(path)} - run ./setup.sh"

    return ""


# ---------------------------------------------------------------------------
# The child process
# ---------------------------------------------------------------------------
def engine_alive():
    try:
        with urllib.request.urlopen(f"{ENGINE_URL}/v1/models", timeout=3):
            return True
    except urllib.error.HTTPError:
        # Answering at all is enough - some builds 404 that path.
        return True
    except Exception:
        return False


def start_engine():
    global _engine, _problem

    _problem = why_not_ready()

    if _problem:
        log(_problem)

        return

    command = [
        binary("tts-server"),
        "--model", talker_path(),
        "--codec", codec_path(),
        "--port", str(ENGINE_PORT),
        "--alias", f"qwen3-tts-{MODE}",
    ]

    if NO_FLASH_ATTENTION:
        command.append("--no-fa")

    if MAX_BATCH:
        command += ["--max-batch", MAX_BATCH]

    if CHUNK_DUR:
        command += ["--codec-chunk-dur", CHUNK_DUR]

    if EXTRA:
        command += EXTRA.split()

    log(f"starting engine: {MODE} {SIZE} {QUANT}")
    log(f"  {' '.join(command[1:])}")

    try:
        _engine = subprocess.Popen(
            command, cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as e:
        _problem = f"couldn't start tts-server: {e}"
        log(_problem)

        return

    # Its output is the only place a Vulkan or model failure shows up,
    # so it is relayed rather than swallowed - prefixed, so it's obvious
    # which process is talking.
    threading.Thread(target=_relay, daemon=True).start()

    started = time.time()

    while time.time() - started < START_TIMEOUT:
        if _engine.poll() is not None:
            _problem = f"tts-server exited ({_engine.returncode}) - see the log above"
            log(_problem)

            return

        if engine_alive():
            log(f"engine up in {time.time() - started:.1f}s")
            register_voices()
            warm_up()
            _ready.set()

            return

        time.sleep(0.5)

    _problem = f"tts-server didn't answer within {START_TIMEOUT:.0f}s"
    log(_problem)


def _relay():
    for line in _engine.stdout:
        log(f"  engine| {line.rstrip()}")


def warm_up():
    """One short synthesis, so the first thing she says isn't the one
    that pays for lazy allocation."""
    try:
        speak("Ready.", default_voice(), 1.0)
        log("warmed up")
    except Exception as e:
        log(f"warmup failed (continuing): {e}")


def stop_engine():
    if _engine and _engine.poll() is None:
        _engine.send_signal(signal.SIGTERM)

        try:
            _engine.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _engine.kill()


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
def default_voice():
    if MODE == "customvoice":
        return os.environ.get("QWEN_VOICE", "serena")

    if MODE == "voicedesign":
        return os.environ.get(
            "QWEN_VOICE", "female, young adult, moderate pitch, warm"
        )

    return _registered[0] if _registered else ""


def reference_clips():
    try:
        entries = sorted(os.listdir(VOICE_DIR))
    except OSError:
        return []

    return [
        (os.path.splitext(e)[0], os.path.join(VOICE_DIR, e))
        for e in entries if e.lower().endswith(".wav")
    ]


def register_voices():
    """Hand every clip in voices/ to the engine, once, at startup.

    Only `base` clones. The other two modes have their own idea of what
    a voice is, and registering against them would either be refused or
    silently ignored - neither of which is worth the confusion.
    """
    _registered.clear()

    if MODE != "base":
        return

    for name, path in reference_clips():
        transcript = ""
        text_file = os.path.splitext(path)[0] + ".txt"

        if os.path.exists(text_file):
            try:
                with open(text_file) as handle:
                    transcript = handle.read().strip()
            except OSError:
                pass

        try:
            with open(path, "rb") as handle:
                payload = {
                    "name": name,
                    "wav_b64": base64.b64encode(handle.read()).decode(),
                }

            # A matching transcript turns x-vector cloning into
            # in-context cloning, which is noticeably better. Optional,
            # because writing one out for every clip is a chore.
            if transcript:
                payload["ref_text"] = transcript

            post(f"{ENGINE_URL}/v1/audio/voices", payload, timeout=120)
            _registered.append(name)
            log(f"registered voice: {name}"
                f"{' (with transcript)' if transcript else ''}")
        except Exception as e:
            log(f"couldn't register {name}: {e}")


def voices():
    if MODE == "customvoice":
        return list(SPEAKERS)

    if MODE == "voicedesign":
        return ["<any description: 'female, young adult, bright'>"]

    return list(_registered)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def post(url, payload, timeout=300, raw=False):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()

    return body if raw else (json.loads(body) if body else {})


def speak(text, voice, speed):
    payload = {
        "input": text,
        "response_format": "wav",
        "language": LANGUAGE,
    }

    # In voicedesign mode the "voice" is a sentence describing one, and
    # the field it goes in is different. Everywhere else it is a name.
    if MODE == "voicedesign":
        payload["instruct"] = voice
    elif voice:
        payload["voice"] = voice

    with _generating:
        audio = post(f"{ENGINE_URL}/v1/audio/speech", payload, raw=True)

    return stretch(audio, speed)


def stretch(wav_bytes, speed):
    """Change the tempo without moving the pitch.

    The engine has no speed control and the contract has one. Done by
    dropping or repeating whole frames rather than resampling, so the
    pitch stays put - a resample would make "slightly faster" sound
    like a different, smaller person.
    """
    if abs(float(speed or 1.0) - 1.0) < 0.02:
        return wav_bytes

    try:
        import array

        with wave.open(io.BytesIO(wav_bytes)) as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if width != 2:
            return wav_bytes

        samples = array.array("h")
        samples.frombytes(frames)

        # Overlap-add would be better; this is a plain frame-drop with a
        # short crossfade, which at the 0.8-1.25 range anyone actually
        # uses is inaudible and costs nothing.
        step = float(speed)
        out = array.array("h")
        position = 0.0
        total = len(samples) // channels

        while position < total - 1:
            index = int(position) * channels

            for channel in range(channels):
                out.append(samples[index + channel])

            position += step

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(out.tobytes())

        return buffer.getvalue()
    except Exception as e:
        log(f"speed change failed, sending it unmodified: {e}")

        return wav_bytes


def audio_seconds(wav_bytes):
    try:
        with wave.open(io.BytesIO(wav_bytes)) as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        return 0.0


def synthesize(text, voice, speed):
    started = time.time()
    chosen = voice or default_voice()

    # A name that means nothing in this mode would come back as an
    # engine error mid-sentence. Fall back instead: a wrong voice costs
    # you the character, a failed request costs you the reply.
    if MODE == "customvoice" and chosen not in SPEAKERS:
        chosen = default_voice()
    elif MODE == "base" and _registered and chosen not in _registered:
        chosen = _registered[0]

    audio = speak(text, chosen, speed)

    spent = time.time() - started
    seconds = audio_seconds(audio)

    _stats["requests"] += 1
    _stats["seconds"] += spent
    _stats["audio_seconds"] += seconds

    log(f'"{text[:44]}" -> {seconds:.1f}s in {spent:.1f}s '
        f"(x{seconds / max(spent, 0.001):.1f} realtime, voice={chosen[:32]})")

    return audio


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()

        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/health":
            ready = _ready.is_set()
            body = {"ok": ready, "engine": "qwen3", "mode": MODE,
                    "size": "1.7b" if MODE == "voicedesign" else SIZE}

            if not ready:
                body["loading"] = not _problem
                body["error"] = _problem

            return self._send(200, body)

        if path == "/voices":
            return self._send(200, {
                "voices": voices(), "default": default_voice(), "mode": MODE,
            })

        if path in ("/", "/stats"):
            done = _stats["requests"] or 1

            return self._send(200, {
                "engine": "qwen3", "mode": MODE, "ready": _ready.is_set(),
                "requests": _stats["requests"],
                "realtime_factor": round(
                    _stats["audio_seconds"] / max(_stats["seconds"], 0.001), 2
                ),
                "average_seconds": round(_stats["seconds"] / done, 2),
                "voices": voices(),
            })

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/tts":
            return self._send(404, {"error": "not found"})

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError) as e:
            return self._send(400, {"error": f"bad request: {e}"})

        if not isinstance(payload, dict):
            return self._send(400, {"error": "expected a JSON object"})

        text = str(payload.get("text", "")).strip()

        if not text:
            return self._send(400, {"error": "no text"})

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit(" ", 1)[0]

        if not _ready.is_set():
            return self._send(503, {"error": _problem or "still starting up"})

        try:
            audio = synthesize(
                text, payload.get("voice"), payload.get("speed", 1.0)
            )
        except Exception as e:
            log(f"generation failed: {type(e).__name__}: {e}")

            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        return self._send(200, audio, "audio/wav")


def main():
    os.makedirs(VOICE_DIR, exist_ok=True)

    threading.Thread(target=start_engine, daemon=True).start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on http://{HOST}:{PORT}  (mode={MODE})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        stop_engine()


if __name__ == "__main__":
    main()
