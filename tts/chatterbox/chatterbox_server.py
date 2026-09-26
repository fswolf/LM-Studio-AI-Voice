"""Chatterbox TTS, behind the same three endpoints kokoro-reader answers.

    POST /tts   {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes
    GET  /health                                           ->  {"ok": true}
    GET  /voices                                           ->  the voice list

That contract is the whole point. The assistant is a client of whatever
is on the other end of port 8899 and knows nothing else about it, so
swapping engines is a port number - not a change to the assistant.

## Why this one

Chatterbox is 0.5B and wants 4-6GB, which matters when the same card is
already holding a 9B language model. Orpheus sounds better and wants
8-12GB, which on a 16GB card means choosing between her voice and her
brain.

What it buys over Kokoro is the part you can't get by picking a nicer
preset: a voice cloned from a few seconds of reference audio, and an
`exaggeration` dial that actually changes how something is delivered
rather than how fast it is read.

## Voices are files

Kokoro has named voices baked in. Chatterbox has one built-in voice and
clones anything else from a reference clip, so "the voice list" here is
whatever .wav files are sitting in voices/. Drop `luna.wav` in and ask
for `"voice": "luna"`. Ask for a name that isn't there and you get the
built-in voice rather than an error, because a missing voice file
should cost you the character, not the sentence.

## One GPU, one generation

Requests are serialized behind a lock. The assistant streams sentence
by sentence and will happily have the next chunk in flight while the
current one plays, and two generations on one card at once is slower
than doing them in order, not faster.
"""
import io
import json
import os
import threading
import time
import wave

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("CHATTERBOX_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHATTERBOX_PORT", "8899"))

# "turbo" is faster and takes a 10s reference; "" is the original, which
# is the one with the exaggeration and cfg_weight dials.
VARIANT = os.environ.get("CHATTERBOX_VARIANT", "").strip().lower()

# 0.5 is neutral. Higher is more dramatic and less stable; past about
# 0.8 it starts acting rather than talking, which is either what you
# wanted or very much not.
EXAGGERATION = float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.5"))
# Lower slows the delivery down and leaves more room between phrases.
CFG_WEIGHT = float(os.environ.get("CHATTERBOX_CFG", "0.5"))
TEMPERATURE = float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.8"))

DEVICE = os.environ.get("CHATTERBOX_DEVICE", "").strip()
VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

# The assistant chunks to under 1000 characters before it gets here.
# This is the backstop for anything else that posts to the port.
MAX_CHARS = int(os.environ.get("CHATTERBOX_MAX_CHARS", "1200"))

_model = None
_ready = threading.Event()
_load_error = ""
_generating = threading.Lock()

_stats = {"requests": 0, "seconds": 0.0, "audio_seconds": 0.0}


def log(message):
    print(f"[chatterbox] {message}", flush=True)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def pick_device():
    """ROCm reports itself as 'cuda' to torch, so this covers both."""
    if DEVICE:
        return DEVICE

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        # Apple silicon, for completeness.
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def load():
    """Load the model once, on a worker, so the port answers immediately.

    /health reports ok:false until this finishes. The assistant already
    knows what to do with that - it shows the server as offline and
    carries on as a text chat rather than hanging on the first sentence.
    """
    global _model, _load_error

    started = time.time()
    device = pick_device()

    try:
        if VARIANT == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Engine
        elif VARIANT in ("multilingual", "ml"):
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Engine
        else:
            from chatterbox.tts import ChatterboxTTS as Engine

        log(f"loading {VARIANT or 'original'} on {device}...")
        _model = Engine.from_pretrained(device=device)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        log(f"FAILED to load: {_load_error}")

        return

    log(f"ready in {time.time() - started:.1f}s "
        f"| {_model.sr} Hz | device={device}")

    # One short generation now, so the first thing she says isn't the
    # one that pays for CUDA graph setup and lazy weight loading.
    try:
        _model.generate("Ready.")
        log("warmed up")
    except Exception as e:
        log(f"warmup failed (continuing): {e}")

    _ready.set()


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
def voices():
    """Reference clips, by name. The built-in voice is always offered."""
    found = ["default"]

    try:
        for entry in sorted(os.listdir(VOICE_DIR)):
            stem, extension = os.path.splitext(entry)

            if extension.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                found.append(stem)
    except OSError:
        pass

    return found


def reference_for(name):
    """The clip for a voice name, or None for the built-in voice."""
    name = str(name or "").strip()

    if not name or name.lower() in ("default", "none", "built-in", "builtin"):
        return None

    for extension in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        path = os.path.join(VOICE_DIR, name + extension)

        if os.path.exists(path):
            return path

    return None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def stretch(wav, rate):
    """Change the tempo without changing the pitch.

    Chatterbox has no speed control, and the contract has one. Resampling
    would be one line but it moves the pitch with it, which turns "a bit
    faster" into a chipmunk. A phase vocoder keeps the pitch where it
    was, and torchaudio already ships one.
    """
    if abs(rate - 1.0) < 0.02:
        return wav

    try:
        import torch
        import torchaudio

        window = torch.hann_window(1024, device=wav.device)
        spectrogram = torch.stft(
            wav, n_fft=1024, hop_length=256, window=window, return_complex=True,
        )
        phase_advance = torch.linspace(
            0, torch.pi * 256, spectrogram.shape[-2], device=wav.device
        )[..., None]
        stretched = torchaudio.functional.phase_vocoder(
            spectrogram, rate, phase_advance
        )

        return torch.istft(stretched, n_fft=1024, hop_length=256, window=window)
    except Exception as e:
        log(f"speed change failed, sending it unmodified: {e}")

        return wav


def to_wav_bytes(wav, sample_rate):
    """16-bit PCM. The assistant reads any common WAV format, but this is
    the one every other thing that might poll this port also reads."""
    import torch

    audio = wav.detach().to("cpu").float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Clip rather than normalise: normalising makes every sentence a
    # different loudness, which is far more noticeable than a rare
    # clipped peak.
    audio = torch.clamp(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).to(torch.int16).numpy()

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(pcm.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.T.tobytes() if pcm.shape[0] > 1 else pcm.tobytes())

    return buffer.getvalue()


def synthesize(text, voice, speed, options):
    started = time.time()
    reference = reference_for(voice)

    arguments = {
        "exaggeration": float(options.get("exaggeration", EXAGGERATION)),
        "cfg_weight": float(options.get("cfg_weight", CFG_WEIGHT)),
        "temperature": float(options.get("temperature", TEMPERATURE)),
    }

    if reference:
        arguments["audio_prompt_path"] = reference

    with _generating:
        try:
            wav = _model.generate(text, **arguments)
        except TypeError:
            # Turbo takes a narrower set than the original does. Rather
            # than keep a table of which build accepts what, drop the
            # extras and try again - a plainer voice beats no voice.
            wav = _model.generate(
                text, **({"audio_prompt_path": reference} if reference else {})
            )

    wav = stretch(wav.squeeze(), float(speed or 1.0))
    data = to_wav_bytes(wav, _model.sr)

    spent = time.time() - started
    seconds = max(len(wav) / float(_model.sr), 0.001)

    _stats["requests"] += 1
    _stats["seconds"] += spent
    _stats["audio_seconds"] += seconds

    log(f'"{text[:48]}" -> {seconds:.1f}s audio in {spent:.1f}s '
        f"(x{seconds / spent:.1f} realtime, voice={voice or 'default'})")

    return data


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # the useful lines are logged where they happen

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
            body = {"ok": ready, "engine": "chatterbox",
                    "variant": VARIANT or "original", "device": pick_device()}

            if not ready:
                body["loading"] = not _load_error
                body["error"] = _load_error

            # 200 either way: "ok": false is the answer, not an error.
            return self._send(200, body)

        if path == "/voices":
            return self._send(200, {"voices": voices(), "default": "default"})

        if path in ("/", "/stats"):
            done = _stats["requests"] or 1

            return self._send(200, {
                "engine": "chatterbox", "ready": _ready.is_set(),
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
            return self._send(503, {
                "error": _load_error or "still loading the model",
            })

        try:
            audio = synthesize(
                text, payload.get("voice"), payload.get("speed", 1.0), payload,
            )
        except Exception as e:
            log(f"generation failed: {type(e).__name__}: {e}")

            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        return self._send(200, audio, "audio/wav")


def main():
    os.makedirs(VOICE_DIR, exist_ok=True)

    threading.Thread(target=load, daemon=True).start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on http://{HOST}:{PORT}")
    log(f"voices: {', '.join(voices())}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")


if __name__ == "__main__":
    main()
