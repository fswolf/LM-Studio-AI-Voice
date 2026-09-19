import json
import os

from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LM_URL = "http://localhost:1234/v1/chat/completions"
SAMPLE_RATE = 16000

with open(os.path.join(BASE_DIR, "agent", "agent.json"), "r") as file:
    agent = json.load(file)

with open(os.path.join(BASE_DIR, "agent", "memory.json"), "r") as file:
    memory = json.load(file)

AGENT_NAME = agent["name"]

# -------------------------
# TTS (kokoro-reader server)
# -------------------------
# Speech is synthesized by the standalone kokoro-reader server
# (github.com/fswolf/kokoro-reader), not by an in-process KPipeline.
# Start it once and every app that speaks shares the same loaded model.
#
# Optional "tts" block in agent.json:
#   "tts": { "url": "http://127.0.0.1:8899", "speed": 1.0, "volume": 1.0 }
# Environment variables win over agent.json, so KOKORO_VOICE / KOKORO_URL
# match the names the server itself uses.
_tts_cfg = agent.get("tts", {})

KOKORO_URL = os.environ.get("KOKORO_URL", _tts_cfg.get("url", "http://127.0.0.1:8899")).rstrip("/")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", _tts_cfg.get("speed", 1.0)))
KOKORO_VOLUME = float(os.environ.get("KOKORO_VOLUME", _tts_cfg.get("volume", 1.0)))
VOICE = os.environ.get("KOKORO_VOICE", agent.get("voice", "af_bella"))

# Shown on the UI's VServer line.
KOKORO_ADDRESS = urlparse(KOKORO_URL).netloc or KOKORO_URL

# -------------------------
# Control socket (Hyprland / Wayland push-to-talk)
# -------------------------
CONTROL_SOCKET = os.environ.get(
    "AI_VOICE_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "ai-voice.sock"),
)

# Hotkey the assistant registers with Hyprland at startup and removes on
# exit. Override in agent.json: "hotkey": "SUPER ALT, V"
#
# Set it to "" / "none" / "off" if you'd rather bind the key permanently
# in your own config - runtime keywords don't survive a config reload,
# so a config-file bind is the sturdier option.
HOTKEY = os.environ.get("AI_VOICE_HOTKEY", agent.get("hotkey", "SUPER, HOME"))
AUTO_BIND = HOTKEY.strip().lower() not in ("", "none", "off", "false")

# -------------------------
# TUI theme
# -------------------------
# MewNix Candy by default. Override any subset in agent.json:
#   "theme": { "accent": "#ff87d7", "user": "#5fd7ff" }
# Values are hex colours; terminals without truecolor get the nearest
# 256-colour match automatically.
_THEME_DEFAULTS = {
    # sfav-style purple
    "accent": "#ff87d7",
    "border": "#b48cff",
    "title": "#ffafd7",
    "label": "#b48cff",

    # Main text
    "text": "#ccd0ea",
    "value": "#ccd0ea",

    # Agent / user
    "agent": "#ff87d7",
    "user": "#5fd7ff",

    # System/status
    "system": "#d9b46a",
    "ok": "#87ffaf",
    "warn": "#ff5f87",

    # Footer / separators
    "footer": "#8787af",
    "dim": "#5a5a78",

    # Prompt
    "prompt": "#ff87d7",

    # Links
    "link": "#b48cff",
}

THEME = dict(_THEME_DEFAULTS)
THEME.update(
    {k: v for k, v in agent.get("theme", {}).items() if k in _THEME_DEFAULTS}
)

# -------------------------
# Speech-to-text / voice activity detection
# -------------------------
# Optional "stt" block in agent.json:
#   "stt": { "model": "small", "language": "en", "sensitivity": 1.0,
#            "silence_seconds": 1.2, "max_seconds": 60,
#            "no_speech_timeout": 8 }
_stt_cfg = agent.get("stt", {})

# How push-to-talk behaves:
#   auto   - HOME starts listening, silence ends it (the default)
#   manual - HOME starts, HOME stops. Silence is ignored, so you can
#            pause mid-thought without it cutting you off
#   open   - hands free. The mic stays armed, recording starts when you
#            speak and ends on silence, then it re-arms. HOME toggles
#            the whole loop on and off.
STT_MODE = str(_stt_cfg.get("mode", "auto")).strip().lower()

if STT_MODE not in ("auto", "manual", "open"):
    STT_MODE = "auto"

# Which voice-activity detector decides when you're talking.
#   silero - a real speech/not-speech model (~1MB, CPU, already in the
#            requirements). Knows your voice from a fan, so it can
#            trigger fast without also triggering on room noise.
#   energy - plain RMS level. No dependencies, but it can only measure
#            loudness, so it is late to start and cuts off quiet
#            syllables. Kept as a fallback.
#   auto   - silero if importable, else energy.
STT_VAD = str(_stt_cfg.get("vad", "auto")).strip().lower()

if STT_VAD not in ("auto", "silero", "energy"):
    STT_VAD = "auto"

# Silero speech probability above which a frame counts as speech. Lower
# picks up sooner and is more forgiving of a quiet mic; higher ignores
# more background. Speech only *ends* below (threshold - 0.15), which is
# the hysteresis that stops it cutting you off mid-sentence.
STT_VAD_THRESHOLD = float(_stt_cfg.get("vad_threshold", 0.4))

STT_MODEL = _stt_cfg.get("model", "small")
# None lets Whisper auto-detect, which is unreliable on short clips.
STT_LANGUAGE = _stt_cfg.get("language", "en") or None
# >1 makes it easier to trigger on a quiet mic, <1 harder in a noisy room.
STT_SENSITIVITY = float(_stt_cfg.get("sensitivity", 1.0))
# Silence after speech before we stop recording and transcribe.
STT_SILENCE_SECONDS = float(_stt_cfg.get("silence_seconds", 1.2))
# Hard cap on one recording.
STT_MAX_SECONDS = float(_stt_cfg.get("max_seconds", 60))
# Give up if you never started talking (auto mode only - manual waits
# for you, and open mode waits indefinitely by design).
STT_NO_SPEECH_TIMEOUT = float(_stt_cfg.get("no_speech_timeout", 8))
# Backstop for manual mode, where nothing else stops the recording.
STT_MANUAL_MAX_SECONDS = float(_stt_cfg.get("manual_max_seconds", 300))
# Pause after Luna finishes speaking before the mic re-arms in open
# mode, so the tail of her own voice doesn't retrigger it.
STT_SETTLE_SECONDS = float(_stt_cfg.get("settle_seconds", 0.6))

_history_cfg = agent.get("history", {})
MAX_RAW_MESSAGES = _history_cfg.get("max_raw_messages", 15)
SUMMARIZE_CHUNK = _history_cfg.get("summarize_chunk", 8)

LONG_TERM_MEMORY_ENABLED = agent.get("long_term_memory", {}).get("enabled", True)
LONG_TERM_MEMORY_MAX_FACTS = agent.get("long_term_memory", {}).get("max_facts", 40)

# -------------------------
# Tool calling
# -------------------------
# Lets the model call set_reminder, web_search, remember_fact and so on
# for itself instead of relying on keyword triggers. Needs a model with
# a tool template (Qwen, Llama 3.1+, Mistral, Hermes...); if the server
# rejects the payload we fall back automatically at runtime.
_tools_cfg = agent.get("tools", {})
TOOLS_ENABLED = _tools_cfg.get("enabled", True)
# How many times the model may call tools before it must answer in prose.
MAX_TOOL_ROUNDS = int(_tools_cfg.get("max_rounds", 4))

_reminders_cfg = agent.get("reminders", {})
REMINDERS_ENABLED = _reminders_cfg.get("enabled", True)
REMINDER_CHECK_INTERVAL_SECONDS = _reminders_cfg.get("check_interval_minutes", 10) * 60

_web_search_cfg = agent.get("web_search", {})
WEB_SEARCH_ENABLED = _web_search_cfg.get("enabled", True)
WEB_SEARCH_MAX_RESULTS = _web_search_cfg.get("max_results", 5)
