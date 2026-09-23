import json
import os

from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LM_URL = "http://localhost:1234/v1/chat/completions"
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Two files, two jobs.
#
#   agent/agent.json  - who she is. Name, personality, tone, traits, rules.
#   config.json       - how the machine runs. Voice, models, thresholds,
#                       timeouts, theme.
#
# They were one file, which meant tuning a VAD threshold and rewriting her
# personality were the same edit, and you couldn't share either one without
# handing over the other.
#
# config.json wins where both define a key, and agent.json is still read as
# a fallback so an install that never split them keeps working untouched.
# ---------------------------------------------------------------------------
def _load_json(path, on_missing=None):
    """Read a JSON file, or explain why not and carry on.

    A stray comma used to take the whole app down on startup with a bare
    JSONDecodeError - and hand-editing these files is the entire point of
    them being JSON, so a typo shouldn't be fatal.
    """
    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {} if on_missing is None else on_missing()
    except (json.JSONDecodeError, OSError) as error:
        print(
            f"\nCouldn't read {path}:\n  {error}\n"
            "Carrying on with defaults. The file has been left alone - "
            "fix the syntax and restart.\n"
        )
        return {}


AGENT_FILE = os.path.join(BASE_DIR, "agent", "agent.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MEMORY_FILE = os.path.join(BASE_DIR, "agent", "memory.json")

agent = _load_json(AGENT_FILE)
settings = _load_json(CONFIG_FILE)
memory = _load_json(MEMORY_FILE)


def setting(key, default=None):
    """A machine setting: config.json first, then agent.json."""
    if key in settings:
        return settings[key]

    return agent.get(key, default)


AGENT_NAME = agent.get("name") or settings.get("name") or "Luna"

# Persona, kept separate from the machine settings so llm.py doesn't have
# to know which file anything came from.
PERSONALITY = agent.get("personality", "")
TONE = agent.get("tone", "")
TRAITS = agent.get("traits", [])
RULES = agent.get("rules", [])
GENERATION = setting("generation", {})

# -------------------------
# TTS (kokoro-reader server)
# -------------------------
# Speech is synthesized by the standalone kokoro-reader server
# (github.com/fswolf/kokoro-reader), not by an in-process KPipeline.
# Start it once and every app that speaks shares the same loaded model.
#
# Optional "tts" block in agent.json:
#   "tts": { "url": "http://127.0.0.1:8899", "speed": 1.0, "volume": 1.0 }
# Named for the job, not the server. kokoro-reader is what this speaks
# to today, but the client only needs POST /tts, GET /health and GET
# /voices - anything answering that contract works, so the constants
# shouldn't be called KOKORO_*.
#
# The KOKORO_* environment variables still work, because kokoro-reader
# uses those names and it would be rude to break someone's shell.
_tts_cfg = setting("tts", {})


def _tts_env(name, fallback):
    return os.environ.get(f"TTS_{name}") or os.environ.get(f"KOKORO_{name}") \
        or fallback


TTS_URL = str(_tts_env("URL", _tts_cfg.get("url", "http://127.0.0.1:8899"))).rstrip("/")
TTS_SPEED = float(_tts_env("SPEED", _tts_cfg.get("speed", 1.0)))
TTS_VOLUME = float(_tts_env("VOLUME", _tts_cfg.get("volume", 1.0)))
# Semitones. Positive is higher and younger-sounding, negative lower and
# older. Applied here rather than in the speech server, so it works with
# whichever engine is on the port - Kokoro today, something else later.
#
# Deliberately shifts the formants along with the pitch, which is what a
# naive resample does. A formant-preserving shift keeps the same person
# sounding like themselves an octave up; moving them together is what
# reads as a different age of person, which is the point here.
TTS_PITCH = float(_tts_env("PITCH", _tts_cfg.get("pitch", 0.0)))
VOICE = _tts_env("VOICE", setting("voice", "af_bella"))

# Shown on the UI's TTS line.
TTS_ADDRESS = urlparse(TTS_URL).netloc or TTS_URL

# Old names, kept so nothing outside this file has to change at once.
KOKORO_URL, KOKORO_SPEED = TTS_URL, TTS_SPEED
KOKORO_VOLUME, KOKORO_ADDRESS = TTS_VOLUME, TTS_ADDRESS

# -------------------------
# Control socket (Hyprland / Wayland push-to-talk)
# -------------------------
CONTROL_SOCKET = os.environ.get(
    "AI_VOICE_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "ai-voice.sock"),
)

# -------------------------
# TUI
# -------------------------
# Mouse capture gives wheel scrolling at the cost of text selection -
# the terminal can only hand drags to one of them. Off by default,
# because a window full of log lines you can't copy is worse than one
# you scroll with PgUp.
UI_MOUSE = bool(setting("ui", {}).get("mouse", False))

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
    {k: v for k, v in setting("theme", {}).items() if k in _THEME_DEFAULTS}
)

# -------------------------
# Speech-to-text / voice activity detection
# -------------------------
# Optional "stt" block in agent.json:
#   "stt": { "model": "small", "language": "en", "sensitivity": 1.0,
#            "silence_seconds": 1.2, "max_seconds": 60,
#            "no_speech_timeout": 8 }
_stt_cfg = setting("stt", {})

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

# -------------------------
# Barge-in
# -------------------------
# Listen while Luna is speaking, and cut her off when you start
# talking. Needs Silero - the energy detector can't tell your voice
# from her own coming back through the speakers.
#
# That bleed is the whole problem with barge-in and there is no
# acoustic echo cancellation here, so the mic level during playback is
# measured and used as the baseline: you have to be noticeably louder
# than she is at the microphone. On headphones that's trivially true.
# On speakers, turn the volume down or raise the margin.
STT_BARGE_IN = bool(_stt_cfg.get("barge_in", True))
# How long you have to keep talking before she stops. Too low and a
# cough cuts her off; too high and you're talking over her for a
# second before she notices.
STT_BARGE_IN_SECONDS = float(_stt_cfg.get("barge_in_seconds", 0.35))
# How much louder than her own bleed you have to be. Lower it on
# headphones, raise it if she keeps interrupting herself.
STT_BARGE_IN_MARGIN = float(_stt_cfg.get("barge_in_margin", 2.5))
# Added to the normal VAD threshold while she's talking - a higher bar
# for "that's speech" when we already know speech is playing.
STT_BARGE_IN_BOOST = float(_stt_cfg.get("barge_in_boost", 0.25))

# -------------------------
# Wake word
# -------------------------
# Replaces HOME in open mode: say her name and she listens. Optional
# "wake_word" block in agent.json:
#
#   "wake_word": {
#       "enabled": true,
#       "model": "~/models/hey_luna.onnx",
#       "threshold": 0.5,
#       "follow_up_seconds": 12
#   }
#
# model is either a path to one you trained with openWakeWord, or the
# name of one of theirs (hey_jarvis, alexa, hey_mycroft, hey_rhasspy).
# There is no pretrained "hey Luna" - see wakeword.py.
_wake_cfg = setting("wake_word", {})
WAKE_WORD_ENABLED = bool(_wake_cfg.get("enabled", False))
WAKE_WORD_MODEL = _wake_cfg.get("model", "hey_jarvis")
# Raise if it fires at the television, lower if it ignores you.
WAKE_WORD_THRESHOLD = float(_wake_cfg.get("threshold", 0.5))
# After answering, how long she keeps listening without the wake word,
# so a follow-up question doesn't need her name again.
WAKE_WORD_FOLLOW_UP_SECONDS = float(_wake_cfg.get("follow_up_seconds", 12))
# Deaf period right after a detection, so the tail of "hey Luna" isn't
# heard as a second one.
WAKE_WORD_COOLDOWN = float(_wake_cfg.get("cooldown_seconds", 1.0))

# -------------------------
# Logging
# -------------------------
# Writes to ~/.cache/ai-voice/ai-voice.log. Off costs nothing, on costs
# a few kilobytes a day and turns "it broke" into a line you can read.
_log_cfg = setting("logging", {})
LOG_ENABLED = bool(_log_cfg.get("enabled", True))
# debug logs every VAD decision and tool argument, which is a lot; info
# logs the things that went wrong and the choices behind them.
LOG_LEVEL = str(_log_cfg.get("level", "info")).upper()
LOG_MAX_KB = int(_log_cfg.get("max_kb", 1024))
LOG_KEEP = int(_log_cfg.get("keep", 3))

# -------------------------
# Vision
# -------------------------
# Lets her look at your screen. Needs grim, and needs a vision model
# loaded in LM Studio - a text-only model will reject the request and
# the error is shown as-is. Optional "vision" block in agent.json:
#   "vision": { "enabled": true, "scale": 0.5 }
_vision_cfg = setting("vision", {})
VISION_ENABLED = bool(_vision_cfg.get("enabled", False))
# grim's own scale factor. Half a 4K screen is still plenty to read an
# error dialog off, and a quarter of the bytes.
VISION_SCALE = float(_vision_cfg.get("scale", 0.5))
VISION_MAX_BYTES = int(_vision_cfg.get("max_kb", 4096)) * 1024

# -------------------------
# Desktop control
# -------------------------
# Acting on the desktop rather than only looking at it - playback,
# volume, clipboard, window focus. Each tool checks for its own program
# at call time, so a missing playerctl costs that one tool rather than
# all four. Optional "desktop" block in config.json:
#   "desktop": { "enabled": true, "clipboard": true }
_desktop_cfg = setting("desktop", {})
DESKTOP_ENABLED = bool(_desktop_cfg.get("enabled", True))
# Separate from the rest, because reading the clipboard means whatever
# you last copied - a password, an API key - can land in the model's
# context and from there in history on disk. On by default, but it is
# the one here worth knowing is a switch.
DESKTOP_CLIPBOARD = bool(_desktop_cfg.get("clipboard", True))

# -------------------------
# Plugins
# -------------------------
# Everything under "plugins" in config.json, keyed by plugin name. The
# core never learns what a plugin's own keys mean - it just hands the
# block over - which is what keeps a plugin droppable.
#
# "chat" is the exception: shared policy for any plugin that feeds her
# live chat from strangers. It lives here rather than in a plugin
# because it is the part that must not be per-plugin. A plugin can ask
# for a wider tool list; chatroom.TOOL_CEILING is what it gets.
#
# API keys are deliberately not in here. config.json is in the repo,
# and a key pasted into it is a key on GitHub - plugins read their own
# from the environment or from ~/.config/ai-voice/.
_plugins_cfg = setting("plugins", {})

_chat_cfg = _plugins_cfg.get("chat") or {}
# Which tools a stranger's question may reach. Everything absent from
# this acts on Ryan's machine or Ryan's data, and a viewer must not be
# able to reach it by asking nicely: the tools are dropped from the
# request, not refused at call time.
CHAT_TOOLS = tuple(_chat_cfg.get(
    "tools", ["get_datetime", "time_until", "web_search"]
))
# Seconds between answers, so she isn't talking over the stream.
CHAT_COOLDOWN_SECONDS = float(_chat_cfg.get("cooldown_seconds", 8))
# And per viewer, so one person can't monopolise her.
CHAT_USER_COOLDOWN_SECONDS = float(_chat_cfg.get("user_cooldown_seconds", 30))
# A very long chat message is a paste, and a paste aimed at her is
# usually an attempt at something.
CHAT_MAX_MESSAGE_CHARS = int(_chat_cfg.get("max_message_chars", 300))
# How much of the room she can see. Memory only, never stored.
CHAT_CONTEXT_LINES = int(_chat_cfg.get("context_lines", 12))


def plugin_settings(name):
    """One plugin's settings, over the shared chat defaults.

    Merged rather than either/or so a chat plugin gets the limits for
    free and overrides only what is genuinely its own - a channel name,
    a list of bots to ignore.
    """
    merged = {
        "tools": list(CHAT_TOOLS),
        "cooldown_seconds": CHAT_COOLDOWN_SECONDS,
        "user_cooldown_seconds": CHAT_USER_COOLDOWN_SECONDS,
        "max_message_chars": CHAT_MAX_MESSAGE_CHARS,
        "context_lines": CHAT_CONTEXT_LINES,
    }

    block = _plugins_cfg.get(name)

    if isinstance(block, dict):
        merged.update(block)

    return merged


def set_plugin_enabled(name, on):
    """Remember that a plugin was switched on, for next launch."""
    _store(f"plugins.{name}.enabled", bool(on))

    block = _plugins_cfg.setdefault(name, {})

    if isinstance(block, dict):
        block["enabled"] = bool(on)

_history_cfg = setting("history", {})
MAX_RAW_MESSAGES = _history_cfg.get("max_raw_messages", 15)
SUMMARIZE_CHUNK = _history_cfg.get("summarize_chunk", 8)
# The running summary is re-compressed once it passes this, rather than
# being appended to forever.
SUMMARY_MAX_CHARS = int(_history_cfg.get("summary_max_chars", 2500))
# A full archive of every turn, which summarization never prunes. It is
# what search_history reads; the prompt never sees it.
TRANSCRIPT_ENABLED = bool(_history_cfg.get("transcript", True))
TRANSCRIPT_MAX_MB = float(_history_cfg.get("transcript_max_mb", 20))

_memory_cfg = setting("long_term_memory", {})
LONG_TERM_MEMORY_ENABLED = _memory_cfg.get("enabled", True)
LONG_TERM_MEMORY_MAX_FACTS = _memory_cfg.get("max_facts", 40)
# How many facts may go into one system prompt. Under this, all of them
# do; over it, the ones relevant to what was just said.
LONG_TERM_MEMORY_CONTEXT_FACTS = _memory_cfg.get("context_facts", 25)

# -------------------------
# Tool calling
# -------------------------
# Lets the model call set_reminder, web_search, remember_fact and so on
# for itself instead of relying on keyword triggers. Needs a model with
# a tool template (Qwen, Llama 3.1+, Mistral, Hermes...); if the server
# rejects the payload we fall back automatically at runtime.
_tools_cfg = setting("tools", {})
TOOLS_ENABLED = _tools_cfg.get("enabled", True)
# How many times the model may call tools before it must answer in prose.
MAX_TOOL_ROUNDS = int(_tools_cfg.get("max_rounds", 4))

_reminders_cfg = setting("reminders", {})
REMINDERS_ENABLED = _reminders_cfg.get("enabled", True)
REMINDER_CHECK_INTERVAL_SECONDS = _reminders_cfg.get("check_interval_minutes", 10) * 60

_web_search_cfg = setting("web_search", {})
WEB_SEARCH_ENABLED = _web_search_cfg.get("enabled", True)
WEB_SEARCH_MAX_RESULTS = _web_search_cfg.get("max_results", 5)
# Opening a page she found, rather than summarizing the search blurb.
PAGE_FETCH_ENABLED = bool(_web_search_cfg.get("fetch_pages", True))
# How much of a page reaches the model. Roughly a thousand tokens per
# 4000 characters, so this is the main cost control.
PAGE_MAX_CHARS = int(_web_search_cfg.get("page_max_chars", 6000))
PAGE_TIMEOUT = float(_web_search_cfg.get("page_timeout", 20))


# ---------------------------------------------------------------------------
# Changing settings from the terminal
#
# Every value above is read once at import and copied into a constant,
# which is fast and simple and means nothing can be changed at runtime.
# That's fine for most of them - you don't retune a Whisper model size
# mid-conversation - but it's wrong for the handful you actually tune by
# ear, where the loop is "change it, say something, listen, change it
# again" and a restart each time makes it useless.
#
# So: the ones worth tuning live are listed here by their dotted path in
# config.json, and the modules that use them read config.<NAME> at call
# time rather than importing the value. Everything else saves to the file
# and waits for a restart, which /set says plainly.
# ---------------------------------------------------------------------------
SETTINGS = {
    # path in config.json          constant        applies without a restart
    "voice":                      ("VOICE",                        False),
    "generation.max_tokens":      (None,                           False),
    "generation.reasoning":       (None,                           False),

    "stt.mode":                   ("STT_MODE",                     True),
    "stt.vad_threshold":          ("STT_VAD_THRESHOLD",            True),
    "stt.sensitivity":            ("STT_SENSITIVITY",              True),
    "stt.silence_seconds":        ("STT_SILENCE_SECONDS",          True),
    "stt.max_seconds":            ("STT_MAX_SECONDS",              True),
    "stt.manual_max_seconds":     ("STT_MANUAL_MAX_SECONDS",       True),
    "stt.no_speech_timeout":      ("STT_NO_SPEECH_TIMEOUT",        True),
    "stt.settle_seconds":         ("STT_SETTLE_SECONDS",           True),
    "stt.barge_in":               ("STT_BARGE_IN",                 True),
    "stt.barge_in_seconds":       ("STT_BARGE_IN_SECONDS",         True),
    "stt.barge_in_margin":        ("STT_BARGE_IN_MARGIN",          True),
    "stt.barge_in_boost":         ("STT_BARGE_IN_BOOST",           True),
    "stt.vad":                    ("STT_VAD",                      False),
    "stt.model":                  ("STT_MODEL",                    False),
    "stt.language":               ("STT_LANGUAGE",                 False),

    "tts.speed":                  ("TTS_SPEED",                    True),
    "tts.volume":                 ("TTS_VOLUME",                   True),
    "tts.pitch":                  ("TTS_PITCH",                    True),
    "tts.url":                    ("TTS_URL",                      False),

    "ui.mouse":                   ("UI_MOUSE",                     True),

    "logging.enabled":            ("LOG_ENABLED",                  False),
    "logging.level":              ("LOG_LEVEL",                    False),

    "vision.enabled":             ("VISION_ENABLED",               True),
    "vision.scale":               ("VISION_SCALE",                 True),

    "desktop.enabled":            ("DESKTOP_ENABLED",              True),
    "desktop.clipboard":          ("DESKTOP_CLIPBOARD",            True),

    # Shared by every chat plugin. A plugin's own keys - channels,
    # account ids - are edited in config.json; these are the ones worth
    # turning mid-stream.
    #
    # Marked live, though a ChatRoom copies them when it is built - so
    # "live" here means the next /<plugin> off, /<plugin> on picks them
    # up, rather than a whole restart. Saying "needs a restart" when
    # toggling the plugin is enough sends people the long way round.
    "plugins.chat.cooldown_seconds":      ("CHAT_COOLDOWN_SECONDS",      True),
    "plugins.chat.user_cooldown_seconds": ("CHAT_USER_COOLDOWN_SECONDS", True),
    "plugins.chat.context_lines":         ("CHAT_CONTEXT_LINES",         True),
    "plugins.chat.max_message_chars":     ("CHAT_MAX_MESSAGE_CHARS",     True),

    "wake_word.enabled":          ("WAKE_WORD_ENABLED",            False),
    "wake_word.model":            ("WAKE_WORD_MODEL",              False),
    "wake_word.threshold":        ("WAKE_WORD_THRESHOLD",          True),
    "wake_word.follow_up_seconds": ("WAKE_WORD_FOLLOW_UP_SECONDS", True),

    "tools.enabled":              ("TOOLS_ENABLED",                False),
    "tools.max_rounds":           ("MAX_TOOL_ROUNDS",              True),

    "web_search.enabled":         ("WEB_SEARCH_ENABLED",           True),
    "web_search.max_results":     ("WEB_SEARCH_MAX_RESULTS",       True),
    "web_search.fetch_pages":     ("PAGE_FETCH_ENABLED",           True),
    "web_search.page_max_chars":  ("PAGE_MAX_CHARS",               True),

    "reminders.enabled":          ("REMINDERS_ENABLED",            True),

    "history.max_raw_messages":   ("MAX_RAW_MESSAGES",             True),
    "history.summarize_chunk":    ("SUMMARIZE_CHUNK",              True),
    "history.summary_max_chars":  ("SUMMARY_MAX_CHARS",            True),
    "history.transcript":         ("TRANSCRIPT_ENABLED",           True),

    "long_term_memory.enabled":   ("LONG_TERM_MEMORY_ENABLED",     True),
    "long_term_memory.max_facts": ("LONG_TERM_MEMORY_MAX_FACTS",   True),
    "long_term_memory.context_facts": ("LONG_TERM_MEMORY_CONTEXT_FACTS", True),
}

_TRUE = ("1", "true", "yes", "on", "y")
_FALSE = ("0", "false", "no", "off", "n")


def current(path):
    """The value in effect right now, whatever file it came from."""
    name = SETTINGS.get(path, (None, False))[0]

    if name and name in globals():
        return globals()[name]

    node = settings

    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None

        node = node[part]

    return node


def _stored(path):
    """The value as it sits in config.json, or None if it isn't there.

    Preferred over the live constant when deciding a type: config.py
    wraps most numbers in float(), so "60" in the file becomes 60.0 in
    memory and writing that back turns every integer setting into a
    decimal for no reason.
    """
    node = settings

    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None

        node = node[part]

    return node


def _coerce(path, raw):
    """Turn terminal text into the type the setting already has."""
    if not isinstance(raw, str):
        return raw

    existing = _stored(path)

    if existing is None:
        existing = current(path)

    text = raw.strip()

    if isinstance(existing, bool):
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False

        raise ValueError(f"{path} is on/off - say true or false, not {raw!r}")

    if isinstance(existing, int) and not isinstance(existing, bool):
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"{path} is a number - {raw!r} isn't one") from None

        # Keep it an int if it still is one, so config.json doesn't
        # gain a decimal point every time you touch it.
        return int(number) if number.is_integer() else number

    if isinstance(existing, float):
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{path} is a number - {raw!r} isn't one") from None

    return text


def save_setting(path, raw):
    """Write one dotted key into config.json, and apply it if it can be.

    Returns (value, applied_now). Raises ValueError for an unknown key or
    a value of the wrong shape, so /set can say what went wrong.
    """
    if path not in SETTINGS:
        raise ValueError(f"unknown setting {path!r}")

    value = _coerce(path, raw)
    name, live = SETTINGS[path]

    # Write the file first: if saving fails, nothing should look applied.
    _store(path, value)

    if name:
        globals()[name] = value

    return value, live


def _store(path, value):
    """Write one dotted key into config.json, creating the nesting.

    Split out of save_setting because plugin switches aren't in
    SETTINGS and never will be - the core doesn't know what plugins
    exist until it has looked in the folder, and a whitelist it can't
    populate isn't a whitelist.
    """
    stored = _load_json(CONFIG_FILE)
    node = stored
    parts = path.split(".")

    for part in parts[:-1]:
        existing = node.get(part)
        node = node.setdefault(part, {}) if isinstance(existing, (dict, type(None))) else {}

    node[parts[-1]] = value

    with open(CONFIG_FILE, "w") as file:
        json.dump(stored, file, indent=4)
        file.write("\n")

    settings.clear()
    settings.update(stored)
