<img width="1210" alt="Luna AI Assistant" src="assets/ui-conversation.png" />

# AI Voice Assistant

A local AI voice assistant powered by:

- 🎤 Faster-Whisper (Speech-to-Text)
- 🧠 LM Studio (Local LLM) with tool calling
- 🗣️ [kokoro-reader](https://github.com/fswolf/kokoro-reader) (Kokoro TTS over local HTTP)
- 🎹 Push-to-talk, three modes including hands free
- 💬 Full-screen terminal interface

Everything runs locally. No cloud APIs required.

---

# Requirements

- Python 3.12+
- LM Studio
- A downloaded language model
- LM Studio Local Server enabled
- [kokoro-reader](https://github.com/fswolf/kokoro-reader) running locally

Speech is **not** synthesized in this process. The assistant is a client
of the kokoro-reader server and stays silent without it.

---

# Create a Virtual Environment

```bash
python3.12 -m venv ai-voice-venv

source ai-voice-venv/bin/activate
```

# Python Dependencies

```text
requests>=2.32.0
numpy>=2.2.0
scipy>=1.15.0
sounddevice>=0.5.2
silero-vad
faster-whisper>=1.1.0
ctranslate2>=4.6.0
kokoro>=0.9.4
torch>=2.8.0
torchaudio>=2.8.0
huggingface_hub>=0.34.0
pynput>=1.8.1
evdev>=1.9.2
prompt_toolkit>=3.0.0
wcwidth>=0.8.2
ddgs>=9.14.4
```

Install dependencies:

```bash
pip install \
requests \
numpy \
scipy \
sounddevice \
silero-vad \
faster-whisper \
ctranslate2 \
kokoro \
torch \
torchaudio \
huggingface_hub \
pynput \
evdev \
prompt_toolkit \
wcwidth \
ddgs
```

`kokoro`, `torch` and `torchaudio` are for the **TTS server**, not the
assistant. If the server has its own venv, the assistant only needs
`requests` to speak.

---

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# Linux Dependencies

## Debian / Ubuntu

```bash
sudo apt install \
python3-dev \
build-essential \
portaudio19-dev \
ffmpeg
```

## Fedora

```bash
sudo dnf install \
ffmpeg \
portaudio-devel \
python3-devel \
gcc
```

---

# macOS Dependencies

Install Homebrew if needed, then:

```bash
brew install portaudio ffmpeg
```

---

# Running LM Studio

1. Install LM Studio
2. Download a model
3. Start the Local Server
4. Verify it is running on:

```
http://localhost:1234
```

---

# Running the Kokoro Server

Speech is synthesized by [kokoro-reader](https://github.com/fswolf/kokoro-reader)
rather than loading Kokoro in-process. One loaded model serves every app
that asks, so the assistant starts in seconds instead of waiting on its
own copy.

```bash
git clone https://github.com/fswolf/kokoro-reader.git ~/kokoro-reader

source ~/ai-voice-venv/bin/activate
KOKORO_VOICE=af_bella python3 ~/kokoro-reader/server/kokoro_server.py
```

Check it:

```bash
curl -s localhost:8899/health
curl -s localhost:8899/voices
```

Long replies are split on sentence boundaries under the server's
1200-character limit, and the next chunk is synthesized while the
current one plays. If the server isn't answering, the assistant says so
and keeps working as a text chat.

Optional `tts` block in `agent/agent.json`:

```json
"tts": {
    "url": "http://127.0.0.1:8899",
    "speed": 1.0,
    "volume": 1.0
}
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `KOKORO_URL` | `http://127.0.0.1:8899` | Server address |
| `KOKORO_VOICE` | `agent.json` → `voice` | Voice (`af_bella`, `am_adam`, ...) |
| `KOKORO_SPEED` | `1.0` | 0.5 – 2.0 |
| `KOKORO_VOLUME` | `1.0` | Playback gain |

Environment variables win over `agent.json`.

---

# Run the Assistant

```bash
python main.py
```

---

# Controls

| Key | Action |
|------|--------|
| Home | Push to talk — depends on the voice mode below |
| Enter | Send typed message |
| Tab | Open / close the help panel |
| PgUp / PgDn | Scroll the conversation |
| End | Jump back to the newest message |
| Esc | Quit |

<img width="1210" alt="Help panel" src="assets/ui-help.png" />

Slash commands:

| Command | Action |
|---------|--------|
| `/mode` | `auto`, `manual` or `open` — switch without restarting |
| `/mic` | What the voice detector measured on the last recording |
| `/reminders` | List what's scheduled, with countdowns |
| `/cancel N` | Cancel reminder N |
| `/tools` | Which tools the model can call |
| `/keys` | Hotkey + socket diagnostics |
| `/clear` | Wipe the conversation and saved history |
| `/quit` | Exit |

## Theme

```json
"theme": {
    "accent": "#ff87d7",
    "border": "#b48cff",
    "user":   "#5fd7ff",
    "agent":  "#ff87d7"
}
```

Keys: `accent`, `border`, `title`, `label`, `text`, `value`, `agent`,
`user`, `system`, `ok`, `warn`, `footer`, `dim`, `prompt`, `link`.
Terminals without truecolor fall back to the nearest 256-colour match.

---

# Voice Modes

How HOME behaves. Set `stt.mode` in `agent/agent.json` or switch live
with `/mode <name>`.

| Mode | HOME | Recording ends when |
|------|------|---------------------|
| `auto` | starts listening | you stop talking (default) |
| `manual` | starts recording immediately | you press HOME again |
| `open` | arms the mic and leaves it armed | you stop talking — then it re-arms |

```
auto   - the quick question. Waits for you to speak, stops on silence.

manual - records the moment you press and ignores silence entirely, so
         you can think, trail off and come back. Press again to stop.

open   - hands free. Speech starts a turn, silence ends it, Luna
         answers, and the mic re-arms. HOME toggles the whole loop.
```

Open mode only records *between* turns, never while Luna is speaking,
and waits `settle_seconds` after she finishes before re-arming —
otherwise her own voice out of the speakers retriggers the mic and she
talks to herself. Headphones make it moot.

---

# Speech Detection

Speech is detected with [Silero VAD](https://github.com/snakers4/silero-vad),
a small speech/not-speech model, rather than by measuring loudness. A
level meter can't tell your voice from a fan, so the bar has to sit high
enough to ignore the room — which makes it late to trigger *and* prone
to cutting off quiet syllables. Silero knows the difference, and only
ends a turn below `threshold - 0.15`, so trailing off doesn't end the
recording.

Without `silero-vad` installed it falls back to an RMS detector that
calibrates against your noise floor. `/mic` says which is running:

```
vad=silero | speech above 0.40, ends below 0.25 | last peak=0.0210 |
triggered=True | captured=3.4s | mode=auto
```

Cutting off or slow to start? Lower `vad_threshold` to 0.25–0.3.
Triggering on background noise? Raise it to 0.5–0.6.

```json
"stt": {
    "mode": "auto",
    "vad": "auto",
    "vad_threshold": 0.4,
    "model": "small",
    "language": "en",
    "sensitivity": 1.0,
    "silence_seconds": 1.2,
    "max_seconds": 60,
    "manual_max_seconds": 300,
    "no_speech_timeout": 8,
    "settle_seconds": 0.6
}
```

| Key | Default | Purpose |
|-----|---------|---------|
| `mode` | `auto` | `auto`, `manual` or `open` |
| `vad` | `auto` | `silero`, `energy`, or `auto` |
| `vad_threshold` | `0.4` | Silero speech probability. Lower picks up sooner |
| `model` | `small` | Whisper size — `tiny`, `base`, `small`, `medium` |
| `language` | `en` | `null` to auto-detect (unreliable on short clips) |
| `sensitivity` | `1.0` | Energy fallback only. >1 triggers more easily |
| `silence_seconds` | `1.2` | Quiet time before it stops and transcribes |
| `max_seconds` | `60` | Cap on one recording in auto/open |
| `manual_max_seconds` | `300` | Backstop in manual |
| `no_speech_timeout` | `8` | Give up if you never start talking (auto only) |
| `settle_seconds` | `0.6` | Pause before re-arming in open mode |

---

# Talking From Other Windows (Optional)

HOME works while the assistant's window is focused, on Linux, macOS and
Windows, with no permissions and nothing to configure.

A *global* hotkey — one that fires while your browser or a game is
focused — is platform-specific. On X11 or a TTY, `evdev` handles it if
you can read the input devices:

```bash
sudo usermod -aG input "$USER"   # then log out and back in
```

On Wayland there is no application-level global key grab; that's a
protocol decision, not a missing library. The compositor has to own the
hotkey and tell the assistant, which is what the control socket at
`$XDG_RUNTIME_DIR/ai-voice.sock` is for:

```conf
# hyprland.conf
bind = SUPER, HOME, exec, python3 ~/ai-voice/ai-voice-ctl.py ptt
```

```lua
-- hyprland.lua
hl.bind("SUPER + HOME", hl.dsp.exec_cmd("python3 ~/ai-voice/ai-voice-ctl.py ptt"))
```

`ai-voice-ctl.py` is stdlib-only and needs no virtualenv, which matters
because `exec` runs outside it. It takes `ptt`, `stop` or `quit`.

> Keep a modifier. A bare `HOME` bind is swallowed compositor-wide, so
> `Home` stops working in your terminal, editor and browser — including
> the assistant's own prompt.

---

# Tool Calling

The model calls tools for itself instead of relying on keyword triggers,
so it decides when a question needs looking up and can chain steps —
check the time, then schedule something.

| Tool | What it does |
|------|--------------|
| `get_datetime` | Current date and time, so it stops guessing |
| `set_reminder` | Schedule N minutes from now, optionally repeating |
| `set_reminder_at` | Schedule at a clock time, optionally daily |
| `list_reminders` / `cancel_reminder` | Read and cancel what's pending |
| `remember_fact` / `recall_facts` | Long-term memory, written deliberately |
| `web_search` | DuckDuckGo, for anything it can't know |

Every call is echoed into the conversation, so it's never a mystery why
a reminder appeared:

```
sys  │ set_reminder() -> Scheduled: 14:40 - stretch (in 10m)
```

Needs a model with a tool template — Qwen, Llama 3.1+, Mistral, Hermes
and similar. If LM Studio rejects the payload, tool calling switches off
for the session and the original keyword triggers take over, so loading
a model without tool support degrades rather than breaks.

```json
"tools": {
    "enabled": true,
    "max_rounds": 4
}
```

`max_rounds` caps how many times the model may call tools before it has
to answer in words.

> Web results are untrusted text. They're handed to the model labelled
> as data to summarize, never as instructions — worth remembering before
> adding any tool with side effects.

---

# Reminders

```
Ask in plain language:

> "Hey Luna, remind me in 10 minutes to clean the desk."
> "Wake me up at 7:30 tomorrow."
> "Every morning at 8 remind me to take my meds."
> "Every 30 minutes remind me to fix my posture."

The model only extracts what you said - a number, a unit, a clock time.
All the date arithmetic happens in Python, because models are bad at
"what time is it in 90 minutes" and fine at "the user said 90 minutes".
```

Each one confirms itself the moment it's scheduled:

```
sys  │ Reminder set - 14:40 - stretch (in 10m)
```

If it looked like a reminder but the time couldn't be read, it says that
too, rather than failing silently.

Reminders live in `reminders/reminders.json` and survive restarts.
Anything that came due while the app was closed is delivered in one
message on next launch. A reminder is removed only once delivery
succeeds — if LM Studio is down when it fires, it retries rather than
vanishing. The scanner sleeps until the next one is actually due, so "in
one minute" means one minute.

---

# Web Search

```
Handled by the web_search tool: the model decides a question needs
looking up and calls it, rather than you having to say the words
"web search" in your sentence.

Results come from DuckDuckGo via the `ddgs` package - no API key.

Without tool calling it falls back to the old keyword trigger.
```

---

# Extras

```bash
./push.sh "commit message"            # commit and push, with a check for
                                      # personal files still being tracked

python3 kokoro-say.py                 # read the clipboard aloud through
                                      # the kokoro server - bind it to a key
```

---

# Project Structure

```text
ai-voice/
│
├── ai-voice-ctl.py
├── assistant.py
├── config.py
├── control.py
├── history.py
├── kokoro-say.py
├── llm.py
├── longterm.py
├── main.py
├── ptt.py
├── push.sh
├── reminders.py
├── speech.py
├── state.py
├── tools.py
├── ui.py
├── websearch.py
│
├── agent/
│   ├── agent.json
│   └── memory.json
│
├── assets/
│   ├── ui-conversation.png
│   └── ui-help.png
│
├── history/
│   └── conversation.json
│
├── input/
│   ├── __init__.py
│   ├── linux_keyboard.py
│   ├── mac_keyboard.py
│   └── windows_keyboard.py
│
├── reminders/
│   └── reminders.json
│
└── README.md
```

---

# Features

- Local speech recognition with Silero voice detection
- Local language model with tool calling
- Local text-to-speech via a shared Kokoro server
- Three push-to-talk modes, including hands free
- Configurable AI personality
- Long-term memory the model writes deliberately
- Reminders with repeats, confirmation and retry
- Web search the model reaches for on its own
- Themeable full-screen terminal interface
- Cross-platform architecture

---

# License

MIT License
