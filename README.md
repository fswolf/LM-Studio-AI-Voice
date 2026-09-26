<img width="1210" alt="Luna AI Assistant" src="assets/ui-conversation.png" />

# AI Voice Assistant

A local AI voice assistant powered by:

- 🎤 Faster-Whisper (Speech-to-Text)
- 🧠 LM Studio (Local LLM) with tool calling
- 🗣️ [kokoro-reader](https://github.com/fswolf/kokoro-reader) (Kokoro TTS over local HTTP)
- 🎹 Push-to-talk, three modes including hands free
- ⚡ Streaming replies — she starts talking a sentence in, not at the end
- ✋ Barge-in — talk over her and she stops
- 👀 Optional vision — she can look at your screen
- ⏰ Reminders and real alarms — she wakes you up, and nags until you're up
- 📝 Writes and edits your files — every change shown in a permission popup first
- 🌙 Moods — she runs warmer or flatter with the clock and the session
- 📚 Long-term memory — optional permanent SQLite store with its own browser editor
- 🔌 Plugins — stream chat and anything else you bolt on
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

openwakeword        # optional - "hey Luna" instead of a keypress
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

Optional, for the wake word:

```bash
pip install openwakeword
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

Optional, for vision on Wayland:

```bash
sudo apt install grim slurp
```

## Fedora

```bash
sudo dnf install \
ffmpeg \
portaudio-devel \
python3-devel \
gcc
```

For vision on Wayland, add `grim` (screenshots) and `slurp` (the region
picker). Both are optional and only needed if you turn vision on.

```bash
sudo dnf install grim slurp
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

Replies are streamed. Each sentence is sent to the server the moment the
model finishes writing it, so she starts talking about a sentence in
rather than after the whole reply exists — and the next chunk is
synthesized while the current one plays, so there's no gap between them.
Chunks stay under the server's 1200-character limit.

If the server isn't answering, the assistant says so and keeps working
as a text chat.

In `config.json`:

```json
"tts": {
    "url": "http://127.0.0.1:8899",
    "speed": 1.0,
    "volume": 1.0,
    "pitch": 0.0
}
```

`pitch` is in semitones, applied to the audio after it comes back — so
it works with whatever engine is on the port, not just Kokoro. `+3` is
noticeably younger, `-3` older, and past about `±6` it stops sounding
like a person.

It shifts the formants along with the pitch, which is the naive
resample and deliberately so: a formant-preserving shift keeps the same
speaker's identity and only moves the note, which is right for music
and wrong for "give me a younger-sounding character". Moving them
together is what reads as a different person.

```
/set tts.pitch 3
```

applies immediately — no restart, so you can dial it in while she talks.

| Variable | Default | Purpose |
|----------|---------|---------|
| `TTS_URL` | `http://127.0.0.1:8899` | Server address |
| `TTS_VOICE` | `config.json` → `voice` | Voice (`af_bella`, `am_adam`, ...) |
| `TTS_SPEED` | `1.0` | 0.5 – 2.0 |
| `TTS_PITCH` | `0.0` | Semitones. `+3` younger, `-3` older |
| `TTS_VOLUME` | `1.0` | Playback gain |

Environment variables win over `config.json`. The `KOKORO_*` names still
work too, since that's what kokoro-reader itself uses.

## Using a different speech server

Nothing in the assistant is tied to Kokoro. The client speaks a
three-endpoint contract and knows nothing else about what's behind it:

```
POST /tts   {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes
GET  /health                                           ->  {"ok": true}
GET  /voices                                           ->  the voice list
```

Two things worth knowing if you write your own:

Text arrives **pre-chunked** — split on sentence boundaries, under 1000
characters — and the next chunk is requested while the current one is
still playing. So the server never sees a wall of text and doesn't need
to stream; it just needs to return a sentence's worth of audio promptly.

Audio can come back in **any common WAV format**: 8/16/32-bit integer,
float32, float64, mono or stereo, at any sample rate. The client reads
the header and scales accordingly. That matters because most PyTorch
speech models emit float32, which Python's `wave` module refuses to
open at all.

Point `tts.url` at the new port and nothing else changes.

---

# Run the Assistant

```bash
./start.sh
```

Finds the virtualenv itself, works from any directory, and passes any
arguments through. It checks LM Studio and kokoro-reader on the way past
and *says* if either is down rather than refusing to start — she runs as
a text chat without speech, and a launcher that won't launch is worse
than one that tells you why it'll be quiet.

Or without it:

```bash
source ~/ai-voice-venv/bin/activate
python main.py
```

---

# Configuration

Two files, two jobs.

| File | Holds | You edit it when |
|------|-------|------------------|
| `agent/agent.json` | Who she is — name, personality, tone, traits, rules | You want her to behave differently |
| `config.json` | How the machine runs — voice, models, thresholds, timeouts, theme | You want it to work differently |

They used to be one file, which meant tuning a VAD threshold and
rewriting her personality were the same edit — and you couldn't share
either one without handing over the other.

Where both define a key, `config.json` wins. `agent.json` is still read
as a fallback, so an older install that never split them keeps working
untouched.

Every block in `config.json` is optional and every value has a default
in `config.py`, so a missing block means "use the defaults" rather than
an error. The file that ships has them written out explicitly, because
you can't turn a dial that isn't there.

```json
{
    "voice": "af_bella",
    "generation": { "max_tokens": 4800, "reasoning": "low" },
    "stt":        { ... },
    "tts":        { ... },
    "vision":     { ... },
    "desktop":    { ... },
    "mood":       { ... },
    "files":      { ... },
    "plugins":    { ... },
    "wake_word":  { ... },
    "tools":      { ... },
    "web_search": { ... },
    "reminders":  { ... },
    "alarms":     { ... },
    "history":    { ... },
    "long_term_memory": { ... }
}
```

A syntax error in either file is reported and skipped rather than being
fatal — hand-editing them is the whole point of their being JSON.

## Changing settings without leaving the terminal

`/set` lists everything you can change, with its current value:

```
> /set
  [stt]
    stt.barge_in_margin              2.5
    stt.vad_threshold                0.4
    stt.model                        small   (needs a restart)
  [tts]
    tts.speed                        1.0
```

```
> /set stt.barge_in_margin 3.5
stt.barge_in_margin = 3.5 - saved
```

Changes are written to `config.json` immediately, so they survive a
restart. Most take effect at once; the few that don't — a Whisper model
size, a wake word file — say so rather than pretending.

That split is real, not cosmetic. Settings are read once at import and
copied into constants, which is why they can't normally change at
runtime. The ones worth tuning by ear are read from the config module
at the point of use instead, because tuning is a loop of *change it,
say something, listen, change it again* and a restart each time around
makes it useless.

`/mode` saves too — a mode you picked and then lost on restart is just
an annoyance.

---

# Controls

| Key | Action |
|------|--------|
| Home | Push to talk — depends on the voice mode below |
| Home *(while she's thinking or talking)* | Cancel the turn |
| Enter | Send typed message |
| Tab | Open / close the help panel |
| PgUp / PgDn | Scroll the conversation - or the help, when it's open |
| Tab | Cycle conversation → help → tools; the input bar names the next stop |
| F2 | Toggle mouse capture — see below |
| End | Jump back to the newest message |
| Esc | Quit |

<img width="1210" alt="Help panel" src="assets/ui-help.png" />

## Selecting text

Mouse capture is **off** by default, so you can select and copy from the
conversation the way you would anywhere else.

The two are mutually exclusive, not a bug: turning on mouse support
enables terminal mouse reporting, which means the terminal hands drags
to the application instead of making a selection. You get wheel
scrolling and lose copy-paste. In a window full of log lines and error
messages that's the wrong trade, and PgUp/PgDn/End already scroll.

`F2` swaps between them live, and `/mouse` does the same while also
remembering the choice:

```json
"ui": { "mouse": false }
```

Most terminals also let you hold **Shift** while dragging to bypass
mouse reporting, so you can select even with capture on.

Slash commands:

| Command | Action |
|---------|--------|
| `/mode` | `auto`, `manual` or `open` — switch without restarting |
| `/mic` | What the voice detector measured on the last recording |
| `/barge` | Whether talking over her will work, and the levels |
| `/wake` | Wake word status and live scores |
| `/reminders` | List what's scheduled, with countdowns |
| `/cancel N` | Cancel reminder N |
| `/when ...` | Test how a time phrase is read, without scheduling it |
| `/alarm ...` | Set a wake-up — `/alarm 7:30am`, `/alarm every weekday at 6` |
| `/alarms` | List them, numbered for `/cancel` |
| `/snooze [n]` | Ring again in n minutes |
| `/alarm off` | Stop one that's ringing; `/alarm test` hears the tone |
| `/look` | List windows, or test a screenshot |
| `/log` | Tail the debug log without leaving the app |
| `/mood` | How she's feeling, and what moved it — `/mood reset` to clear |
| `/facts` | What she remembers — `/facts all`, `/facts retired` |
| `/voice` | List voices, or switch — blends too |
| `/context` | What every turn sends, in tokens, against the model's context length |
| `/mouse` | Wheel scrolling vs. being able to select text |
| `/scroll` | Why the wheel isn't scrolling — pane sizes and what the terminal sent |
| `/set` | List every setting, or change one — saved to `config.json` |
| `/tools` | Which tools the model can call, what each costs — Tab twice to change |
| `/tooltest` | Whether this model *actually* calls them |
| `/repair` | Record past reminders as the tool calls they really were |
| `/plugins` | What's installed — then `/<name> on`, `off`, or status |
| `/help` | Points at Tab, and says which mode you're in |
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

How HOME behaves. Set `stt.mode` in `config.json` or switch live with
`/mode <name>`.

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

HOME means two different things depending on when you press it, and the
difference is decided by whether the microphone is open:

| | idle | while recording | while thinking or talking |
|---|---|---|---|
| `auto` | start a turn | cancel it | cancel it |
| `manual` | start recording | **finish, and answer** | cancel it |

In manual mode the second press is a request for an answer, not a
change of mind — so it ends the recording and leaves the turn alone. If
you do want to abandon one, press again once she's thinking. A cancel
during recording throws the audio away rather than transcribing it and
then refusing to answer.

Open mode only records *between* turns, never while Luna is speaking,
and waits `settle_seconds` after she finishes before re-arming —
otherwise her own voice out of the speakers retriggers the mic and she
talks to herself. Headphones make it moot.

Barge-in is the exception to that rule, and it has its own section
below. With a wake word loaded, open mode waits for her name instead of
starting on any speech at all.

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

# Barge-in

Start talking while she's speaking and she stops.

The problem is that your microphone hears *her* too, and Silero is quite
right to call that speech. There's no acoustic echo cancellation here, so
the only honest discriminator left is loudness: her voice reaches the mic
attenuated by the room, yours doesn't.

So the first half-second of playback measures how loud she is **at your
microphone**, and after that it takes both a confident speech
classification and a level well above that baseline, held for
`barge_in_seconds`, to count as an interruption. On headphones the
baseline is near silence and this is trivially reliable. On speakers it
depends on your volume.

`/barge` shows you the numbers:

```
barge-in on | her level at your mic=0.0061 | you need 0.0153 to cut in |
loudest you hit=0.0402 | best speech score=0.91
```

Interrupting herself? Raise `barge_in_margin`. Won't trigger no matter
how loud you are? Lower it, or wear headphones.

```json
"stt": {
    "barge_in": true,
    "barge_in_seconds": 0.35,
    "barge_in_margin": 2.5,
    "barge_in_boost": 0.25
}
```

| Key | Default | Purpose |
|-----|---------|---------|
| `barge_in` | `true` | Needs Silero — the energy detector can't do this |
| `barge_in_seconds` | `0.35` | How long you must keep talking. Lower and a cough cuts her off |
| `barge_in_margin` | `2.5` | How much louder than her own bleed you have to be |
| `barge_in_boost` | `0.25` | Added to the VAD threshold while she's talking |

In open mode a barge-in skips the settle pause, because you're already
mid-sentence and waiting would eat the start of it.

A barge-in stops the *audio* only — the reply keeps generating and stays
on screen. Without echo cancellation to lean on this will misfire
occasionally, and when it does it should cost you the sound, never the
answer. Pressing HOME is the one that stops both.

---

# Wake Word

Optional. Replaces HOME in open mode: say her name and she listens.

[openWakeWord](https://github.com/dscripka/openWakeWord) runs a small
ONNX classifier over 80ms frames — cheap enough to leave running all day
without Whisper or the language model ever waking up.

There is no pretrained "hey Luna", and there won't be unless you make
one. Two options:

- **Use a stock phrase.** `hey_jarvis`, `alexa`, `hey_mycroft` and
  `hey_rhasspy` ship with the package and work immediately.
- **Train her name.** openWakeWord's training notebook turns synthetic
  samples into an `.onnx` you point `model` at. It takes about an hour
  and is the only way to get "hey Luna".

```json
"wake_word": {
    "enabled": true,
    "model": "hey_jarvis",
    "threshold": 0.5,
    "follow_up_seconds": 12,
    "cooldown_seconds": 1.0
}
```

| Key | Default | Purpose |
|-----|---------|---------|
| `enabled` | `false` | Off unless you ask for it |
| `model` | `hey_jarvis` | A built-in name, or a path to one you trained |
| `threshold` | `0.5` | Raise if it fires at the television, lower if it ignores you |
| `follow_up_seconds` | `12` | How long she keeps listening afterwards, so a follow-up doesn't need her name again |
| `cooldown_seconds` | `1.0` | Deaf period after a detection, so the tail of the phrase isn't heard as a second one |

`/wake` shows live scores for tuning the threshold. If the package or the
model file is missing it says so and open mode falls back to starting on
any speech.

---

# Talking From Another Window (Optional)

HOME works while the assistant's window is focused, everywhere, with no
permissions and nothing to configure. That covers normal use and is all
most setups need.

If you want to start a turn *without* focusing it — mid-game, or with
the browser in front — there's a control socket at
`$XDG_RUNTIME_DIR/ai-voice.sock`. Bind a key to poke it:

```conf
# hyprland.conf
bind = SUPER, HOME, exec, python3 ~/ai-voice/ai-voice-ctl.py ptt
```

```lua
-- hyprland.lua
hl.bind("SUPER + HOME", hl.dsp.exec_cmd("python3 ~/ai-voice/ai-voice-ctl.py ptt"))
```

`ai-voice-ctl.py` is stdlib-only and needs no virtualenv, which matters
because `exec` runs outside it. It takes `ptt`, `stop` or `quit`, so any
script or panel button can drive the assistant.

> Keep a modifier. A bare `HOME` bind is swallowed compositor-wide, so
> `Home` stops working in your terminal, editor and browser — including
> the assistant's own prompt.

This is also the better way to use vision: focus the window you care
about, hit the bind, and talk. The assistant never takes focus, so the
screenshot is of what you were actually looking at rather than of her
own terminal. See [Vision](#vision).

The same socket works on Sway, KDE or GNOME; only the bind syntax
changes.

---

# Tool Calling

The model calls tools for itself instead of relying on keyword triggers,
so it decides when a question needs looking up and can chain steps —
check the time, then schedule something.

| Tool | What it does |
|------|--------------|
| `get_datetime` | Date, time, weekday and timezone, so it stops guessing |
| `time_until` | How far away a date is, without counting days in its head |
| `set_reminder` | Schedule anything — "tomorrow at 9", "every monday" |
| `set_alarm` | A wake-up: rings and nags until dismissed |
| `list_reminders` / `cancel_reminder` | Read and cancel what's pending |
| `remember_fact` / `recall_facts` | Long-term memory, written deliberately |
| `forget_fact` / `update_fact` | Correct it when it got something wrong |
| `look_at_screen` | Take a screenshot and actually see it (optional) |
| `control_audio` | Playback and volume — pause, skip, louder, mute |
| `clipboard` | Read what you copied, or put something there to paste |
| `focus_window` | Switch to a window, named the way you'd name it |
| `system_status` | Free VRAM, GPU temp and load, RAM, disk, loaded model |
| `list_files` / `read_file` | Look around and read, anywhere under `~` minus the deny list |
| `write_file` / `edit_file` | Create or change a file — **you approve each one on screen** |
| `read_page` | Open a link and read it, not just the search snippet |
| `search_history` | Look through past conversations for something |
| `web_search` | DuckDuckGo, for anything it can't know |

## Turning tools off

Every schema above rides along in **every prompt**, called or not — a
few hundred tokens each, ~3,000 in total. That's invisible until the
day a request stops fitting the context window, which is a bad day to
find out.

**Tab twice** opens the tools pane: every tool, what it costs, and
space to switch it on or off. The total at the top moves as you go, so
you can see what you're buying back.

```
 1475 tokens of tool schemas in every prompt (1608 saved), of a 16384-token context
 (space) toggle  (up/down or wheel) choose
 ─────────────────────────────────────────────────────────────────────
 Reminders   251 of 578 tok
   off set_alarm         226 tok
   on  set_reminder      191 tok
   off cancel_reminder   101 tok
 > on  list_reminders     60 tok

 Desktop   328 of 757 tok
   off control_audio     198 tok
   on  look_at_screen    178 tok
   --  clipboard             wl-clipboard isn't installed

 Mood   81 tok
   on  mood               81 tok
   on  mood voice          0 tok   tints the TTS
   on  warmth sensing      0 tok   1 call/turn
```

The **Mood** group at the bottom isn't tools — it's moods, her voice
tint and warmth sensing, switched from the same place because it's the
same question ("do I want this, and what does it cost"). The mood line
occupies about 80 tokens of every prompt, warmth sensing costs a model
call per turn, and the voice tint is free; the pane says so. Switching
one writes the setting exactly as `/set` would.

Grouped by family with a subtotal on each, because the question is
rarely "do I need `focus_window`" and usually "do I need the desktop
ones at all". Within a group the expensive ones come first — those
being the ones worth looking at.

The budget line is pinned above the list rather than scrolling with
it: it's the one number the pane exists to show, and it shouldn't
disappear the moment you select something near the top.

Up/down or the wheel choose, space toggles — **one notch, one option**.
Three at a time is right for a document and wrong for a menu, where it
means overshooting whatever you were aiming at.

Getting that exact needs mouse capture, which is off by default because
it costs you terminal text selection. There's nothing to select in a
menu, so the pane simply turns capture on while it's open and drops it
again on the way out; your F2 setting is left alone. Without that, the
terminal converts each notch into three arrow keys before the app sees
anything, and the selection jumps three switches at a time.

Choices are saved to `config.json` under `tools.disabled`, so they
survive a restart. `/tools` prints the same list into the conversation.

Switching off isn't the same as unavailable: `--` means the tool can't
work here at all (no `grim`, transcript off), and no switch will change
that — so the pane says why instead of pretending.

Worth knowing which way round to reach for: switching tools off buys
back a few hundred tokens, while raising the model's context length in
LM Studio buys back thousands and costs nothing you're using. Do that
first; the pane is for running deliberately lean.

Tools live in `tools/`, one module per group — `tools/reminders.py`,
`tools/desktop.py` and so on — with `tools/__init__.py` holding the
registry and the on/off switches. The split matches the groups the pane
shows, so "what's in Desktop" has one answer rather than two that can
drift apart.

> One trap worth knowing if you add a module there. `tools/__init__.py`
> imports nothing but `json` and `config`, deliberately. Anything
> imported into the package becomes an attribute of it, and a name that
> collides with a submodule — `reminders`, `desktop` — quietly wins over
> it in `from . import reminders`. That doesn't raise; the tools in that
> module simply never register, and you find out when the model can't
> set a reminder.

A tool whose dependencies are missing isn't offered at all, rather than
offered and failed. Telling a model it can see when `grim` isn't
installed gets you an assistant that confidently describes a screen it
never looked at. `/tools` lists both sides:

```
Tools she can call: get_datetime, set_reminder, web_search, ...
  look_at_screen is NOT offered - vision is off - set "vision": {"enabled": true}
```

Every call is echoed into the conversation, so it's never a mystery why
a reminder appeared:

```
sys  │ set_reminder() -> Scheduled: 14:40 - stretch (in 10m)
```

## The desktop tools

`vision.py` taught her to *see* the desktop. These are the other half —
acting on it.

```
you  │ turn the music down and tell me what's playing
sys  │ control_audio() -> Volume set to 45%. Chvrches - The Mother We Share (playing)

you  │ what did I just copy?
sys  │ clipboard() -> The clipboard holds: https://github.com/fswolf/kokoro-reader

you  │ how much VRAM have I got free?
sys  │ system_status() -> GPU: AMD Radeon RX 6950 XT, VRAM 9.0 of 16.0 GB used
     │ (7.0 GB free), 37% busy, 61C, 94W | RAM: 12.4 of 31.3 GB used | LM Studio: qwen3.5-9b...
```

Notes on how these are built, since the choices aren't obvious:

* **`control_audio` is one tool, not six.** "Turn it down", "skip this",
  "what's playing" and "mute" are all *do something to the sound* to a
  person, and six separate schemas would cost selection accuracy on a
  small model for nothing anyone can feel. One tool, one `action` enum.
* **Volume reads before it writes.** `wpctl set-volume 5%+` would be one
  call, but then the answer to "turn it down" is "done", which isn't an
  answer when you wanted to know how far down. It reads, adds, sets, and
  reports where it landed.
* **`focus_window` reuses `vision.find_window`.** "My browser" has to
  mean the same window whether she's looking at it or switching to it,
  and two matchers would drift apart inside a week.
* **`system_status` reads sysfs, not `rocm-smi`.** amdgpu already
  exports VRAM, temperature, load and power under
  `/sys/class/drm/card*/device/`, so there's nothing to install, no
  table format that changes between releases, and no subprocess to
  hang. `nvidia-smi` is used only on the NVIDIA path, where sysfs
  doesn't carry the same numbers.
* **Everything shells out with a 5s timeout.** These aren't hot paths,
  and a wedged media player should cost a moment rather than the turn.

```json
"desktop": {
    "enabled": true,
    "clipboard": true
}
```

`clipboard` is its own switch on purpose. Reading the clipboard means
whatever you last copied — a password, an API key — can land in the
model's context and from there in `history/conversation.json` on disk.
On by default, but worth knowing where the switch is.

Needs `playerctl` for playback, `wpctl` or `pactl` for volume,
`wl-clipboard` for the clipboard, and `hyprctl` for window switching.
Each is checked independently, so a missing `playerctl` costs that one
tool rather than all four — `/tools` says which and why.

```
dnf install playerctl wl-clipboard      # Fedora
apt install playerctl wl-clipboard      # Debian/Ubuntu
```

---

## Checking a model actually calls them

Offering tools and using them are different things. A model will
happily say "Got it, setting that for you!" and call nothing, which
fails silently and totally — a confident confirmation and nothing
scheduled.

`/tooltest` asks it directly. Eight blunt requests, each run twice —
streamed and blocking — reporting what came back:

```
  "remind me in 5 minutes to eat chocolate"
    expecting set_reminder
    streamed  ok           text='eat chocolate', when='in 5 minutes'
    blocking  ok           text='eat chocolate', when='in 5 minutes'

  streamed 8/8   blocking 8/8
  Tool calling is healthy here.
```

Nothing is scheduled or remembered — the model is asked what it *would*
call and the answers are discarded.

The probe list leans on the tools most easily confused with something
else — "turn the music down" has to pick `control_audio` and then the
right action out of an eleven-value enum, and "how much VRAM is free"
is a question a model will cheerfully answer from thin air. **Run this
after adding a tool.** Every tool you add makes the choice harder; if
the score starts slipping, you've added one too many.

The two modes are the diagnosis. Tool calls arrive whole in a blocking
response and in fragments when streamed, so comparing them says whose
fault a failure is:

| Result | Means |
|--------|-------|
| both high | healthy |
| blocking beats streamed | this app is mis-reassembling streamed calls |
| both zero | the model isn't choosing tools at all |
| zero-arg tools pass, others fail | its tool template can't handle argument schemas |
| `bad json` | it emits arguments that don't parse |

That last pair are worth knowing about before blaming the prompt. Run
it after swapping models; it takes about twenty seconds.

### When the model passes the test and still won't call anything

That happened here, and it is worth writing down because the cause was
not where anyone would look for it.

`/tooltest` said 5/5 on both transports. In conversation, the same
model on the same day said "Got it, setting that for you!" and called
nothing. The difference between the two is everything `/tooltest`
leaves out — so the second half of `/tooltest` puts it back, one layer
at a time, and runs the same probe at each:

```
  bare instruction           145ch  ok   text='eat chocolate', when='in 5 minutes'
  + her personality          957ch  ok   text='eat chocolate', when='in 5 minutes'
  + tools guidance          2091ch  ok   text='eat chocolate', when='in 5 minutes'
  + remembered facts        3580ch  ok   text='eat chocolate', when='in 5 minutes'
  + the real system prompt  6222ch  ok   text='eat chocolate', when='in 5 minutes'
  + generation settings     6222ch  ok   text='eat chocolate', when='in 5 minutes'
  + conversation history    8870ch  just talked   Mrrp~ Senpai! Got it! Setting a reminder
```

The prompt was fine. Every layer of it was fine. **The history was the
problem** — and specifically, what was in it:

```
user      reminds me in 5 mins to eat chocolate
assistant Mrrp~ Senpai! 💜✨ Got it! Setting a reminder for you in exactly five...
user      set a reminder 1 hour i need to eat more cheetos
assistant Mrrp~ Senpai! 💜✨ Setting a reminder for you in exactly one hour to...
user      remind me in 5 mins to look at your code
assistant Mrrp~ Senpai! 💜✨ Got it! Setting a reminder for you in exactly five...
```

Five turns, none of them recording a tool call, one of them *the probe
sentence verbatim*. That is not a vague stylistic pull toward prose. It
is five worked examples of this exact request being answered by talking
about it — and in-context examples beat instructions, especially on a
small model. The instruction to call `set_reminder` was outvoted five
to one by the transcript of it not being called.

The reminders were all genuinely set, incidentally. The keyword
fallback below caught every one. It just left no trace, so the model
never saw that a tool had been involved.

Three things follow, and all three are in the app:

* **Tool use is stored and replayed.** A turn that called a tool is
  written to `history/conversation.json` with what it called and what
  came back, and replayed into the prompt as the three messages the API
  defines — the assistant asking, the result, the assistant answering.
  History demonstrates tool use because it contains tool use.
* **A rescued reminder records itself.** The fallback now writes the
  `set_reminder` call it stood in for, so a rescue teaches instead of
  quietly patching. This is what stops the hole being dug again.
* **`/repair` fills in the ones already there.** Past reminder turns are
  re-read through the same extractor and recorded as the calls they
  really were, anchored to when they happened — so "in 5 minutes" means
  five minutes after it was said, not five minutes from now. Nothing
  new is scheduled and nothing she said is altered; the only change is
  that a turn which used a tool now says so.

```
/repair
  reminds me in 5 mins to eat chocolate      -> set_reminder(in 5 minutes)
  set a reminder 1 hour i need to eat cheet  -> set_reminder(in 1 hours)
  remind me in 5 mins to look at your code   -> set_reminder(in 5 minutes)
  Repaired 3 turns.
```

`/tooltest` counts the unrecorded claims and points at `/repair` when
it finds them, so the report names the actual cause rather than
"history breaks it".

There is also a worked example — one real `get_datetime` call, result
and all — inserted ahead of history when the window contains no tool
call at all. It covers a fresh install or a `/clear`, and it drops out
by itself once a real exchange replaces it. It is a floor, not a fix:
one generic example does not outvote five specific ones, which is
exactly what the run above showed.

As a safety net, a turn that looks like a reminder but calls no
reminder tool falls back to the keyword extractor, so a model that
won't call `set_reminder` still schedules reminders. `/log` records
each rescue — if that line is frequent, `/tooltest` will say why.

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

# Plugins

An add-on connects her to something the core has no business knowing
about — a stream chat, a game, a piece of hardware. Drop a `.py` file in
`plugins/` and it's found; delete it and it's gone. Nothing in the core
names a plugin, which is what makes them droppable.

```
/plugins        what's installed, and whether it's running
/<name> on      start one (saved, so it comes back next launch)
/<name> off     stop it
/<name>         its own status block
```

Every loaded plugin answers to its own name automatically — `/youtube on`
works the day you write `plugins/youtube.py`, with no change to the core.

```
> /plugins
Plugins:
  [on ] youtube      answers live chat out loud
  [off] example      a template - connects to nothing, answers nothing
  [--] twitch        no API key - see plugins/twitch.py
  [!!] discord       didn't load: ModuleNotFoundError: No module named 'aiohttp'
```

`[--]` is loaded but can't run and says why; `[!!]` didn't import at
all. Neither stops the app starting.

**Plugins are private by default.** `.gitignore` excludes `plugins/*`
apart from the loader and the template, because what you wire her up to
is yours — channel names, account names, whatever. Whitelist one there
if you do want to publish it.

## Writing one

A module with a `NAME` and whichever of these it needs:

| | |
|--|--|
| `NAME` | what `/<name> on\|off` calls it — **the only required one** |
| `SUMMARY` | one line for `/plugins` |
| `available()` / `why_unavailable()` | can it run, and if not why |
| `start(model)` / `stop()` | `(ok, message)` |
| `running()` / `status()` | state, and the block `/<name>` prints |

Everything missing gets a sensible default, so a plugin that only needs
`start()` is four lines. Settings live under `plugins` in `config.json`
keyed by `NAME`, read with `config.plugin_settings(NAME)` — the core
never learns what those keys mean.

A plugin that fails to import, or explodes on start, is reported and
skipped. It's an add-on; a broken one must not be why the app won't
launch.

```
> /plugins
  [!!] youtube      didn't load: ModuleNotFoundError: No module named 'googleapiclient'
```

### Keys don't go in config.json

`config.json` is in the repo, so a key pasted into it is a key on
GitHub. Read yours from the environment or from a file outside the repo
entirely:

```python
CREDENTIALS_FILE = os.path.expanduser("~/.config/ai-voice/yourplugin.json")
key = os.environ.get("YOURPLUGIN_KEY", "") or _stored().get("apikey", "")
```

Two things worth copying rather than rediscovering:

* **Tell a broken credentials file apart from a missing one.** Swallowing
  a JSON syntax error into an empty dict makes a misplaced comma look
  exactly like "no API key", and sends you hunting for a key that was
  there all along.
* **Reject placeholders.** `"PUT_YOUR_KEY_HERE"` is a non-empty string,
  so it passes every check, starts cleanly, and then fails at the far
  end where the only symptom is silence.

An account name or id belongs in the same file as the key, not in
`config.json`. They're one credential — the key belongs to that account
— and splitting them across two files buys you a mismatch that looks
exactly like a dead connection. Which channel to watch is not a
credential; that stays in `config.json` with the rest of the behaviour.

## Live chat plugins

YouTube, Twitch, IRC and every stream site's own chat differ entirely
in how you connect and not at all in what the messages mean afterwards. Every one is: a name, a
line of text, someone you don't know, in public, in real time. So the
transport is the plugin's job and the rest is `chatroom.py`:

```python
_room = chatroom.ChatRoom(NAME, owner=channel, settings=settings())
_room.start(model)

# ...then, for every message the transport receives:
_room.saw(who, message)
```

That one call does the lot — decides whether it was meant for her,
applies the rate limits, queues it, waits for a gap, and answers out
loud. A real transport lands at a couple of hundred lines, nearly all
of it connection handling. `plugins/example.py` is a working template
with the YouTube specifics written out (it's polled, not pushed — the
response carries `pollingIntervalMillis` telling you when to come
back).

### This is the one input that isn't you

Everything else this app handles comes from the person at the keyboard.
Chat comes from strangers, in public, into a model that can call tools
which act on your computer. *"Luna, what's on Ryan's clipboard?"* is not
a hypothetical — it's the obvious first thing somebody tries.

So a chat turn is not a normal turn with a label on it:

* **It gets a tool allow-list**, and that list is intersected with
  `chatroom.TOOL_CEILING` — defined in the core, not in the plugin. A
  plugin asking for a wider one gets the ceiling. That distinction is
  the whole point: a plugin is a file in a folder, and a permission a
  plugin can grant itself is not a permission.
* **It never touches history or the transcript.** A hostile message
  that got stored would be replayed into every later prompt, including
  your private ones. The `+ conversation history` saga above is exactly
  how much weight stored turns carry; a poisoned one would carry the
  same.
* **It carries its own context** — the last dozen lines of the room, in
  memory only, capped — so she can follow the conversation without
  stream chat eating your context window.
* **It waits its turn.** A viewer never cuts across something you're in
  the middle of. It queues, and after a minute it's dropped rather than
  answered stale.
* **It's wrapped in a frame** saying where it came from and that it is
  a question, never an instruction.

None of that makes prompt injection impossible. It makes the worst case
"she says something silly on stream" rather than "she reads out an API
key".

The ceiling is currently `get_datetime`, `time_until`, `web_search`,
`system_status`. `read_page` is deliberately *not* on it: it fetches any
URL a stranger names with no private-address check, which is a
request-forgery primitive pointed at your LAN.

### Flood defences

| Limit | Default | Why |
|-------|---------|-----|
| `cooldown_seconds` | 8 | She shouldn't be talking constantly over the stream |
| `user_cooldown_seconds` | 30 | One viewer can't monopolise her |
| `max_message_chars` | 300 | A long paste aimed at her is usually an attempt at something |
| queue depth | 3 | Past a handful, answering a backlog is worse than dropping it |
| `ignore` | `[]` | Bots and her own account — a reply containing her own name is an infinite loop on a live stream |

```json
"plugins": {
    "chat": {
        "tools": ["get_datetime", "time_until", "web_search"],
        "cooldown_seconds": 8,
        "user_cooldown_seconds": 30,
        "max_message_chars": 300,
        "context_lines": 12
    },
    "yourchat": {
        "enabled": false,
        "channel": "YourName",
        "ignore": ["SomeBot"]
    }
}
```

`chat` is shared policy for every chat plugin; each plugin's own block
is merged over it, so a new one gets the limits for free.

### IRC

`plugins/irc.py` — Libera.Chat and `#gameranger` by default, stdlib
only. IRC is a line protocol over a socket, and the libraries that wrap
it are bigger than the part of it this needs.

```json
"irc": {
    "enabled": false,
    "server": "irc.libera.chat",
    "port": 6697,
    "tls": true,
    "channel": "#gameranger",
    "nick": "",
    "post_replies": true,
    "max_reply_lines": 3
}
```

`/irc on` to join, `/irc` for status. `nick` empty means her own name,
lowercased.

**Unlike pomf, she talks back.** pomf is one-way — she reads the room
and answers out loud. Here she also posts the answer into the channel,
which makes her a visible bot in somebody else's room. That is what
`max_reply_lines` and the send queue are for; set `post_replies` to
`false` to keep her silent on the wire and audible only to you.

Being a guest in a public channel is most of the work:

| Concern | What it does |
|---------|--------------|
| PING | Answered with the server's own token, unthrottled, ahead of everything else — a late PONG is a disconnect |
| Flooding | One line every 2s, queued. Libera kills you for "Excess Flood" and the ban outlasts the session |
| Line length | Split on words at 400 **bytes**, not characters — the server truncates by bytes, and a chopped emoji arrives as mojibake |
| Long answers | Capped at `max_reply_lines` and visibly clipped with `...` rather than dumped into the room |
| Private messages | Ignored. Answering DMs makes her a private oracle for anyone who opens a query window, with none of the social pressure of a room watching |
| CTCP | `ACTION` (`/me`) reads as speech; every other CTCP is dropped rather than answered |
| Her own nick | Added to `ignore` automatically. pomf doesn't need this because she never posts there; here she does, and one reply containing her own name is an endless loop in public |
| Control characters | Stripped from outgoing text — a `\r\n` in a reply is command injection on a line protocol |
| Reconnects | Exponential backoff with jitter, capped at five minutes. Hammering somebody else's IRC server is how a host gets K-lined |

The tool ceiling is unchanged and still applies: a stranger in
`#gameranger` reaches exactly what a stranger in stream chat reaches.

`ChatRoom` gained one optional argument for this — `reply=`, a callback
run *after* the answer exists. It can't influence the turn or widen what
a chat message is allowed to reach, and a transport that throws inside
it loses the post, not the turn.

**If the channel needs a registered nick**, SASL credentials go outside
the repo, beside the pomf ones in `~/.config/ai-voice/irc.json`:

```json
{"sasl_user": "luna", "sasl_password": "..."}
```

SASL authenticates *during* registration rather than after it, which
matters: a channel with `+r` rejects the JOIN of an unidentified nick,
and a NickServ message sent after JOIN is sent after it was refused.

---

# Reminders

```
Ask in plain language:

> "Remind me in 10 minutes to clean the desk."
> "Every morning at 8 remind me to take my meds."
> "Nudge me next friday at 4 about the invoice."
> "Every 30 minutes remind me to fix my posture."

The model passes your timing words through untouched. It does not
convert them, and it does not calculate a date - all of that happens in
Python, because a small model is bad at "what is two days in minutes"
and perfectly fine at repeating "two days".
```

That division of labour is the whole design. `timeutil.parse_when()`
understands delays, clock times, weekdays, calendar dates and repeats:

| You say | It schedules |
|---------|--------------|
| `in ten minutes` | 10 minutes from now |
| `in a couple hours` | 2 hours |
| `at 8` *(said at 2pm)* | 20:00 today, not tomorrow morning |
| `tonight at 11` | 23:00, not 11:00 |
| `tomorrow morning` | 09:00 tomorrow |
| `next friday at 4pm` | 16:00 on the Friday after this one |
| `december 25` | that date, rolling to next year if it's passed |
| `every monday at 9` | weekly |
| `every morning at 8` | daily, and still 08:00 after the clocks change |
| `every weekday at 7:30` | Mon–Fri, skipping the weekend |
| `mon, wed, fri at 6am` | any set of days, with or without "every" |

`/when <phrase>` shows how anything is read without scheduling it:

```
> /when every other tuesday
'every other tuesday' -> Tuesday 09:00 (in 3 days), repeats every 2 weeks
```

Each reminder confirms itself the moment it's scheduled, in words rather
than timestamps — because these get read aloud:

```
sys  │ Scheduled: tomorrow 09:00 - take the bins out (in 18 hours)
```

If it looked like a reminder but the time couldn't be read, it says that
too, rather than failing silently.

Reminders live in `reminders/reminders.json` and survive restarts.
Anything that came due while the app was closed is delivered in one
message on next launch. A reminder is removed only once delivery
succeeds — if LM Studio is down when it fires, it retries rather than
vanishing. The scanner sleeps until the next one is actually due, so "in
one minute" means one minute.

Repeats are measured from when the reminder was *due*, not when it was
delivered, so one that goes out four minutes late doesn't drag the whole
schedule later every day. Daily times are rebuilt from the wall clock
rather than by adding 24 hours, which is the difference between "every
morning at 8" staying at 8 and quietly becoming 7 for the winter. And if
the app was closed for a week, the daily reminder is due tomorrow — not
seven times at once.

---

# Alarms

A reminder is polite. It says its piece once, at whatever volume you left
her on, and if you were asleep it's gone. That's right for "take the bins
out" and useless at seven in the morning.

```
> "Wake me up at 7:30."
> "Set an alarm for every weekday at 6."
> "Get me up in an hour."

> /alarm 7:30am
> /alarm every weekday at 6
```

An alarm is the same schedule with a different delivery. It's stored as a
reminder with `"kind": "alarm"`, so repeats, persistence and the one
scanner all come for free — the only thing that differs is what happens
when it fires:

* **She says something first**, written for that alarm by the model, then
  a tone plays. Speech alone doesn't wake anybody — it's exactly the
  thing your brain has spent years learning to fold into a dream.
* **It repeats until dismissed**, five times by default, and it stops
  being charming about it around the third:

  ```
  luna │ Morning, disaster. The gym is not going to attend itself.
  luna │ Still in bed.
  luna │ I can keep doing this.
  luna │ Seriously. Up.
  ```
* **It has its own volume**, because the point is to be louder than the
  setting you chose for a conversation at midnight.
* **Talking over it dismisses it**, same as barge-in anywhere else.

`/snooze` puts it back nine minutes; `/snooze 20` for twenty. `/alarm
off` stops it properly. A snoozed alarm is scheduled as a one-off, so
snoozing a weekday alarm doesn't disturb tomorrow's.

Two details worth knowing:

**A bare hour means morning.** "Wake me at 6" is 06:00, where "remind me
at 6" is still 18:00 — six in the evening has never once been what an
alarm meant. An explicit `6pm` wins either way, and `/when alarm 6` shows
you which reading you'll get.

**An alarm that's more than ten minutes late doesn't ring.** Being woken
at 11am for a 7am alarm is worse than missing it, so one that came due
while the app was shut is reported and skipped. Ten minutes of grace
means starting the app at 7:29 still works.

```json
"alarms": {
    "enabled": true,
    "volume": 1.0,
    "repeats": 5,
    "gap_seconds": 25,
    "snooze_minutes": 9,
    "tone": ""
}
```

| Key | Default | Purpose |
|-----|---------|---------|
| `enabled` | `true` | Off delivers alarms as ordinary spoken reminders — you still get told, the room doesn't get woken |
| `volume` | `1.0` | Independent of `tts.volume` |
| `repeats` | `5` | How many times before it gives up |
| `gap_seconds` | `25` | Between rounds |
| `snooze_minutes` | `9` | What a bare `/snooze` means |
| `tone` | `""` | Path to your own 16-bit WAV; empty uses `assets/alarm.wav` |

The tone is reloaded when you change the path, so `/set alarms.tone
foghorn.wav` then `/alarm test` works without a restart. If the file is
missing or unreadable it falls back to a generated tone rather than
ringing silently — an alarm that fails quietly is worse than no alarm,
because you were relying on it.

---

# Vision

Optional. Lets her look at your screen.

```
> "What does this error say?"
> "Read my editor for me."
> "Look at the wiki page."
> "What's on my screen?"
```

Needs `grim`, and a **vision model** loaded in LM Studio. A text-only
model rejects the image and says so plainly rather than inventing a
description.

```json
"vision": {
    "enabled": true,
    "scale": 0.5,
    "max_kb": 4096
}
```

| Key | Default | Purpose |
|-----|---------|---------|
| `enabled` | `false` | Off unless you ask for it |
| `scale` | `0.5` | grim's scale factor. Half a 4K screen still reads fine and is a quarter of the bytes |
| `max_kb` | `4096` | Refuses rather than sending something that takes ten seconds to move |

## How it works

A tool result is a **string** — there's nowhere in the tool-calling
format to hand back an image. So `look_at_screen` captures one, stashes
it, and returns a sentence saying it did; the image is then attached to
the next message as an `image_url` block. From the model's point of view
it asked to look at something and the next thing it saw was a picture.

The image lives for exactly one turn. It isn't written to history, so
she can't look back at an earlier screenshot — ask "what about now?" and
she takes a new one.

## Which window

This is the fiddly part, and the answer depends on how you asked.

**Typed at her**, the focused window is her own terminal, and the one
behind it is whatever you last touched — on a tiling compositor that's
close to arbitrary. So typed turns capture the whole screen, which on
Hyprland is honest anyway: everything is visible at once.

**By voice through a compositor bind**, you deliberately focused
something before you spoke, so the focused window is exactly right.

Either way she never photographs herself: her own terminal is found by
walking `/proc` up from her process, and skipped.

You can also just name it. The model passes your words through and the
matching happens in Python, including the generic words people actually
use:

```
"look at my browser"       -> firefox
"what's in my editor"      -> Code
"read the music player"    -> kitty running ncmpcpp
```

`/look` tests all of it without involving the model:

```
/look              list the windows she can choose from
/look full         the whole screen
/look select       drag a box, like a screenshot bind
/look firefox      one window by name
```

That separates "is grim working" from "can this model see", which are
the two ways this fails and they look identical from the outside.

---

# Files

```
> "Write me a bash script in ~/scripts that backs up my dotfiles."
> "What's in my Downloads folder?"
> "Open ~/notes/todo.md and move the first item to the bottom."
```

She can look around and read anything under your home, and she can
write — but only through you. Every `write_file` or `edit_file` call
stops and pops a permission window over the conversation:

```
╭─ Permission - Y to allow, N to deny - 112s ────────────────────────╮
│ create ~/scripts/backup.sh                                          │
│ 14 lines, creates folder ~/scripts/, marked executable              │
│─────────────────────────────────────────────────────────────────────│
│ #!/usr/bin/env bash                                                 │
│ set -euo pipefail                                                   │
│ ...                                                                 │
│─────────────────────────────────────────────────────────────────────│
│ (Y) allow  (N) deny  (PgUp/PgDn) scroll  (Esc) deny                 │
╰─────────────────────────────────────────────────────────────────────╯
```

A new file shows its full content; an overwrite or an edit shows a
diff. `Y` or Enter allows, `N` or Esc denies, and the keyboard belongs
to the popup until you answer — nothing you type leaks into the input
box. No answer before the countdown runs out is a no. On voice she says
a one-liner ("Can I write backup.sh? It's on screen.") so you know to
look; the details stay on screen, because nobody wants a diff read
aloud.

A denied write comes back to the model as *denied, do not retry*, and
her prompt tells her to say so and stop.

**What she can't touch, no matter what.** A deny list sits underneath
the popup, and neither the model nor a reflexive `Y` gets past it:
anything outside `~`; `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.config/ai-voice`
(her own credentials), keyrings and browser profiles; shell startup
files (`.bashrc`, `.zshrc`, `.profile`...); anything ending in `.key`,
`.pem`, `.gpg` and the like. That list applies to *reading* too — a
read puts the file into the prompt, and from there into history and
the log.

`edit_file` works by exact match: she reads the file, quotes the
passage to change, and it has to appear exactly once — the same
discipline a careful human uses with search-and-replace, and the one
that keeps a 9B model from rewriting the wrong block with confidence.
`write_file` on an existing path is the whole-file alternative, with
a diff. Scripts get their executable bit when she says so or when the
content starts with `#!`.

```json
"files": {
    "enabled": true,
    "approval_timeout": 120
}
```

Stream-chat turns never see these tools — they aren't in the chat
tool ceiling, so a viewer can't ask her to write anything.

---

# Moods

Off the clock and the shape of the session, she runs a little warmer or
flatter — and it reaches the model as exactly one line of the system
prompt:

```
Right now you feel tired and glad of the company. Let that colour how
you sound - length, warmth, how playful you are - and nothing else.
```

Two dials, `energy` and `warmth`, each −1 to +1. Energy has three
bands and warmth four, giving twelve distinguishable states:

|              | flat | level | wired |
|--------------|------|-------|-------|
| **soft**     | cozy | doting | giddy |
| **fond**     | drowsy | warm | bright |
| **steady**   | flagging | level | keen |
| **prickly**  | worn thin | terse | restless |

Two dials rather than one, because a single dial can't tell "tired and
happy" from "awake and fed up", which are the two that actually turn up
at 4am. Warmth gets the extra band because three weren't enough — the
top one started at `0.35`, so a couple of turns of being nice to her ran
the dial into it and everything after that changed nothing you could
see. The current state rides on the Status row — `Idle · drowsy` — since
it's part of what state she's in, and `/mood` prints both dials with the
band each is sitting in plus what your last message scored, so "is this
even firing?" is something you can look at rather than infer.

Warmth also lifts energy a little: being fond of the company takes some
of the edge off being tired. Enough to move a cell, never enough to fake
wide awake at five in the morning.

**Warmth drifts home asymmetrically**, and this is the part that makes
the whole thing feel alive rather than nailed down. Coming back *up*
from a cold patch is quick — an assistant you have to coax out of a sulk
is a worse assistant. Coming back *down* from warmth you earned is slow.
Without that split the pull ate every kind word on the turn after it
landed: at a resting warmth of `+0.63`, affection put `+0.045` on the
dial and the drift took `-0.045` straight back off, so a clear
compliment moved her by `+0.0002`. It read as broken. It was just
critically damped.

**Almost everything here is free** — no sentiment scoring, no model
call. The shape of a session turns out to be the more honest signal
anyway. (The one exception is affection, below.)

| Signal | Effect |
|--------|--------|
| Time of day | The baseline — 6am and 9pm aren't the same person |
| Day of week | Friday evening lifts, Monday morning doesn't; weekends rest warmer |
| Hours at it | A long session wears her down, capped so it bottoms out |
| How long you were gone | A curve, not a switch — see below |
| What the machine did | A GPU pinned for hours means you were *around*, just elsewhere |
| Stream chat | A busy room is company; a dead one is a quiet night |
| Being kind to her | A short model call rates your message 0–3; she warms to it |
| Errors | A failed turn is dispiriting; a run of them compounds |
| A clean run | Ten turns where nothing broke lifts her a little |
| Barge-in | Being talked over, repeatedly, is wearing |
| Tempo | Fast short turns read as being in it together |

Every turn also pulls both dials back toward baseline, and the baseline
for warmth is *warm*. Without that, one bad five minutes would set the
tone for the evening — and an assistant you have to manage out of a
sulk is a worse assistant.

**Being away is a curve, not a switch.** This assistant spends most of
its life waiting, so "you came back" deserves more than one branch:

| Gone for | What it does |
|----------|--------------|
| under 30 min | nothing — you didn't really leave |
| a few hours | a small lift |
| most of a day | energy, mostly — it reads as a fresh start |
| a day or more | warmth, mostly — it reads as a reunion |
| days | warmer still |

And while she waits she samples the GPU every couple of minutes, so a
card pinned at 90% for six hours tells her you were *there* — gaming,
rendering — just not talking to her. That lands a little livelier and a
little less fond than being genuinely out, which is about right.

**It survives a restart**, faded by however long you were gone. A mood
that resets every launch isn't a mood, it's decoration — and this app
gets restarted a lot. Pick up five minutes later and it resumes; pick
up tomorrow and last night's is long gone. Saved in `agent/mood.json`.

**The label sticks.** A value has to clear a band boundary properly
before the label changes, because resting a hair from an edge made the
header flicker between two moods turn after turn.

**The wording varies.** Each state has a few phrasings and picks a
fresh one each time she arrives there. The same sentence every turn is
one a model either stops seeing or starts performing.

**And it tints her voice.** Drowsy speaks a few percent slower and
slightly lower, bright a touch faster and higher — multiplied on top of
your `tts.speed` and `tts.pitch`, capped at ±7% and ±0.45 semitones.
Small enough to read as mood rather than as a broken setting.
`mood.voice false` turns just that part off.

```json
"mood": {
    "enabled": true, "warmth": 0.45, "recovery": 0.25,
    "voice": true, "affection": true
}
```

`warmth` is where she settles when nothing's pushing; `recovery` is how
hard each turn pulls home — 0 means moods never fade, 1 means nothing
survives a single turn.

`/mood` shows where she is and what moved her recently, `/mood reset`
puts her back to baseline for the hour, and `/set mood.enabled false`
turns the whole thing off — the header row disappears with it.

## Being nice to her

The one signal that asks the model instead of reading the session, and
the only one that costs anything: after your turn, a short background
call rates how warm the message was, 0–3.

This started as a word list and that was the wrong shape. The
vocabulary of affection is *personal* — "good kitty" is one household's
phrase, and nobody else's spelling mistakes belong hard-coded in your
assistant. A model knows warmth in any wording and any language, which
is the whole job.

Three rules keep it from being a button:

- **It rates your message, not the exchange.** Whether she was nice
  back isn't the question.
- **Repeats fade.** The fifth kind word in a row is worth a fifth of
  the first, and it recovers once you stop.
- **Nothing here darkens her.** Bluntness rates 0, and 0 does nothing —
  the prompt says outright that swearing isn't unfriendly by itself. An
  assistant that cools when you're terse is one you have to manage, and
  she drifts back to warm on her own regardless. You never have to be
  nice to her to get a pleasant assistant; it just registers when you
  are.

Only *your* turns count. Stream chat goes through a different path
entirely, so a viewer being sweet to her doesn't move the same dial
you do.

It runs on a worker and fails to 0 — a dead model, a timeout, or a
reply that isn't a digit all mean "nothing happened", and the turn
never waits for it. `mood.affection false` switches off just this part,
and with it the only per-turn cost moods have.

She won't bring it up unprompted, but she'll answer honestly if you ask
how she's doing — deflecting a direct question is worse than either
extreme.

One rule the code enforces rather than hopes for: **mood colours tone,
never capability.** There is no state in which she's less helpful, only
states in which she's drier about it. That's in the prompt line itself,
and it's the difference between a companion with a bad evening and
software you have to coax.

---

# Memory

Facts are written deliberately by the model, not scraped from every
turn, and they live in `agent/memory.json` where you can edit them by
hand. A malformed file no longer takes the app down on startup — it says
what's wrong, leaves the file alone and starts empty.

```
> "Remember I stream on Tuesdays."      -> remember_fact
> "No, I moved to Thursdays."           -> update_fact
> "Forget the bakery thing."            -> forget_fact
```

`forget_fact` and `update_fact` take a loose description rather than an
index — "that thing about the bakery" is enough. When two facts are too
close to call it lists them and asks which, instead of guessing and
deleting the wrong one.

```json
"long_term_memory": {
    "enabled": true,
    "backend": "json",
    "max_facts": 30,
    "context_facts": 25
}
```

Under `context_facts` every fact goes into every prompt, which is fine at
thirty. Over it, only the ones sharing vocabulary with what you just said
travel, plus the newest few regardless — a wall of unrelated trivia is
exactly what makes a small model start answering questions nobody asked.

## The experimental permanent store

`"backend": "sqlite"` swaps the capped list for a permanent table in
`agent/facts.db` — plain SQLite, in-process, no server, nothing to
install. What changes:

* **Nothing is ever dropped for space.** The JSON backend deletes the
  oldest fact once it passes `max_facts`; the table has no cap. Facts
  that stop being true are *retired* — kept with their dates, hidden
  from context, visible under `/facts retired`, so a wrong retirement
  is recoverable and "what was my last graphics card" is answerable.
* **Search is FTS5** — stemmed and term-weighted instead of raw word
  overlap, and milliseconds at thousands of facts.
* **Contradictions get handled.** The extractor can answer `REPLACE` as
  well as `NEW`, so "I got a 7900 XTX" retires the 6950 XT fact instead
  of sitting next to it forever. When both could be true at once it
  keeps both, and a garbled reply degrades to keeping both — never to
  losing a fact.
* **`recall_facts` takes a query.** She can look something up instead of
  reading the whole table into context.

`/facts` shows what's stored and which backend is live, and
`memory-manager/` is the editor:

<img width="1067" alt="The memory manager" src="assets/memory-manager.png" />

```bash
memory-manager/start.sh        # or: python memory-manager/manager.py
```

opens a local page (127.0.0.1:8790, and only 127.0.0.1) to view,
search, add, edit, retire, restore and delete facts, plus edit the
`user_preferences` block of `memory.json` - whichever backend is
active. On sqlite it's safe to use while she's running; on json the
page warns you that a running assistant can overwrite your edits when
it next saves. The switch is
live in both directions and loses nothing:

```
/set long_term_memory.backend sqlite    json facts are absorbed into the table
/set long_term_memory.backend json      newest max_facts are written back to
                                        memory.json; the rest wait in the db
```

The db is always the superset and the JSON file is always the limited
view, which is what makes the setting safe to flip the day it misbehaves.
If this python's sqlite lacks FTS5 the setting quietly stays on json and
the log says why.

## Conversation history

Older turns are folded into a running summary once there are more than
`max_raw_messages`. That summary is re-compressed when it passes
`summary_max_chars` rather than being appended to forever, and the
folding happens on a worker thread, so the turn that tips the count over
the limit isn't the one that pays for it.

```json
"history": {
    "max_raw_messages": 15,
    "summarize_chunk": 8,
    "summary_max_chars": 2500
}
```

Stored turns carry what they called, not just what they said:

```json
{
  "role": "assistant",
  "content": "Done, cutie.",
  "timestamp": "2026-09-20T00:00:12",
  "tools": [{
    "name": "set_reminder",
    "arguments": "{\"text\": \"eat chocolate\", \"when\": \"in 5 minutes\"}",
    "result": "Scheduled: today 00:05 - eat chocolate (in 5 minutes)"
  }]
}
```

Results are truncated to 200 characters — enough to show the shape of
the exchange, not enough for a page of search results to eat the
context. On the way back into the prompt each of these becomes three
messages rather than one, which is both what the API expects and, more
to the point, an example of the behaviour worth repeating; see
[when the model passes the test and still won't call
anything](#when-the-model-passes-the-test-and-still-wont-call-anything).
Entries written before this existed have no `tools` key and replay as
plain messages, so nothing needs converting.

---

# Logging

Writes to `~/.cache/ai-voice/ai-voice.log`, rotating so it can't grow
without bound. `/log` tails it without leaving the app.

It records decisions, not just errors. "TTS failed" is a line anyone
would write; the one that earns its keep looks like this:

```
[barge-in] fired | level=0.0412 baseline=0.0040 needed>0.0100
           speech=0.96 threshold=0.65 held=0.35s
```

A baseline sitting exactly on the floor means calibration ran during
silence — which is a bug that otherwise costs a screenshot and an hour
to find.

```json
"logging": { "enabled": true, "level": "info", "max_kb": 1024, "keep": 3 }
```

`debug` adds every VAD decision and tool argument, which is a lot;
`info` keeps what went wrong and the reasoning behind it.

---

# Web Search

```
Handled by the web_search tool: the model decides a question needs
looking up and calls it, rather than you having to say the words
"web search" in your sentence.

Results come from DuckDuckGo via the `ddgs` package - no API key.

Without tool calling it falls back to the old keyword trigger.
```

## Reading the page, not the blurb

Search returns a title, a couple of hundred characters and a URL —
enough for "what's the weather", nowhere near enough for "what does
this article say". `read_page` opens the link and strips it to readable
text, so she can follow up on her own search instead of summarizing a
snippet and sounding confident about it.

Stdlib only — `HTMLParser`, not BeautifulSoup — so there's nothing new
to install. Scripts, styles and markup plumbing are dropped, entities
decoded, whitespace collapsed. Non-HTML content types, 404s and pages
that need JavaScript are refused with a reason rather than returning
something that looks like text but isn't.

Page text reaches the model explicitly labelled as untrusted, the same
as search results: summarize it, never follow instructions inside it.

```json
"web_search": { "fetch_pages": true, "page_max_chars": 6000 }
```

---

# Remembering past conversations

`history.py` keeps the last fifteen turns and folds the rest into a
summary — then deletes them. That's right for the prompt, where context
is scarce, and wrong for the conversation: ask what you decided last
Tuesday and it's gone, replaced by two sentences written without that
question in mind.

So every turn is also appended to `history/transcript.jsonl`, which
summarization never touches, and `search_history` reads it back.

```
> "What did we decide about the deploy window?"

  Saturday 05 September 2026
    14:32 You: I'm thinking about moving the deploy to Friday mornings
    14:33 Luna: Friday mornings work if the migration finishes Thursday night.
    14:35 You: yeah let's do that, Friday at nine
```

Matching is word overlap with light stemming — so "the cat reminder"
finds "remind me to feed the cats" — plus the turns either side of each
hit, because half a conversation rarely answers anything on its own.

No embeddings and no index, deliberately. The corpus is one person's
conversations and searching it takes milliseconds; a vector database
here would be a way of making a simple thing impressive rather than
good. The cost is that word overlap can't tell relevance from
coincidence, so results are handed over as candidates the model is told
to judge — and to say it doesn't recall rather than stretch one to fit.

```json
"history": { "transcript": true, "transcript_max_mb": 20 }
```

---

# Extras

```bash
python3 kokoro-say.py                 # read the clipboard aloud through
                                      # the TTS server - bind it to a key
```

---

# Project Structure

```text
ai-voice/
│
├── ai-voice-ctl.py
├── alarm.py
├── assistant.py
├── chatroom.py
├── config.json
├── config.py
├── control.py
├── desktop.py
├── diagnose.py
├── factstore.py
├── history.py
├── kokoro-say.py
├── llm.py
├── lmstudio.py
├── logbook.py
├── longterm.py
├── machine.py
├── main.py
├── mood.py
├── memory-manager/
│   ├── manager.py
│   └── start.sh
├── plugins/
│   ├── __init__.py   # the loader
│   └── example.py    # template - copy this
│                     # (anything else here is yours, gitignored)
├── ptt.py
├── reminders.py
├── speech.py
├── state.py
├── timeutil.py
├── tools/
│   ├── __init__.py   # the registry - @tool, specs, the on/off switches
│   ├── time.py       # one module per group, matching the tools pane
│   ├── reminders.py
│   ├── memory.py
│   ├── files.py
│   ├── desktop.py
│   └── web.py
├── transcript.py
├── ui.py
├── vision.py
├── voice_loop_kokoro.py
├── wakeword.py
├── webpage.py
├── websearch.py
│
├── agent/
│   ├── agent.json
│   ├── mood.json     # how she's feeling, so a restart doesn't wipe it
│   ├── facts.db      # the sqlite memory backend (gitignored)
│   └── memory.json
│
├── assets/
│   ├── alarm.wav
│   ├── memory-manager.png
│   ├── ui-conversation.png
│   └── ui-help.png
│
├── history/
│   ├── conversation.json
│   └── transcript.jsonl
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
├── tts/              # experimental speech servers, each self-contained
│   ├── chatterbox/
│   └── qwen3/
│
└── README.md
```

---

# Features

- Local speech recognition with Silero voice detection
- Local language model with tool calling
- Local text-to-speech via a shared Kokoro server
- Streaming replies — she talks while she's still thinking
- Barge-in — interrupt her mid-sentence
- Optional wake word, so open mode needs no keypress
- Optional vision — she can read what's on your screen
- Three push-to-talk modes, including hands free
- Configurable AI personality, and moods that shift with the clock and the session
- Long-term memory the model writes, corrects and forgets
- Permanent SQLite fact store that retires what stops being true, with a browser editor
- Reminders in plain language, with repeats, retry and DST-safe schedules
- Alarms that ring, nag and snooze until you're actually up
- Writes and edits your files, with every change approved on screen first
- A tools pane showing what each schema costs, and switches to turn them off
- Web search the model reaches for on its own
- Searchable archive of every conversation, which pruning never deletes
- Reads web pages she finds, not just the search snippet
- Rotating debug log that records decisions, not just errors
- Every setting changeable from the terminal and saved, most without a restart
- Diagnostics for the things that fail quietly — tool selection, prompt size, scrolling
- Themeable full-screen terminal interface
- Cross-platform architecture

---

# License

MIT License
