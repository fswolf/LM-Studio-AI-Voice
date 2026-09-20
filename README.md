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
    "volume": 1.0
}
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `TTS_URL` | `http://127.0.0.1:8899` | Server address |
| `TTS_VOICE` | `config.json` → `voice` | Voice (`af_bella`, `am_adam`, ...) |
| `TTS_SPEED` | `1.0` | 0.5 – 2.0 |
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
    "wake_word":  { ... },
    "tools":      { ... },
    "web_search": { ... },
    "reminders":  { ... },
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
| PgUp / PgDn | Scroll the conversation |
| End | Jump back to the newest message |
| Esc | Quit |

<img width="1210" alt="Help panel" src="assets/ui-help.png" />

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
| `/look` | List windows, or test a screenshot |
| `/log` | Tail the debug log without leaving the app |
| `/set` | List every setting, or change one — saved to `config.json` |
| `/tools` | Which tools the model can call — and which it can't, and why |
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

# Global Hotkey on Hyprland (Optional)

HOME works while the assistant's window is focused, on Linux, macOS and
Windows, with no permissions and nothing to configure. That covers
normal use.

If you want push-to-talk to fire while a *different* window is focused —
mid-game, or with OBS in front — the compositor has to own the hotkey,
because Wayland gives applications no global key grabs. That's a
protocol decision, not a missing library, so it's the same in any
language.

A control socket at `$XDG_RUNTIME_DIR/ai-voice.sock` exists for exactly
this. Bind a key to poke it:

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
script or panel button can drive the assistant too.

> Keep a modifier. A bare `HOME` bind is swallowed compositor-wide, so
> `Home` stops working in your terminal, editor and browser — including
> the assistant's own prompt.

The same socket works on Sway, KDE or GNOME; only the bind syntax
changes. On **X11 or a TTY** none of this is needed — `evdev` picks up
HOME globally as long as you can read the input devices:

```bash
sudo usermod -aG input "$USER"   # then log out and back in
```

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
| `list_reminders` / `cancel_reminder` | Read and cancel what's pending |
| `remember_fact` / `recall_facts` | Long-term memory, written deliberately |
| `forget_fact` / `update_fact` | Correct it when it got something wrong |
| `look_at_screen` | Take a screenshot and actually see it (optional) |
| `read_page` | Open a link and read it, not just the search snippet |
| `search_history` | Look through past conversations for something |
| `web_search` | DuckDuckGo, for anything it can't know |

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

> "Remind me in 10 minutes to clean the desk."
> "Wake me up at 7:30 tomorrow."
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
    "max_facts": 30,
    "context_facts": 25
}
```

Under `context_facts` every fact goes into every prompt, which is fine at
thirty. Over it, only the ones sharing vocabulary with what you just said
travel, plus the newest few regardless — a wall of unrelated trivia is
exactly what makes a small model start answering questions nobody asked.

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

---

# Extras

```bash
./push.sh "commit message"            # commit and push, with a check for
                                      # personal files still being tracked

python3 kokoro-say.py                 # read the clipboard aloud through
                                      # the TTS server - bind it to a key
```

## What push.sh protects you from

Two different problems, with opposite fixes.

**Private files** — `history/conversation.json`,
`history/transcript.jsonl`, `reminders/reminders.json` — should never
be public at all. `.gitignore` covers them, but only while they're
untracked: anything committed before a rule existed keeps getting
updated, which is how a chat log ends up on GitHub without anyone
deciding to put it there. `push.sh` spots those and prints the
`git rm --cached` to drop them.

**Personal files** — `config.json`, `agent/agent.json`,
`agent/memory.json` — *should* ship, so a fresh clone works. But your
copies are tuned to your machine, and a `git pull --rebase` will
cheerfully put the committed versions back over them. `push.sh` offers
to mark them `skip-worktree`, which keeps the published version intact
while git stops watching yours:

```
3 file(s) ship with the repo but are tuned to this machine:
  agent/agent.json
  agent/memory.json
  config.json
Mark them now? [y/N] y
  protected agent/agent.json
  protected agent/memory.json
  protected config.json
```

Say yes once and it stays quiet afterwards. The trade-off is that your
local edits to those files stop being pushed — change the published
defaults by unprotecting, committing, and protecting again.

> One thing `skip-worktree` does *not* do is make a conflicting pull
> seamless. If the remote changes a file you've protected, git refuses
> the merge with "your local changes would be overwritten" and aborts.
> That's the safe outcome — you're told rather than quietly losing work
> — but it looks like a broken repo. `push.sh` detects that case before
> you hit it and prints the way out: copy yours aside, unprotect, pull,
> re-protect, merge back by hand.

---

# Project Structure

```text
ai-voice/
│
├── ai-voice-ctl.py
├── assistant.py
├── config.json
├── config.py
├── control.py
├── history.py
├── hyprland.py
├── kokoro-say.py
├── llm.py
├── lmstudio.py
├── logbook.py
├── longterm.py
├── main.py
├── ptt.py
├── push.sh
├── reminders.py
├── speech.py
├── state.py
├── timeutil.py
├── tools.py
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
│   └── memory.json
│
├── assets/
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
- Configurable AI personality
- Long-term memory the model writes, corrects and forgets
- Reminders in plain language, with repeats, retry and DST-safe schedules
- Web search the model reaches for on its own
- Searchable archive of every conversation, which pruning never deletes
- Reads web pages she finds, not just the search snippet
- Rotating debug log that records decisions, not just errors
- Every setting changeable from the terminal and saved, most without a restart
- Themeable full-screen terminal interface
- Cross-platform architecture

---

# License

MIT License
