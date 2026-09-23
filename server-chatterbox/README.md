# server-chatterbox

A drop-in replacement for kokoro-reader, speaking the same three
endpoints on the same port:

```
POST /tts   {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes
GET  /health                                           ->  {"ok": true}
GET  /voices                                           ->  the voice list
```

The assistant needs no change. Stop kokoro-reader, start this, done.

## Why Chatterbox

| | VRAM | Licence | Cloning | Expression |
|---|---|---|---|---|
| Kokoro-82M | 2–3 GB | Apache 2.0 | no | no |
| **Chatterbox 0.5B** | **4–6 GB** | **MIT** | **yes, ~7s** | **yes** |
| Orpheus 3B | 8–12 GB | Apache 2.0 | yes | yes |
| F5-TTS | 4–8 GB | CC-BY-**NC** | yes | no |

Orpheus sounds better. It also wants 8–12 GB, and on a 16 GB card
that's a choice between her voice and her brain — the 9B in LM Studio
is already sitting there. Chatterbox fits alongside it.

What it buys over Kokoro isn't a nicer preset. It's a voice cloned from
a few seconds of reference audio, and an `exaggeration` dial that
changes *how a line is delivered* rather than how fast it's read.

## Install

```bash
cd ~/ai-voice/server-chatterbox
chmod +x setup.sh run.sh
./setup.sh
```

Its own venv, deliberately: `chatterbox-tts` pins a torch that would
fight with the assistant's, and the assistant only needs `requests` to
talk to this thing, so there's no reason to share a Python.

The script installs torch from the **ROCm** index, then chatterbox with
`--no-deps` — because its requirements pin a CUDA torch and pip will
happily replace the ROCm build you just installed with it. That one
flag is the difference between using the 6950 XT and not.

Overrides:

```bash
ROCM_INDEX=https://download.pytorch.org/whl/rocm6.4 ./setup.sh   # newer ROCm
CPU=1 ./setup.sh                                                 # no GPU
```

If pip starts compiling things from source, you're on Python 3.12 and
some dependency has no wheel for it. `sudo dnf install python3.11` and
re-run; the script prefers 3.11 when it's there.

## Run

```bash
./run.sh
```

It answers the port immediately and loads the model on a worker, so
`/health` reports `{"ok": false, "loading": true}` for the first minute
rather than leaving the assistant hanging on its first sentence. The
assistant already knows what to do with that — it shows the server
offline and carries on as a text chat until it comes up.

`run.sh` sets `HSA_OVERRIDE_GFX_VERSION=10.3.0`, which is what makes
gfx1030 work on ROCm builds that don't list it as officially supported.

## Voices

**The built-in voice is not Luna and never will be.** Chatterbox ships
one stock voice — fairly neutral, male-leaning — and everything else
comes from cloning a reference clip. With `voices/` empty you are
listening to the factory setting.

Drop any clip into `voices/` — mp3, whatever — and:

```bash
./voices.sh prepare          # convert everything that needs it
./voices.sh test mysample    # HEAR it, cloned, right now
./voices.sh use mysample     # write it into config.json
```

`prepare` sweeps the folder and rebuilds anything that isn't already a
usable reference: an mp3, a 48 kHz stereo recording, a three-minute
clip. Files that are already fine are left alone, so running it twice
does nothing the second time. Originals move to `voices/originals/`
rather than being deleted, and a file it can't read is reported and
left where it is rather than half-converted.

```
mysample.mp3  (not wav)
  12.0s, 24 kHz mono  (562 KB)

longone.wav  (stereo)
  45.0s -> 20.0s, 24 kHz mono  (937 KB)
```

The rest:

```bash
./voices.sh grab af_sky af_nicole      # pull voices off a running TTS server
./voices.sh from ~/recordings/me.wav luna
./voices.sh list
./voices.sh status                     # what's running, what's selected
```

`test` is the one that matters. It asks the *running* Chatterbox to say
a line in a given voice and plays it back, so auditioning a candidate
costs ten seconds instead of an edit-and-restart cycle. It finds both
servers by asking `/health` what they are rather than trusting port
numbers, since both default to 8899.

`make_voice.py` underneath does the clip-building, if you want the
flags directly:

```bash
# audition Kokoro's voices as Chatterbox references
./make_voice.py --url http://127.0.0.1:8899 af_sky af_nicole af_bella

# keep the one you liked, under a name of your own
./make_voice.py --url http://127.0.0.1:8899 af_sky --name luna

# or clean up a clip you already have
./make_voice.py --from-file ~/recordings/me.wav --name luna

./make_voice.py --list
```

Both routes produce the same thing — 24 kHz mono, silence trimmed,
capped at 20 seconds — which is what a good reference looks like.

Kokoro is the obvious source: you can already listen to its voices, the
output is clean, it's synthetic so there's nobody's voice being taken,
and cloning one keeps the timbre you picked while adding the expression
control Kokoro hasn't got. Both servers default to port 8899, so run
one elsewhere while you do this:

```bash
CHATTERBOX_PORT=8900 ./run.sh
```

Drop any clip in `voices/` by hand and its filename is the voice name:

```
voices/luna.wav   ->  {"voice": "luna"}
```

Then in the assistant's `config.json`:

```json
"voice": "luna"
```

A good reference clip is 7–20 seconds, one speaker, no music, no
reverb, delivered normally — it clones the room as faithfully as the
voice, and a reference that's already shouting leaves `exaggeration`
nowhere to go. `default` is always available and is Chatterbox's own
built-in voice. Asking for a name with no file behind it falls back to
that rather than erroring, so a typo costs you the character, not the
sentence.

## Dials

Environment variables, or per-request fields in the POST body:

| | Default | |
|---|---|---|
| `CHATTERBOX_EXAGGERATION` | 0.5 | Neutral. Past ~0.8 it starts acting rather than talking |
| `CHATTERBOX_CFG` | 0.5 | Lower slows delivery and leaves room between phrases |
| `CHATTERBOX_TEMPERATURE` | 0.8 | Variation between takes |
| `CHATTERBOX_VARIANT` | *original* | `turbo` is faster; `multilingual` does 23 languages |
| `CHATTERBOX_DEVICE` | auto | Force `cpu` or `cuda` |

For a character voice, exaggeration around 0.6–0.7 with cfg near 0.4 is
a reasonable place to start — expressive without going pantomime.

`GET /` returns the running stats, including the realtime factor, which
is the number to watch: under 1.0 means she generates slower than she
speaks and the streaming playback will stutter.

## Speed

Chatterbox has no speed control and the contract has one, so it's done
here with a phase vocoder. Resampling would be one line but it drags
the pitch along with it, and "slightly faster" becomes a chipmunk.

## What was tested

The HTTP contract is covered end to end against a stubbed model — 33
checks on the health-before-loaded behaviour, WAV framing and sample
rate, voice resolution and fallback, speed stretching, bad input, the
over-long-post trim, the retry when a build rejects the extra dials,
and that generation serializes on one GPU. What that does *not* cover
is how it sounds, which is the part you're testing.
