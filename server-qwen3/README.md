# server-qwen3

A drop-in replacement for kokoro-reader, speaking the same three
endpoints:

```
POST /tts   {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes
GET  /health                                           ->  {"ok": true}
GET  /voices                                           ->  the voice list
```

Runs on **8899** — the same port kokoro-reader uses — so anything
already pointed at it finds this instead with no change: the assistant,
the Firefox extension, anything else on the contract. Only one process
can hold the port, so stop kokoro-reader first.

To run both side by side and compare:

```bash
QWEN_PORT=8901 ./run.sh
```

## Why this one

| | VRAM | Licence | Speed | Cloning |
|---|---|---|---|---|
| Kokoro-82M | 2–3 GB | Apache 2.0 | ~30× | no |
| Chatterbox 0.5B | 4–6 GB | MIT | too slow here | yes |
| **Qwen3-TTS 0.6B Q8** | **~1.3 GB** | **Apache 2.0** | **~97ms first chunk** | **yes** |

Smaller than what you're already running, and it doesn't fight the 9B
in LM Studio for the card.

The bigger deal is that **there is no Python in the inference path.**
It's a C++17/GGML build — qwentts.cpp — so none of the
ROCm-versus-CUDA-wheel problems apply. It uses **Vulkan**, which on a
6950 XT is usually steadier than ROCm anyway, and sidesteps the whole
`HSA_OVERRIDE_GFX_VERSION` class of trouble. A build either works or
fails loudly; there's no silent fall back to CPU.

## Install

```bash
cd ~/ai-voice/server-qwen3
chmod +x setup.sh run.sh
./setup.sh      # clone + cmake + ~1.3 GB of weights
./run.sh
```

`setup.sh` checks its prerequisites first and tells you the dnf line if
anything's missing. The one people trip over is **glslc** — it compiles
the Vulkan compute shaders, ships separately from the driver, and its
absence is the most common reason a Vulkan build dies:

```bash
sudo dnf install git cmake gcc-c++ vulkan-loader vulkan-headers \
                 mesa-vulkan-drivers glslc curl
```

Overrides: `QWEN_BACKEND=cpu ./setup.sh`, `QWEN_QUANT=Q4_K_M`,
`QWEN_SIZE=1.7b`.

## Three ways to have a voice

Qwen3-TTS is really three models, and which is loaded decides what
"voice" means. One server is one mode; switching is a restart.

### `customvoice` — named speakers (the default)

Nothing to set up. Nine voices baked in:

```
serena  vivian  ono_anna  sohee  uncle_fu  ryan  aiden  eric  dylan
```

```json
"voice": "ono_anna"
```

### `base` — cloning

```bash
QWEN_MODE=base ./run.sh
```

Every `.wav` in `voices/` is registered with the engine at startup and
selectable by filename. A matching `luna.txt` holding *exactly what the
clip says* upgrades it from x-vector to in-context cloning, which is
audibly better — optional, and worth two minutes for a voice you'll
keep.

### `voicedesign` — describe it in words

```bash
QWEN_MODE=voicedesign ./run.sh
```

Then the voice **is** the description:

```json
"voice": "female, young adult, bright, slightly breathy"
```

This is the age-and-pitch control you were after, done properly rather
than by resampling the output. 1.7B only, so ~2.4 GB.

## Notes

**Speed** is a frame-drop stretch, not a resample, so changing it
doesn't move the pitch. (Pitch shifting lives in the assistant's
client and applies on top of whatever engine is running — `/set
tts.pitch 3`.)

**Unknown voice names fall back** to the default rather than erroring.
A typo costs you the character, not the sentence — which matters when
your `config.json` still says `af_bella` from the Kokoro days.

**The engine's own port** is 8971, not 8899 — that's the child
process, and nothing should talk to it directly.

**`GET /`** reports the realtime factor. Under 1.0 means she generates
slower than she speaks and streaming playback will stutter; that's the
number that decides whether this stays.

**The engine's output is relayed**, prefixed with `engine|`. A Vulkan
or model failure only shows up there, so it isn't swallowed.

## What was tested

35 checks against a stand-in for `tts-server`: health before the engine
is up, the OpenAI-shape translation, all three voice modes including
that `voicedesign` sends `instruct` rather than `voice`, clone
registration with and without a transcript, base64 framing, the
fallback on unknown names, speed stretching in both directions, WAV
framing and sample rate, trimming over-long input, engine failures
surfacing as 500, and that the model filenames match the ones actually
in the Hugging Face repo.

None of that covers how it sounds, which is the part you're testing.
