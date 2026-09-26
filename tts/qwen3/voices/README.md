# Voices

What goes here depends on the mode.

**`base`** — zero-shot cloning. Drop a .wav in and its filename is the
voice name; they're registered with the engine at startup.

    voices/luna.wav   ->  {"voice": "luna"}

A matching `luna.txt` containing *exactly what is said in the clip*
upgrades it from x-vector cloning to in-context cloning, which is
audibly better. Optional, and worth the two minutes for a voice you're
going to keep.

Use `../server-chatterbox/make_voice.py --prepare` to convert clips —
it produces the same 24 kHz mono this engine wants.

**`customvoice`** (the default) — this folder is ignored. The voices
are baked in: serena, vivian, ono_anna, sohee, uncle_fu, ryan, aiden,
eric, dylan.

**`voicedesign`** — also ignored. The voice *is* the description:
`"voice": "female, young adult, bright, slightly breathy"`.
