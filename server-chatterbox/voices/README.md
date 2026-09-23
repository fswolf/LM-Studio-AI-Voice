# Voices

Drop a reference clip in here and its filename becomes a voice name:

    voices/luna.wav   ->  {"voice": "luna"}

What makes a good reference clip:

* **7 to 20 seconds.** Shorter and the voice wanders; much longer and
  you're mostly adding noise for the model to copy.
* **One speaker, no music, no reverb.** It clones the room as
  faithfully as it clones the voice.
* **Normal delivery.** The clip sets the baseline; `exaggeration`
  pushes away from it. A reference that's already shouting leaves
  nowhere to go.
* 24 kHz or better, mono. `.wav` is safest; mp3/flac/ogg/m4a also work.

`default` is always available and is Chatterbox's own built-in voice.
Asking for a name with no file behind it quietly falls back to that,
so a typo costs you the character rather than the sentence.
