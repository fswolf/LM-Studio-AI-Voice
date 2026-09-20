# -------------------------
# Assistant State
# -------------------------
assistant_busy = False
stop_speaking = False
stop_listening = False

# "typed" or "voice" - how the current turn reached her. Vision uses
# it to decide what "my screen" means: typed, the focused window is her
# own terminal; by voice through a compositor bind, you focused the
# window you meant before you spoke.
turn_source = "typed"

# True only while the microphone is open. HOME means different things
# either side of this: during recording it ends the recording, after it
# it cancels the turn.
recording = False

# Stop *generating*, not just stop speaking. HOME sets both; a barge-in
# sets only stop_speaking, because cutting the token stream on a false
# trigger loses the whole reply - and barge-in, with no echo
# cancellation to lean on, will sometimes trigger falsely.
stop_generating = False

# Set when the user talked over Luna, so the hands-free loop knows to
# start listening immediately instead of waiting out the settle pause.
barged_in = False

# True once a compositor-level hotkey is registered, so the evdev
# listener doesn't warn about a hotkey you already have.
hotkey_bound = False
