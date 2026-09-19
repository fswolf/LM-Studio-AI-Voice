# -------------------------
# Assistant State
# -------------------------
assistant_busy = False
stop_speaking = False
stop_listening = False

# Set when the user talked over Luna, so the hands-free loop knows to
# start listening immediately instead of waiting out the settle pause.
barged_in = False

# True once a compositor-level hotkey is registered, so the evdev
# listener doesn't warn about a hotkey you already have.
hotkey_bound = False
