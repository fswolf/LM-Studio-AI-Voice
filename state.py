# -------------------------
# Assistant State
# -------------------------
assistant_busy = False
stop_speaking = False
stop_listening = False

# True once a compositor-level hotkey is registered, so the evdev
# listener doesn't warn about a hotkey you already have.
hotkey_bound = False
