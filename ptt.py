"""Push-to-talk, in three shapes.

  auto   - HOME starts listening, silence ends the turn.
  manual - HOME starts recording, HOME stops it. Silence is ignored, so
           a pause mid-sentence doesn't cut you off.
  open   - hands free. HOME arms the mic and it stays armed: speech
           starts a turn, silence ends it, and once Luna has answered it
           re-arms itself. HOME again switches it off.

Shared by the keyboard listeners, the control socket and the TUI, so
every input path behaves identically.
"""
import threading
import time

import config
import state
import ui

from assistant import assistant_task

# Runtime override for config.STT_MODE, set by /mode.
_mode = config.STT_MODE

_hands_free = False
_loop_thread = None


def mode():
    return _mode


def set_mode(new_mode):
    """Switch modes at runtime. Returns the mode actually in effect."""
    global _mode

    new_mode = str(new_mode).strip().lower()

    if new_mode not in ("auto", "manual", "open"):
        return None

    # Leaving open mode while the loop is running would strand it.
    if _mode == "open" and new_mode != "open":
        stop_hands_free()

    _mode = new_mode
    ui.set_mode(_mode)

    return _mode


def hands_free_active():
    return _hands_free


# ---------------------------------------------------------------------------
# Single turn (auto / manual)
# ---------------------------------------------------------------------------
def _start_turn(model):
    state.assistant_busy = True
    state.stop_speaking = False
    state.stop_listening = False

    threading.Thread(
        target=assistant_task, args=(model, _mode), daemon=True
    ).start()


def _interrupt():
    ui.set_status("Stopped")
    state.stop_speaking = True
    state.stop_listening = True


# ---------------------------------------------------------------------------
# Hands-free loop (open)
# ---------------------------------------------------------------------------
def _hands_free_loop(model):
    """Listen, answer, listen again.

    Deliberately sequential: recording only happens between turns, never
    while Luna is speaking. Without that, her own voice out of the
    speakers retriggers the mic and she talks to herself indefinitely.
    The settle pause covers the tail end of playback.
    """
    while _hands_free:
        state.stop_speaking = False
        state.stop_listening = False
        state.assistant_busy = True

        try:
            assistant_task(model, "open")
        except Exception as e:
            ui.add_message("system", f"Listening stopped: {e}")
            break
        finally:
            state.assistant_busy = False

        if not _hands_free:
            break

        time.sleep(config.STT_SETTLE_SECONDS)

    stop_hands_free()


def start_hands_free(model):
    global _hands_free, _loop_thread

    if _hands_free:
        return

    _hands_free = True
    ui.set_status("Listening (hands free)")

    _loop_thread = threading.Thread(
        target=_hands_free_loop, args=(model,), daemon=True
    )
    _loop_thread.start()


def stop_hands_free():
    global _hands_free

    if not _hands_free:
        return

    _hands_free = False

    # Breaks any recording currently in progress.
    state.stop_listening = True
    state.stop_speaking = True

    ui.set_status("Idle")


# ---------------------------------------------------------------------------
# What HOME does
# ---------------------------------------------------------------------------
def toggle(model):
    if _mode == "open":
        if _hands_free:
            stop_hands_free()
        else:
            start_hands_free(model)
        return

    if state.assistant_busy:
        # Mid-turn: in manual this is "stop recording and transcribe",
        # otherwise it's "shut up / cancel".
        _interrupt()
        return

    _start_turn(model)


def stop(model=None):
    """Interrupt only - never starts a turn."""
    if _mode == "open":
        stop_hands_free()
        return

    if state.assistant_busy:
        _interrupt()
