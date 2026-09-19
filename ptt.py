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
import speech
import state
import ui
import wakeword

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
    state.barged_in = False


# ---------------------------------------------------------------------------
# Hands-free loop (open)
# ---------------------------------------------------------------------------
def _hands_free_loop(model):
    """Listen, answer, listen again.

    Recording proper still only happens between turns - her own voice
    out of the speakers would otherwise retrigger the mic and she'd
    talk to herself indefinitely - and the settle pause covers the tail
    end of playback.

    Barge-in is the exception, and it is handled during playback by a
    separate monitor with a much higher bar (speech.watch_for_barge_in).
    When it fires the user is already mid-sentence, so waiting out the
    settle pause would eat the start of what they're saying.
    """
    # With a wake word loaded, she waits for her name before the first
    # turn - and then stays open for a while afterwards, so a follow-up
    # question doesn't need saying it again. Without one, every turn
    # starts on speech, which is what open mode always did.
    follow_up_until = 0.0

    while _hands_free:
        state.stop_speaking = False
        state.stop_listening = False
        state.barged_in = False

        if wakeword.available() and time.monotonic() > follow_up_until:
            if not speech.wait_for_wake_word(lambda: not _hands_free):
                break

            if not _hands_free:
                break

            time.sleep(wakeword.cooldown_seconds())

        state.assistant_busy = True

        try:
            # Once the wake word has armed it, a silent turn should
            # time out rather than wait forever - it was probably the
            # television. Without a wake word, waiting is the point.
            spoke = assistant_task(
                model, "auto" if wakeword.available() else "open"
            )
        except Exception as e:
            ui.add_message("system", f"Listening stopped: {e}")
            break
        finally:
            state.assistant_busy = False

        if not _hands_free:
            break

        # Only hold the door open if she actually answered something.
        follow_up_until = (
            time.monotonic() + config.WAKE_WORD_FOLLOW_UP_SECONDS
            if spoke else 0.0
        )

        if not state.barged_in:
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
