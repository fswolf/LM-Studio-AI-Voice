import os
import threading
import state
import ui

from pynput import keyboard
from assistant import assistant_task


def _handle_home(model):
    if state.assistant_busy:
        ui.set_status("Stopped")
        state.stop_speaking = True
        state.stop_listening = True
        return

    state.assistant_busy = True
    state.stop_speaking = False
    state.stop_listening = False

    thread = threading.Thread(target=assistant_task, args=(model,), daemon=True)
    thread.start()


def start_keyboard(model):
    # macOS requires the app reading global keypresses (this terminal /
    # Python process) to be granted Accessibility access:
    # System Settings -> Privacy & Security -> Accessibility.
    # Without it, pynput's listener runs but never receives events -
    # HOME/ESC will silently do nothing rather than error out.
    ui.add_message(
        "system",
        "If HOME/ESC don't respond, grant Accessibility access to this "
        "terminal in System Settings > Privacy & Security > Accessibility.",
    )

    def on_press(key):
        try:
            if key == keyboard.Key.home:
                _handle_home(model)
            elif key == keyboard.Key.esc:
                # os._exit terminates the whole process immediately,
                # regardless of which thread calls it. This callback
                # runs in a background hotkey thread, so plain exit()/
                # sys.exit() would only kill this thread and leave the
                # main thread hanging on input().
                os._exit(0)
        except Exception as e:
            ui.set_status(f"Keyboard error: {e}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()