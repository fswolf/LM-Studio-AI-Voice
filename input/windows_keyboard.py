import os
import threading
import state
import ui

from pynput import keyboard
from assistant import assistant_task

def on_press(key, model):
    try:
        if key == keyboard.Key.home:
            # Stop current assistant action
            if state.assistant_busy:
                ui.set_status("Stopped")
                state.stop_speaking = True
                state.stop_listening = True
                return

            state.assistant_busy = True
            state.stop_speaking = False
            state.stop_listening = False

            thread = threading.Thread(
                target=assistant_task,
                args=(model,),
                daemon=True
            )

            thread.start()

        elif key == keyboard.Key.esc:
            os._exit(0)

    except Exception as e:
        ui.set_status(f"Keyboard error: {e}")


def start_keyboard(model):
    with keyboard.Listener(on_press=lambda key: on_press(key, model)) as listener:
        listener.join()