import os
import threading
import state
import ui

from assistant import assistant_task

HOME_KEYCODE = "KEY_HOME"
ESC_KEYCODE = "KEY_ESC"

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


# ---------------------------------------------------------------------------
# Path 1: evdev (reads raw device, needs permission to open /dev/input/eventX)
# ---------------------------------------------------------------------------
def _find_keyboard_evdev():
    from evdev import InputDevice, list_devices

    devices = []

    for path in list_devices():
        device = InputDevice(path)

        if "keyboard" in device.name.lower() or "key" in device.name.lower():
            devices.append(device)

    if not devices:
        raise RuntimeError("No keyboard found")

    return devices[0]


def _on_key_evdev(event, model):
    from evdev import categorize, ecodes

    if event.type != ecodes.EV_KEY:
        return

    key = categorize(event)

    # only key down
    if key.keystate != key.key_down:
        return

    if key.keycode == HOME_KEYCODE:
        _handle_home(model)

    elif key.keycode == ESC_KEYCODE:
        os._exit(0)


def _run_evdev(model):
    keyboard = _find_keyboard_evdev()  # raises if no device / no permission

    for event in keyboard.read_loop():
        _on_key_evdev(event, model)


# ---------------------------------------------------------------------------
# Path 2: pynput fallback (X11, no device permissions needed)
# ---------------------------------------------------------------------------
def _run_pynput(model):
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.home:
                _handle_home(model)
            elif key == keyboard.Key.esc:
                os._exit(0)
        except Exception as e:
            ui.set_status(f"Keyboard error: {e}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def start_keyboard(model):
    try:
        _run_evdev(model)
    except (PermissionError, OSError, RuntimeError) as e:
        ui.add_message("system", f"evdev unavailable ({e}), falling back to pynput")
        _run_pynput(model)