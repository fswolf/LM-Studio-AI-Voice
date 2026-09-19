import os
import ui

from ptt import toggle

HOME_KEYCODE = "KEY_HOME"
ESC_KEYCODE = "KEY_ESC"


# ---------------------------------------------------------------------------
# Path 1: evdev (reads raw devices, needs read access to /dev/input/eventX,
# i.e. membership in the "input" group). This is the only path that works
# under Wayland/Hyprland, because Wayland gives no global key grabs.
# ---------------------------------------------------------------------------
def _find_keyboards_evdev():
    """Every device that actually reports the keys we care about.

    Matching on the device *name* was unreliable: plenty of keyboards
    don't say "keyboard", laptops expose several key-ish devices, and
    only the first match was ever read - so HOME landed on a device
    nobody was listening to. Match on capabilities instead, and listen
    to all of them.
    """
    from evdev import InputDevice, ecodes, list_devices

    devices = []

    for path in list_devices():
        try:
            device = InputDevice(path)
        except OSError:
            continue  # no permission on this one, try the rest

        keys = device.capabilities().get(ecodes.EV_KEY, [])

        if ecodes.KEY_HOME in keys and ecodes.KEY_ESC in keys:
            devices.append(device)

    if not devices:
        raise RuntimeError(
            "no readable keyboard in /dev/input "
            "(add yourself to the 'input' group and re-login)"
        )

    return devices


def _on_key_evdev(event, model):
    from evdev import categorize, ecodes

    if event.type != ecodes.EV_KEY:
        return

    key = categorize(event)

    # only key down
    if key.keystate != key.key_down:
        return

    keycode = key.keycode

    # evdev hands back a list when one scancode maps to several names
    if isinstance(keycode, (list, tuple)):
        keycode = keycode[0]

    if keycode == HOME_KEYCODE:
        toggle(model)

    elif keycode == ESC_KEYCODE:
        os._exit(0)


def _run_evdev(model):
    import selectors

    keyboards = _find_keyboards_evdev()  # raises if none readable

    selector = selectors.DefaultSelector()

    for keyboard in keyboards:
        selector.register(keyboard, selectors.EVENT_READ)

    while True:
        for selector_key, _ in selector.select():
            for event in selector_key.fileobj.read():
                _on_key_evdev(event, model)


# ---------------------------------------------------------------------------
# Path 2: pynput fallback (X11 only - under Wayland this sees nothing,
# which is why the control socket exists; see control.py)
# ---------------------------------------------------------------------------
def _run_pynput(model):
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.home:
                toggle(model)
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
    wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or \
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

    try:
        _run_evdev(model)
    except (PermissionError, OSError, RuntimeError, ImportError) as e:
        if wayland:
            # Wayland gives no global key grabs and pynput's X11 backend
            # sees nothing here, so there's nothing to fall back to. Not
            # worth a message: HOME works in the window regardless, which
            # is the only hotkey this app advertises now.
            return

        try:
            _run_pynput(model)
        except Exception:
            # No global hotkey on this box. HOME still works in-window.
            return
