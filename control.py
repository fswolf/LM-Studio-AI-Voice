"""Unix-socket control channel.

Wayland compositors don't hand global key grabs to applications, so on
Hyprland the compositor owns the hotkey and pokes us through this socket
instead:

    bind = SUPER, HOME, exec, python3 ~/ai-voice/ai-voice-ctl.py ptt

Commands: ptt (toggle listening / interrupt), stop, quit.
"""
import os
import socket
import threading

from config import CONTROL_SOCKET


def _serve(server, model, handlers):
    while True:
        try:
            connection, _ = server.accept()
        except OSError:
            return

        handler = None

        try:
            command = connection.recv(64).decode("utf-8", "ignore").strip().lower()
            handler = handlers.get(command)
            connection.sendall(b"ok\n" if handler else b"unknown\n")
        except Exception:
            handler = None
        finally:
            connection.close()

        # Run the action *after* replying so the client script exits
        # immediately instead of blocking for the whole turn.
        if handler:
            try:
                handler(model)
            except Exception:
                pass


def start(model, handlers):
    """Bind the socket and serve it in the background.

    Returns the socket path, or None if it couldn't be bound (stale file
    owned by another user, read-only directory, ...).
    """
    # A stale socket file has to go before bind() will work - but only if
    # nothing is actually listening on it, or a second instance would
    # quietly steal the hotkey from the one already running.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1)

    try:
        probe.connect(CONTROL_SOCKET)
        return None  # somebody's home
    except OSError:
        pass
    finally:
        probe.close()

    try:
        try:
            os.unlink(CONTROL_SOCKET)
        except FileNotFoundError:
            pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(CONTROL_SOCKET)
        os.chmod(CONTROL_SOCKET, 0o600)
        server.listen(8)
    except OSError:
        return None

    threading.Thread(
        target=_serve, args=(server, model, handlers), daemon=True
    ).start()

    return CONTROL_SOCKET


def cleanup():
    try:
        os.unlink(CONTROL_SOCKET)
    except OSError:
        pass
