#!/usr/bin/env python3
"""Poke a running assistant through its control socket.

Stdlib only - no venv needed, which matters because Hyprland's `exec`
runs outside the assistant's virtualenv.

    python3 ai-voice-ctl.py ptt    # toggle listening / interrupt
    python3 ai-voice-ctl.py stop   # interrupt only
    python3 ai-voice-ctl.py quit
"""
import os
import socket
import sys

SOCKET_PATH = os.environ.get(
    "AI_VOICE_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "ai-voice.sock"),
)


def main():
    command = (sys.argv[1] if len(sys.argv) > 1 else "ptt").strip().lower()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(3)

    try:
        client.connect(SOCKET_PATH)
    except OSError as e:
        print(f"ai-voice not running? ({SOCKET_PATH}: {e})", file=sys.stderr)
        return 1

    try:
        client.sendall(command.encode() + b"\n")
        reply = client.recv(16).decode().strip()
    except OSError as e:
        print(f"send failed: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    if reply != "ok":
        print(f"unknown command: {command}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
