#!/usr/bin/env python3
"""Speak text through the kokoro-reader server.

Anything you can select, Luna can read - the Claude desktop app, a man
page, a code review, a wall of docs.

    kokoro-say.py "hello there"      # speak the argument
    somecommand | kokoro-say.py      # speak stdin
    kokoro-say.py                    # speak the clipboard / selection
    kokoro-say.py --stop             # shut up

Bind it to a key and pressing it again while it's talking stops it,
which is what you actually want when a reply turns out to be long:

    # hyprland.lua
    hl.bind("SUPER + R", hl.dsp.exec_cmd("python3 ~/ai-voice/kokoro-say.py"))

Stdlib only, so it doesn't need the venv.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

URL = os.environ.get("KOKORO_URL", "http://127.0.0.1:8899").rstrip("/")
VOICE = os.environ.get("KOKORO_VOICE", "af_bella")
SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))

# The server truncates at 1200 characters, so split below that.
MAX_CHARS = 1000

PID_FILE = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "kokoro-say.pid"
)

# First one installed wins. All of these read a WAV from stdin.
PLAYERS = (
    ("pw-play", ["pw-play", "-"]),
    ("paplay", ["paplay"]),
    ("aplay", ["aplay", "-q", "-"]),
    ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]),
)


def clipboard():
    """Selection first, then clipboard - selecting text and hitting the
    key shouldn't need an explicit copy."""
    for command in (
        ["wl-paste", "--primary", "--no-newline"],
        ["wl-paste", "--no-newline"],
        ["xclip", "-o", "-selection", "primary"],
        ["xclip", "-o", "-selection", "clipboard"],
        ["xsel", "-o"],
    ):
        if not shutil.which(command[0]):
            continue

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            continue

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    return ""


def stop_playing():
    """Kill a previous run. Returns True if something was talking."""
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        stopped = True
    except (ProcessLookupError, PermissionError, OSError):
        stopped = False

    try:
        os.unlink(PID_FILE)
    except OSError:
        pass

    return stopped


def clean(text):
    """Strip the things that sound awful read aloud.

    Markdown from a chat reply is the main offender - nobody wants to
    hear "asterisk asterisk important asterisk asterisk", and a fenced
    code block read character by character is unbearable.
    """
    text = re.sub(r"```.*?```", " (code block) ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> label
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunks(text):
    """Split on sentence boundaries, under the server's limit."""
    pieces = []

    for sentence in re.split(r"(?<=[.!?…])\s+|\n{2,}", text):
        sentence = sentence.strip()

        if not sentence:
            continue

        while len(sentence) > MAX_CHARS:
            cut = sentence.rfind(" ", 0, MAX_CHARS)

            if cut <= 0:
                cut = MAX_CHARS

            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if pieces and len(pieces[-1]) + len(sentence) + 1 <= MAX_CHARS:
            pieces[-1] = f"{pieces[-1]} {sentence}"
        else:
            pieces.append(sentence)

    return [p for p in pieces if p]


def synthesize(text):
    request = urllib.request.Request(
        f"{URL}/tts",
        data=json.dumps({"text": text, "voice": VOICE, "speed": SPEED}).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def player():
    for name, command in PLAYERS:
        if shutil.which(name):
            return command

    return None


def main():
    args = [a for a in sys.argv[1:] if a not in ("--stop", "-s")]
    stopping = len(args) != len(sys.argv[1:])

    # Pressing the same key again is a stop, not a second voice.
    was_playing = stop_playing()

    if stopping or was_playing:
        return 0

    if args:
        text = " ".join(args)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = clipboard()

    text = clean(text)

    if not text:
        print("Nothing to say.", file=sys.stderr)
        return 1

    command = player()

    if command is None:
        print(
            "No audio player found - install one of: "
            + ", ".join(name for name, _ in PLAYERS),
            file=sys.stderr,
        )
        return 1

    # Own process group, so stopping kills the player too.
    os.setpgrp()

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        for piece in chunks(text):
            try:
                audio = synthesize(piece)
            except urllib.error.URLError as e:
                print(f"Kokoro server unreachable at {URL}: {e}", file=sys.stderr)
                return 1

            with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                wav.write(audio)
                wav.flush()

                if command[0] == "paplay":
                    subprocess.run(command + [wav.name], check=False)
                else:
                    subprocess.run(command, stdin=open(wav.name, "rb"), check=False)
    finally:
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
