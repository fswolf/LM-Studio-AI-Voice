#!/usr/bin/env python3
"""Build reference clips for Chatterbox, on demand.

Chatterbox clones whatever you point it at, so a "voice" here is just a
file in voices/. This makes those files two ways:

    # drop any clip into voices/, then:
    ./make_voice.py --prepare

    # one specific file, from anywhere
    ./make_voice.py --from-file ~/recordings/me.wav --name luna

    # or pull voices off a running TTS server to clone
    ./make_voice.py --url http://127.0.0.1:8899 af_sky af_nicole

    ./make_voice.py --list

--prepare is the everyday one: it sweeps voices/ and converts anything
that isn't already a usable reference - an mp3, a 48 kHz stereo
recording, a three-minute clip. Originals are moved to voices/originals/
rather than deleted, and a file that's already fine is left alone, so
running it twice does nothing the second time.

Everything here produces the same shape: 24 kHz mono, silence trimmed,
capped at 20 seconds. Beyond that you are mostly giving it more chances
to copy a room resonance.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voices")

# Roughly 18 seconds read at a normal pace. Statements, a question and
# a bit of list rhythm on purpose: the clone copies how the voice
# *moves*, and twenty seconds of one flat sentence teaches it one flat
# sentence. Deliberately unremarkable - the reference sets the baseline
# that `exaggeration` pushes away from, so anything already breathy or
# dramatic spends the range before you start.
DEFAULT_TEXT = (
    "Hey there. I'm just reading a few lines so you can hear how I "
    "sound when I'm talking normally. Did you get everything working "
    "on the first try, or was it one of those evenings? I can check "
    "the time, look something up, set a reminder, or take a look at "
    "your screen. Whatever you need, really. Let me know."
)

MAX_SECONDS = 20


def say(message):
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # `./make_voice.py --list | head` closes the pipe early, and an
        # unhandled one turns a normal shell idiom into a traceback.
        raise SystemExit(0)


def ffmpeg():
    return shutil.which("ffmpeg")


def tidy(source, destination, seconds=MAX_SECONDS):
    """24 kHz mono, silence trimmed off both ends, capped.

    Falls back to a plain copy when ffmpeg isn't there - an untrimmed
    reference still works, it just carries whatever pauses were in it
    into her speech.
    """
    tool = ffmpeg()

    if not tool:
        say("  ! ffmpeg not found - saving as-is (install it for trimming)")
        shutil.copyfile(source, destination)

        return False

    # silenceremove twice: once from the front, once from the back via
    # the reverse trick, because the filter only trims leading silence.
    command = [
        tool, "-y", "-loglevel", "error", "-i", source,
        "-af", ("silenceremove=start_periods=1:start_silence=0.1:"
                "start_threshold=-50dB,areverse,"
                "silenceremove=start_periods=1:start_silence=0.1:"
                "start_threshold=-50dB,areverse"),
        "-t", str(seconds), "-ar", "24000", "-ac", "1",
        "-c:a", "pcm_s16le", destination,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        # The whole command line is noise; the last line of stderr is
        # the part that says what was wrong with the file.
        detail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        say(f"  ! ffmpeg couldn't read it{': ' + detail[-1][:90] if detail else ''}")

        return False
    except (subprocess.SubprocessError, OSError) as e:
        say(f"  ! ffmpeg failed ({type(e).__name__}) - saving as-is")
        shutil.copyfile(source, destination)

        return False

    return True


def duration(path):
    try:
        with wave.open(path) as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# From a running TTS server
# ---------------------------------------------------------------------------
def engine_at(url):
    """What's answering that port. Both servers speak /health, and
    pointing this at Chatterbox itself is an easy mistake to make when
    they share a default port."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
            return json.loads(response.read()).get("engine", "unknown")
    except Exception:
        return None


def server_voices(url):
    try:
        with urllib.request.urlopen(f"{url}/voices", timeout=5) as response:
            data = json.loads(response.read())
    except Exception:
        return []

    if isinstance(data, dict):
        for key in ("voices", "available", "list"):
            if isinstance(data.get(key), list):
                return data[key]

    return data if isinstance(data, list) else []


def fetch(url, voice, text):
    request = urllib.request.Request(
        f"{url}/tts",
        data=json.dumps({"text": text, "voice": voice, "speed": 1.0}).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def from_server(url, names, text, out_name=None):
    engine = engine_at(url)

    if engine is None:
        say(f"Nothing answering {url} - start your source TTS server first.")
        say("If it's kokoro-reader and Chatterbox has the port, run one of")
        say("them elsewhere:  CHATTERBOX_PORT=8900 ./run.sh")

        return 1

    if engine == "chatterbox":
        say(f"{url} is Chatterbox itself - it can't be its own reference.")
        say("Point --url at kokoro-reader (or whatever you're cloning from).")

        return 1

    available = server_voices(url)

    if available:
        say(f"{engine or 'server'} offers: {', '.join(str(v) for v in available)}")

    os.makedirs(VOICE_DIR, exist_ok=True)
    made = 0

    for name in names:
        target = os.path.join(VOICE_DIR, (out_name or name) + ".wav")
        temporary = target + ".raw.wav"

        say(f"\n{name} -> {os.path.relpath(target, HERE)}")

        try:
            audio = fetch(url, name, text)
        except urllib.error.HTTPError as e:
            say(f"  ! server said {e.code}: {e.read()[:120].decode('utf-8', 'replace')}")
            continue
        except Exception as e:
            say(f"  ! {type(e).__name__}: {e}")
            continue

        with open(temporary, "wb") as handle:
            handle.write(audio)

        tidy(temporary, target)
        os.remove(temporary)

        seconds = duration(target)
        say(f"  {seconds:.1f}s, {os.path.getsize(target) // 1024} KB")

        if seconds < 5:
            say("  ! short for a reference - the voice may wander. Use --text")
            say("    with a longer passage.")

        made += 1

    return 0 if made else 1


# ---------------------------------------------------------------------------
# From a file you already have
# ---------------------------------------------------------------------------
def from_file(path, name):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        say(f"No such file: {path}")

        return 1

    os.makedirs(VOICE_DIR, exist_ok=True)
    target = os.path.join(VOICE_DIR, name + ".wav")

    say(f"{path} -> {os.path.relpath(target, HERE)}")
    tidy(path, target)

    seconds = duration(target)
    say(f"  {seconds:.1f}s, {os.path.getsize(target) // 1024} KB")

    if seconds < 5:
        say("  ! under 5s - expect the voice to drift between sentences.")
    elif seconds >= MAX_SECONDS - 0.2:
        say(f"  (trimmed to {MAX_SECONDS}s - longer doesn't help)")

    return 0


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac")
ORIGINALS = "originals"


def needs_work(path):
    """Is this already a good reference, or does it want converting?

    A .wav at 24 kHz mono under the cap is left alone, so running this
    twice does nothing the second time. Everything else - an mp3 you
    dropped in, a 48 kHz stereo recording, a three-minute clip - gets
    rebuilt.
    """
    if os.path.splitext(path)[1].lower() != ".wav":
        return "not wav"

    try:
        with wave.open(path) as handle:
            if handle.getnchannels() != 1:
                return "stereo"
            if handle.getframerate() != 24000:
                return f"{handle.getframerate()} Hz"
            if handle.getnframes() / handle.getframerate() > MAX_SECONDS + 0.5:
                return "too long"
    except Exception:
        return "unreadable as wav"

    return ""


def prepare():
    """Turn whatever is in voices/ into usable reference clips.

    Drop an mp3 in the folder and run this. Originals are moved aside
    rather than deleted - the conversion is lossy in one direction
    (trimmed, downmixed, capped) and you may want the source back.
    """
    os.makedirs(VOICE_DIR, exist_ok=True)
    keep = os.path.join(VOICE_DIR, ORIGINALS)

    try:
        entries = sorted(os.listdir(VOICE_DIR))
    except OSError as e:
        say(f"can't read {VOICE_DIR}: {e}")

        return 1

    todo = []

    for entry in entries:
        path = os.path.join(VOICE_DIR, entry)

        if not os.path.isfile(path):
            continue

        if os.path.splitext(entry)[1].lower() not in AUDIO_EXTENSIONS:
            continue

        reason = needs_work(path)

        if reason:
            todo.append((entry, reason))

    if not todo:
        say("Everything in voices/ is already a usable reference.")

        return listing()

    if not ffmpeg():
        say("ffmpeg isn't installed, and these need converting:")

        for entry, reason in todo:
            say(f"  {entry}  ({reason})")

        say("\n  sudo dnf install ffmpeg")

        return 1

    os.makedirs(keep, exist_ok=True)
    done = 0

    for entry, reason in todo:
        source = os.path.join(VOICE_DIR, entry)
        stem = os.path.splitext(entry)[0]
        target = os.path.join(VOICE_DIR, stem + ".wav")

        say(f"\n{entry}  ({reason})")

        # Two files with the same stem would silently clobber each
        # other - luna.mp3 overwriting a luna.wav you meant to keep.
        if os.path.exists(target) and os.path.abspath(target) != os.path.abspath(source):
            say(f"  ! {stem}.wav already exists - skipping. Rename one of them.")
            continue

        before = duration(source)
        temporary = target + ".tmp.wav"

        if not tidy(source, temporary):
            # tidy() falls back to a copy when ffmpeg fails, which is
            # not a conversion - don't pretend it was one.
            if os.path.splitext(entry)[1].lower() != ".wav":
                say("  ! couldn't convert it; leaving the original alone")
                if os.path.exists(temporary):
                    os.remove(temporary)
                continue

        # A corrupt or empty source converts "successfully" into nothing.
        # Moving the original away at that point loses the only copy of
        # something that might still be salvageable by hand.
        if duration(temporary) < 0.25:
            say("  ! produced no audio - the file may be corrupt. Leaving it alone.")
            os.remove(temporary)
            continue

        # Move the original aside only once the replacement is known good.
        shutil.move(source, os.path.join(keep, entry))
        shutil.move(temporary, target)

        after = duration(target)
        was = f"{before:.1f}s -> " if before else ""
        say(f"  {was}{after:.1f}s, 24 kHz mono"
            f"  ({os.path.getsize(target) // 1024} KB)")

        if after < 5:
            say("  ! under 5s - the voice will drift between sentences")

        done += 1

    if done:
        say(f"\nOriginals moved to voices/{ORIGINALS}/")

    say("")

    return listing()


def listing():
    say(f"{VOICE_DIR}\n")

    try:
        entries = sorted(os.listdir(VOICE_DIR))
    except OSError:
        entries = []

    found = False

    for entry in entries:
        stem, extension = os.path.splitext(entry)

        if extension.lower() not in AUDIO_EXTENSIONS:
            continue

        found = True
        path = os.path.join(VOICE_DIR, entry)
        seconds = duration(path)
        length = f"{seconds:.1f}s" if seconds else "?"
        note = ""

        reason = needs_work(path)

        if reason:
            note = f"   ({reason} - run: prepare)"
        elif seconds and seconds < 5:
            note = "   (short - may drift)"

        say(f'  {stem:<16} {length:>7}  {entry}{note}')

    if not found:
        say("  (empty - Chatterbox will use its own built-in voice)")

    say('\n  "default" is always available: Chatterbox\'s built-in voice.')
    say('  Pick one with  "voice": "<name>"  in the assistant\'s config.json.')

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Make Chatterbox reference clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument("voices", nargs="*",
                        help="voice names to pull from the source server")
    parser.add_argument("--url", default="http://127.0.0.1:8899",
                        help="source TTS server (default: %(default)s)")
    parser.add_argument("--from-file", metavar="PATH",
                        help="clean up an existing clip instead")
    parser.add_argument("--name", help="save under this name")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="what the reference should say")
    parser.add_argument("--list", action="store_true",
                        help="show the clips you already have")
    parser.add_argument("--prepare", action="store_true",
                        help="convert whatever is sitting in voices/")

    arguments = parser.parse_args()

    if arguments.prepare:
        return prepare()

    if arguments.list:
        return listing()

    if arguments.from_file:
        name = arguments.name or os.path.splitext(
            os.path.basename(arguments.from_file)
        )[0]

        return from_file(arguments.from_file, name)

    if not arguments.voices:
        parser.print_help()

        return 1

    if arguments.name and len(arguments.voices) > 1:
        say("--name takes one voice at a time.")

        return 1

    return from_server(
        arguments.url.rstrip("/"), arguments.voices,
        arguments.text, arguments.name,
    )


if __name__ == "__main__":
    sys.exit(main())
