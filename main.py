import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import threading

import config
from config import AGENT_NAME, VOICE, KOKORO_URL
from speech import load_models
from input import start_keyboard
import assistant
import control
import history
import ptt
import reminders
import speech
import state
import timeutil
import ui
import vision
import wakeword


def shutdown(code=0):
    """os._exit skips atexit, so every exit path comes through here."""
    control.cleanup()
    os._exit(code)


# -------------------------
# Startup
# -------------------------
history.load()
reminders.load()

try:
    models = requests.get("http://localhost:1234/v1/models", timeout=5).json()
except requests.exceptions.ConnectionError:
    print(
        "\nCouldn't reach LM Studio at http://localhost:1234\n"
        "Make sure LM Studio is open, a model is loaded, and its local "
        "server is started (Developer tab) before running this.\n"
    )
    raise SystemExit(1)

MODEL = models["data"][0]["id"]

print("Loading models...")
load_models()

if wakeword.enabled() and not wakeword.load():
    print(f"Wake word unavailable: {wakeword.label()}")

ui.init(
    agent_name=AGENT_NAME,
    model=MODEL,
    voice=VOICE,
    voice_server=speech.server_label(),
)

# Seed the on-screen conversation with what was loaded from disk, so
# past turns are visible right away instead of starting on a blank screen.
for msg in history.get_messages():
    speaker = "user" if msg["role"] == "user" else AGENT_NAME.lower()
    ui.conversation.append((speaker, msg["content"]))

# TTS now lives in the kokoro-reader server, so say so up front rather
# than letting the first reply die with a connection error.
if not speech.kokoro_ok:
    ui.add_message(
        "system",
        f"Kokoro server not answering at {KOKORO_URL} - start kokoro_server.py "
        f"(kokoro-reader) or the assistant will stay silent. ({speech.kokoro_error})",
    )

ui.set_mode(ptt.mode())
ui.set_status("Idle")

# -------------------------
# Input paths:
#   * the TUI itself - HOME while this window is focused, everywhere
#   * control socket - optional, for scripting or a compositor bind
#   * evdev hotkey   - HOME from any window, X11/TTY only
# -------------------------
socket_path = control.start(
    MODEL,
    {
        "ptt": ptt.toggle,
        "stop": ptt.stop,
        "quit": lambda _model: shutdown(),
    },
)

threading.Thread(target=start_keyboard, args=(MODEL,), daemon=True).start()
threading.Thread(target=reminders.run_scanner, args=(MODEL,), daemon=True).start()


# -------------------------
# Turn handling
# -------------------------
def _process(text):
    """One typed turn. Runs on a worker so the TUI stays responsive
    while the model thinks and Luna speaks."""
    ui.add_message("user", text)

    state.assistant_busy = True
    state.stop_speaking = False
    state.stop_generating = False
    state.turn_source = "typed"

    try:
        assistant.respond(text, MODEL)
        ui.set_status("Idle")
    except Exception as e:
        # Without this, an LM Studio error (context overflow, bad param,
        # connection drop, etc.) would kill the worker silently instead
        # of just this one turn.
        ui.add_message("system", f"Error: {e}")
        ui.set_status("Idle")
    finally:
        state.assistant_busy = False


def handle_input(text):
    """Called on the UI thread - dispatch fast, never block here."""
    if text in ("/quit", "/exit"):
        ui.stop()
        return

    if text == "/clear":
        ui.conversation.clear()
        history.clear()
        ui.render()
        return

    if text == "/keys":
        ui.add_message(
            "system",
            "socket={} | session={} | hotkey=HOME (in-window)".format(
                socket_path or "none",
                os.environ.get("XDG_SESSION_TYPE", "?"),
            ),
        )
        return

    if text == "/reminders":
        items = reminders.pending()

        if not items:
            ui.add_message("system", "No reminders pending.")
        else:
            listing = "\n".join(
                f"  {i}. {reminders.describe(r)}" for i, r in enumerate(items, 1)
            )
            ui.add_message("system", f"{len(items)} pending:\n{listing}")
        return

    if text.startswith("/cancel"):
        parts = text.split()

        if len(parts) != 2 or not parts[1].isdigit():
            ui.add_message("system", "Usage: /cancel <number from /reminders>")
            return

        removed = reminders.cancel(int(parts[1]))
        ui.add_message(
            "system",
            f"Cancelled: {removed['text']}" if removed else "No reminder with that number.",
        )
        return

    if text.startswith("/when"):
        phrase = text[5:].strip()

        if not phrase:
            ui.add_message(
                "system",
                "Usage: /when <phrase> - e.g. /when next friday at 4pm. "
                "Shows how the reminder parser reads it, without "
                "scheduling anything.",
            )
            return

        due, repeat = timeutil.parse_when(phrase)

        if due is None:
            ui.add_message("system", f"{phrase!r} -> not a time I can read.")
        else:
            ui.add_message(
                "system",
                "{!r} -> {} ({}){}".format(
                    phrase, timeutil.friendly(due), timeutil.relative(due),
                    f", repeats {timeutil.describe_repeat(repeat)}" if repeat else "",
                ),
            )
        return

    if text.startswith("/mode"):
        parts = text.split()

        if len(parts) == 1:
            ui.add_message(
                "system",
                f"Mode: {ptt.mode()}. "
                "auto = HOME starts, silence stops. "
                "manual = HOME starts, HOME stops. "
                "open = hands free, mic stays armed. "
                "Change with /mode <name>.",
            )
            return

        applied = ptt.set_mode(parts[1])

        ui.add_message(
            "system",
            f"Mode: {applied}" if applied
            else f"Unknown mode {parts[1]!r} - use auto, manual or open.",
        )
        return

    if text == "/tools":
        import llm

        if not llm.tools_active():
            ui.add_message(
                "system",
                "Tool calling is off or unsupported by this model - "
                "keyword triggers are handling reminders and search.",
            )
            return

        import tools as tool_registry

        lines = ["Tools she can call: " + ", ".join(tool_registry.names())]

        # A tool that isn't offered is invisible otherwise, and "why
        # didn't she look at my screen" has exactly one answer worth
        # printing: because she wasn't told she could.
        for name, why in tool_registry.unavailable():
            lines.append(f"  {name} is NOT offered - {why}")

        ui.add_message("system", "\n".join(lines))
        return

    if text.startswith("/look"):
        argument = text[5:].strip()

        if not vision.available():
            ui.add_message("system", f"Can't look: {vision.why_unavailable()}")
            return

        if argument in ("", "list", "windows"):
            # What she'd pick, and what else she could have picked.
            open_windows = vision.windows()

            if not open_windows:
                listing = "  (hyprctl listed nothing she can look at)"
            else:
                listing = "\n".join(
                    "  {} {} - {}".format(
                        "*" if index == 0 else " ",
                        (w.get("class") or "?"),
                        (w.get("title") or "")[:48],
                    )
                    for index, w in enumerate(open_windows[:10])
                )

            ui.add_message(
                "system",
                "Typed at her, \"my screen\" means the whole screen.\n"
                "By voice, it means the focused window (* below).\n"
                "She can also be asked for one by name:\n"
                f"{listing}\n"
                "Try: /look full | /look select | /look <name>",
            )
            return

        if argument in ("full", "active", "select"):
            region, window = argument, None
        else:
            region, window = "auto", argument

        image, detail = vision.capture(region, window=window)

        if image is None:
            ui.add_message("system", f"Screenshot failed: {detail}")
            return

        # Thrown away again - this is a check that grim works and that
        # it grabbed the right thing, not a turn.
        vision.take()

        ui.add_message("system", f"Captured {detail}. That's what she'd see.")
        return

    if text == "/mic":
        levels = speech.levels

        if levels.get("backend") == "silero":
            ui.add_message(
                "system",
                "vad=silero | speech above {:.2f}, ends below {:.2f} | "
                "last peak={:.4f} | triggered={} | captured={:.1f}s | mode={}".format(
                    levels["start"], levels["continue"], levels["peak"],
                    levels["triggered"], levels["speech_seconds"],
                    levels.get("mode", "?"),
                ),
            )
        else:
            ui.add_message(
                "system",
                "vad=energy | noise floor={:.4f} | start>{:.4f} | "
                "continue>{:.4f} | last peak={:.4f} | triggered={} | "
                "captured={:.1f}s | mode={}".format(
                    levels["noise_floor"], levels["start"], levels["continue"],
                    levels["peak"], levels["triggered"], levels["speech_seconds"],
                    levels.get("mode", "?"),
                ),
            )
        return

    if text == "/barge":
        if not speech.barge_in_available():
            ui.add_message(
                "system",
                "Barge-in is off - it needs Silero ({}), and stt.barge_in "
                "in agent.json set to true.".format(speech.vad_backend),
            )
            return

        levels = speech.levels
        ui.add_message(
            "system",
            "barge-in on | her level at your mic={:.4f} | you need "
            "{:.4f} to cut in | loudest you hit={:.4f} | best speech "
            "score={:.2f}\nToo eager: raise stt.barge_in_margin. Won't "
            "trigger: lower it, or wear headphones.".format(
                levels["barge_baseline"],
                levels["barge_baseline"] * config.STT_BARGE_IN_MARGIN,
                levels["barge_peak"],
                levels["barge_probability"],
            ),
        )
        return

    if text == "/wake":
        if not wakeword.enabled():
            ui.add_message(
                "system",
                'Wake word is off. Turn it on with a "wake_word" block in '
                "agent.json - see wakeword.py for how to get a model.",
            )
            return

        if not wakeword.available():
            ui.add_message("system", f"Wake word not loaded: {wakeword.label()}")
            return

        ui.add_message(
            "system",
            'listening for "{}" | fires above {:.2f} | best score so far '
            "{:.2f} | last frame {:.2f} | detections this session {}".format(
                wakeword.label(), config.WAKE_WORD_THRESHOLD,
                wakeword.scores["best"], wakeword.scores["last"],
                wakeword.scores["detections"],
            ),
        )
        return

    if text == "/help":
        ui.add_message(
            "system",
            f"Press Tab for the full list. Mode is {ptt.mode()} - "
            "/mode auto|manual|open to change it.",
        )
        return

    threading.Thread(target=_process, args=(text,), daemon=True).start()


ui.run(on_submit=handle_input, on_hotkey=lambda: ptt.toggle(MODEL))

shutdown()
