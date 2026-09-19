import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import threading

from config import AGENT_NAME, VOICE, KOKORO_URL
from speech import load_models, speak
from input import start_keyboard
from llm import ask
import control
import history
import ptt
import reminders
import speech
import state
import ui


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

    try:
        ui.set_status("Thinking...")
        answer = ask(text, MODEL)
        ui.add_message(AGENT_NAME.lower(), answer)

        # If HOME was pressed while we were still waiting on the model
        # (before speak() even started), don't play the response at all.
        if not state.stop_speaking:
            ui.set_status("Speaking...")
            speak(answer)

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
        history._data["messages"] = []
        history._data["summary"] = ""
        history.save()
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
        else:
            import tools as tool_registry
            ui.add_message(
                "system",
                "Tools available: " + ", ".join(tool_registry.names()),
            )
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
