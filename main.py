import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import threading

from config import AGENT_NAME, VOICE
from speech import load_models, speak
from input import start_keyboard
from llm import ask
import history
import reminders
import state
import ui

# -------------------------
# Startup
# -------------------------
ui.set_status("Starting...")

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

load_models()

ui.init(agent_name=AGENT_NAME, model=MODEL, voice=VOICE)

# Seed the on-screen conversation with what was loaded from disk, so
# past turns are visible right away instead of starting on a blank screen.
for msg in history.get_messages():
    speaker = "user" if msg["role"] == "user" else AGENT_NAME.lower()
    ui.conversation.append((speaker, msg["content"]))

ui.set_status("Idle")

# -------------------------
# Start the global hotkey listener in the background
# (HOME = push-to-talk, ESC = quit; this runs independently
#  of whatever's typed in the terminal below)
# -------------------------
threading.Thread(target=start_keyboard, args=(MODEL,), daemon=True).start()
threading.Thread(target=reminders.run_scanner, args=(MODEL,), daemon=True).start()

# -------------------------
# Typed-input loop
# -------------------------
while True:
    text = ui.prompt()

    if not text:
        continue

    if text in ("/quit", "/exit"):
        os._exit(0)
    elif text == "/clear":
        ui.conversation.clear()
        history._data["messages"] = []
        history._data["summary"] = ""
        history.save()
        continue
    elif text == "/help":
        ui.add_message("system", "Type a message, or use HOME to talk.")
        continue

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
    finally:
        state.assistant_busy = False