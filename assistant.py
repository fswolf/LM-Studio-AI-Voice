from speech import record_audio
from speech import transcribe
from speech import speak

from llm import ask
from config import AGENT_NAME

import state
import ui

def assistant_task(model):
    state.assistant_busy = True
    ui.set_status("Listening...")
    try:
        filename = record_audio()

        ui.set_status("Transcribing...")
        text = transcribe(filename)

        if not text:
            ui.set_status("Idle")
            return

        ui.add_message("user", text)

        ui.set_status("Thinking...")
        answer = ask(text, model)
        ui.add_message(AGENT_NAME.lower(), answer)

        ui.set_status("Speaking...")
        speak(answer)

        ui.set_status("Idle")

    except Exception as e:
        ui.set_status(f"Error: {e}")
    finally:
        state.assistant_busy = False