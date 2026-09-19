from speech import record_audio
from speech import transcribe
from speech import speak

from llm import ask
from config import AGENT_NAME

import state
import ui

def assistant_task(model, mode=None):
    state.assistant_busy = True
    ui.set_status("Listening..." if mode != "manual" else "Recording...")
    try:
        filename = record_audio(mode)

        # None means the detector never heard speech, or heard a blip too
        # short to be a sentence. Don't bother Whisper with it.
        if filename is None:
            ui.set_status("Idle")
            return

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