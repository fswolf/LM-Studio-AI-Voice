"""One turn, from text to speech.

Both input paths end here - typed lines from the TUI and spoken turns
from push-to-talk - so the two behave identically. That matters more
than it sounds: before this they each had their own copy of "ask, then
print, then speak", and the copies drifted.
"""
from config import AGENT_NAME
from llm import ask
from speech import record_audio, transcribe

import speech
import state
import ui


def respond(text, model):
    """Send a turn to the model and speak the reply as it arrives.

    Nothing here waits for the whole answer. Text appears on screen as
    it is generated and each finished sentence goes to the speaker
    immediately, so Luna starts talking about a sentence in rather than
    after the entire reply has been written.
    """
    player = speech.Player()

    ui.set_status("Thinking...")
    ui.begin_message(AGENT_NAME.lower())

    def on_sentence(sentence):
        # HOME during generation means "don't bother speaking this" -
        # the reply still finishes and stays on screen.
        if not state.stop_speaking:
            player.say(sentence)

    try:
        answer = ask(
            text, model, on_text=ui.extend_message, on_sentence=on_sentence
        )

        ui.replace_message(answer)
        ui.end_message()
        player.wait()

        return answer
    except Exception:
        ui.end_message()
        raise


def assistant_task(model, mode=None):
    """A spoken turn: record, transcribe, answer.

    Returns True if there was actually something to answer, which is
    how the hands-free loop knows whether to hold the follow-up window
    open or go back to waiting for the wake word.
    """
    state.assistant_busy = True
    ui.set_status("Listening..." if mode != "manual" else "Recording...")

    try:
        state.recording = True

        try:
            filename = record_audio(mode)
        finally:
            state.recording = False

        # Cancelled rather than finished - throw the audio away instead
        # of transcribing it and then refusing to answer.
        if state.stop_generating:
            ui.set_status("Idle")
            return False

        # None means the detector never heard speech, or heard a blip too
        # short to be a sentence. Don't bother Whisper with it.
        if filename is None:
            ui.set_status("Idle")
            return False

        ui.set_status("Transcribing...")
        text = transcribe(filename)

        if not text:
            ui.set_status("Idle")
            return False

        ui.add_message("user", text)
        state.turn_source = "voice"
        respond(text, model)
        ui.set_status("Idle")

        return True
    except Exception as e:
        ui.set_status(f"Error: {e}")

        return False
    finally:
        state.assistant_busy = False
