"""One turn, from text to speech.

Both input paths end here - typed lines from the TUI and spoken turns
from push-to-talk - so the two behave identically. That matters more
than it sounds: before this they each had their own copy of "ask, then
print, then speak", and the copies drifted.
"""
import threading

from config import AGENT_NAME
from llm import ask
from speech import record_audio, transcribe

import logbook
import speech
import state
import ui

# One turn at a time. Every Enter spawns its own worker, so without
# this a second message sent while she's still speaking would run
# alongside the first: two replies generating at once, both feeding the
# same audio device, and history written in whatever order they
# happened to finish.
_turn = threading.Lock()


def _note_mood(text, model):
    """Her mood hears what Ryan says. Deliberately not wired into
    respond_to_chat: a stranger in stream chat being sweet to her
    shouldn't move the same dial he does."""
    try:
        import mood

        mood.note_affection(text, model)
    except Exception:
        pass  # decoration; never the reason a turn fails


def respond(text, model):
    """Send a turn to the model and speak the reply as it arrives.

    Nothing here waits for the whole answer. Text appears on screen as
    it is generated and each finished sentence goes to the speaker
    immediately, so Luna starts talking about a sentence in rather than
    after the entire reply has been written.

    Turns are serialized. Sending a second message while she's still
    talking means you've moved on, so the previous reply stops being
    spoken - but it is still allowed to finish generating and stays on
    screen, the same bargain barge-in makes. Interrupting should cost
    you the audio, never the answer.
    """
    _note_mood(text, model)

    if _turn.locked():
        logbook.info("turn", "new message while speaking - cutting the last one short")
        state.stop_speaking = True

    with _turn:
        return _respond(text, model)


def busy():
    """Is a turn in flight? Stream chat waits rather than barging in."""
    return _turn.locked()


def respond_to_chat(prompt, model, source, who, said,
                    context=None, tools_allowed=None):
    """A turn that came from stream chat rather than from Ryan.

    Same speaking path - answering out loud is the whole point on a
    stream - but it waits for the lock instead of cutting the current
    reply short. Ryan interrupting her is him changing his mind; a
    viewer interrupting her is a stranger talking over him.

    The prompt, the context and the tool list are built by pomf.py, and
    nothing here is stored: remember=False keeps a stranger's words out
    of history and out of the transcript.
    """
    with _turn:
        state.stop_speaking = False

        player = speech.Player()

        # What they said, not just that somebody said something. The
        # first version showed only the name, which made the one thing
        # you actually want to read - the question she is about to
        # answer on stream - the one thing that wasn't on screen.
        ui.add_message(f"chat:{source}", f"{who}: {said}")
        ui.set_status("Chat...")
        ui.begin_message(AGENT_NAME.lower())

        def on_sentence(sentence):
            if not state.stop_speaking:
                player.say(sentence)

        try:
            answer = ask(
                prompt, model,
                on_text=ui.extend_message, on_sentence=on_sentence,
                context=context, tools_allowed=tools_allowed, remember=False,
            )

            ui.replace_message(answer)
            ui.end_message()
            player.wait()

            return answer
        except Exception:
            ui.end_message()
            raise
        finally:
            ui.set_status("Idle")


def _respond(text, model):
    # Only now, holding the lock: resetting this any earlier would
    # clear the stop we just set on the turn we're waiting for.
    state.stop_speaking = False

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
    if not speech.ready():
        ui.add_message("system", "Still loading the speech models - one moment.")
        ui.set_status("Loading...")

        return False

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

        logbook.info("turn", "heard: %s", text[:200])
        ui.add_message("user", text)
        state.turn_source = "voice"
        respond(text, model)
        ui.set_status("Idle")

        return True
    except Exception as e:
        logbook.exception("turn", "spoken turn failed")
        ui.set_status(f"Error: {e}")

        return False
    finally:
        state.assistant_busy = False
