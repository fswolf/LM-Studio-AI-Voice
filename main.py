import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import threading

import config
from config import AGENT_NAME, VOICE, TTS_URL, LM_URL
from speech import load_models
from input import start_keyboard
import assistant
import control
import history
import lmstudio
import logbook
import ptt
import reminders
import speech
import state
import timeutil
import transcript
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
logbook.start()

if transcript.trim():
    logbook.info("transcript", "trimmed past the size cap")

history.load()
reminders.load()

MODEL = lmstudio.probe()

if not MODEL:
    print(
        f"\nCouldn't reach LM Studio at {LM_URL}\n"
        "Make sure LM Studio is open, a model is loaded, and its local "
        "server is started (Developer tab) before running this.\n"
    )
    raise SystemExit(1)


def _load_speech():
    """Whisper, Silero and the wake word, off the startup path.

    These take several seconds between them, and they used to run
    before the TUI was drawn - so every launch started with a bare
    terminal saying "Loading models..." while nothing was usable.
    Loading them behind the interface means you can read the
    conversation, scroll, and type immediately; only voice has to wait,
    and it says so if you try it early.
    """
    try:
        load_models()
        ui.set_status("Idle")
        logbook.info("startup", "speech models ready | tts=%s", speech.server_label())

        # This check belongs here, not on the startup path. load_models()
        # is what probes the TTS server, and while that was synchronous
        # the caller could read the result immediately. Moving it to a
        # worker meant main.py was asking "is the server up?" a
        # millisecond after starting the thread that finds out - so a
        # perfectly healthy server reported itself offline, with an
        # empty reason, because nothing had looked yet.
        if not speech.tts_ok:
            ui.add_message(
                "system",
                f"No TTS server answering at {TTS_URL} - start one "
                "(kokoro-reader's kokoro_server.py by default) or she'll "
                f"stay silent.{f' ({speech.tts_error})' if speech.tts_error else ''}",
            )

        if wakeword.enabled() and not wakeword.load():
            ui.add_message("system", f"Wake word unavailable: {wakeword.label()}")
    except Exception as e:
        logbook.exception("startup", "speech models failed to load")
        ui.add_message("system", f"Speech models failed to load: {e}")
        ui.set_status("No voice")


threading.Thread(target=_load_speech, daemon=True).start()

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

# Whatever was chosen last time - the setting exists so the preference
# survives a restart, not just the session.
ui.toggle_mouse(config.UI_MOUSE)

ui.set_mode(ptt.mode())
ui.set_status("Loading speech...")
ui.set_model(lmstudio.label())

# Keep the header honest between turns, so an unloaded model shows up
# straight away rather than on the next thing you say.
lmstudio.watch()

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
    state.stop_generating = False
    state.turn_source = "typed"
    # stop_speaking is deliberately not reset here - assistant.respond()
    # sets it to interrupt whatever is still talking, and clears it once
    # it actually holds the turn.

    try:
        assistant.respond(text, MODEL)
        ui.set_status("Idle")
    except Exception as e:
        # Without this, an LM Studio error (context overflow, bad param,
        # connection drop, etc.) would kill the worker silently instead
        # of just this one turn.
        logbook.exception("turn", "typed turn failed")
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

    if text.startswith("/set"):
        parts = text.split(None, 2)

        if len(parts) == 1:
            grouped = {}

            for path in sorted(config.SETTINGS):
                section = path.split(".")[0] if "." in path else ""
                grouped.setdefault(section, []).append(path)

            lines = ["Change any of these with /set <name> <value>:"]

            for section, paths in grouped.items():
                lines.append(f"  [{section or 'general'}]")

                for path in paths:
                    _name, live = config.SETTINGS[path]
                    lines.append("    {:32} {}{}".format(
                        path, config.current(path),
                        "" if live else "   (needs a restart)",
                    ))

            ui.add_message("system", "\n".join(lines))
            return

        if len(parts) == 2:
            path = parts[1]

            if path not in config.SETTINGS:
                ui.add_message("system", f"No setting called {path!r}. /set lists them.")
                return

            ui.add_message("system", f"{path} = {config.current(path)}")
            return

        path, value = parts[1], parts[2]

        try:
            applied, live = config.save_setting(path, value)
        except ValueError as e:
            ui.add_message("system", f"Couldn't set that: {e}")
            return
        except OSError as e:
            ui.add_message("system", f"Couldn't write config.json: {e}")
            return

        # stt.mode has a live switch of its own - flipping the constant
        # isn't enough, the hands-free loop has to be told.
        if path == "stt.mode":
            ptt.set_mode(applied, save=False)

        ui.add_message(
            "system",
            "{} = {}{}".format(
                path, applied,
                " - saved" if live else " - saved, takes effect on restart",
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

    if text == "/tooltest":
        ui.add_message(
            "system",
            "Running the tool-calling check - sixteen requests, a minute or "
            "two on a local model. Results appear as they finish. "
            "Nothing is scheduled or remembered.",
        )

        def check():
            import diagnose

            def show(line):
                # One message per line, as it happens - a local model
                # takes long enough that waiting for the whole report
                # looks identical to a hang.
                if line.strip():
                    ui.add_message("system", line)
                    logbook.info("diagnose", "%s", line)

            try:
                lines = diagnose.tool_calling(MODEL, report=show)

                # A clean sweep is the interesting case: the model can
                # call tools, but doesn't when it matters. The only
                # thing left between the probe and a real turn is the
                # prompt, so bisect it.
                if any("healthy here" in line for line in lines):
                    show("")
                    diagnose.prompt_bisect(MODEL, report=show)
            except Exception as e:
                logbook.exception("diagnose", "tool check failed")
                ui.add_message("system", f"Tool check failed: {e}")

        threading.Thread(target=check, daemon=True).start()
        return

    if text == "/repair":
        ui.add_message(
            "system",
            "Looking for past reminders that were set by the fallback and "
            "never recorded as tool calls. Nothing new gets scheduled and "
            "nothing she said is changed.",
        )

        def repair():
            import reminders

            def show(line):
                if line.strip():
                    ui.add_message("system", line)
                    logbook.info("repair", "%s", line)

            try:
                fixed = reminders.repair_history(MODEL, report=show)

                if fixed:
                    ui.add_message(
                        "system",
                        f"Repaired {fixed} turn{'s' if fixed != 1 else ''}. "
                        "History now shows the tool being used instead of "
                        "talked about - run /tooltest to see the difference.",
                    )
                else:
                    ui.add_message(
                        "system",
                        "Nothing to repair - every reminder turn in history "
                        "already records what it called.",
                    )
            except Exception as e:
                logbook.exception("repair", "history repair failed")
                ui.add_message("system", f"Repair failed: {e}")

        threading.Thread(target=repair, daemon=True).start()
        return

    if text.startswith("/mouse"):
        argument = text[6:].strip().lower()
        wanted = {"on": True, "true": True, "yes": True,
                  "off": False, "false": False, "no": False}.get(argument)

        on = ui.toggle_mouse(wanted)

        try:
            config.save_setting("ui.mouse", on)
            saved = " (saved)"
        except Exception:
            saved = ""

        ui.add_message(
            "system",
            ("Mouse capture on - the wheel scrolls, but text can't be "
             "selected." if on else
             "Mouse capture off - select and copy normally; PgUp/PgDn "
             "scroll.") + saved,
        )
        return

    if text.startswith("/log"):
        count = text[4:].strip()
        count = int(count) if count.isdigit() else 20

        lines = logbook.tail(min(count, 200))

        ui.add_message(
            "system",
            "{}\n{}".format(logbook.path(), "\n".join("  " + l for l in lines)),
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
