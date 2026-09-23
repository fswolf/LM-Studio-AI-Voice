"""Does this model actually call tools, and which ones?

Written because Luna kept saying "Got it, setting that for you!" while
calling nothing at all - and the logs could prove she hadn't, but not
say why. Guessing at that costs a restart and a conversation each time.
This asks the model directly and reports what came back.

It exists as much for the next model as this one. Swapping in a bigger
Qwen, or anything else, raises the same question - does tool calling
work here, and does it work for tools that take arguments? - and this
answers it in about twenty seconds without touching the conversation.

Each probe is run twice, streamed and blocking, because that is the
difference most likely to break: tool calls arrive whole in a blocking
response and in fragments when streamed, and a template that handles
one can mishandle the other. If a tool works blocking and fails
streamed, the fault is in the plumbing here. If it fails both ways, the
model simply isn't choosing it.
"""
import json
import time

import requests

from config import LM_URL

import tools

# Deliberately blunt phrasings - if a model won't call set_reminder for
# "remind me in 5 minutes to eat", no amount of prompt tuning will save
# it. Each names the tool that should fire.
PROBES = (
    ("remind me in 5 minutes to eat chocolate", "set_reminder"),
    ("set a reminder for tomorrow at 9 to call the bank", "set_reminder"),
    ("what reminders do I have?", "list_reminders"),
    ("what time is it?", "get_datetime"),
    ("remember that I stream on Tuesdays", "remember_fact"),
    # Every tool added makes the choice harder, so the ones most likely
    # to be confused with something else get a probe of their own.
    # "turn the music down" has to beat set_volume against six other
    # verbs in one enum, and "how much VRAM is free" is the one a model
    # will happily answer from thin air if it doesn't reach for a tool.
    ("turn the music down a bit", "control_audio"),
    ("how much VRAM have I got free?", "system_status"),
    ("what's on my clipboard?", "clipboard"),
    # set_alarm and set_reminder describe near-identical jobs, so this
    # probe is really testing the pair: it passes only if "wake me" goes
    # to the alarm while "remind me" above still goes to the reminder.
    ("wake me up at 7 tomorrow", "set_alarm"),
)

INSTRUCTION = (
    "You are a helpful assistant with tools. When a tool fits the "
    "request, call it. Do not describe calling it."
)


# Enough room for a reasoning model to think its way to a tool call,
# but not enough to write an essay. Without a cap, every probe that
# *doesn't* call a tool generates a full chatty reply instead, and ten
# of those turn a twenty-second check into several minutes of silence.
PROBE_TOKENS = 400
PROBE_TIMEOUT = 60


def _blocking(model, text, specs):
    response = requests.post(LM_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": text},
        ],
        "tools": specs,
        "max_tokens": PROBE_TOKENS,
    }, timeout=PROBE_TIMEOUT)

    data = response.json()

    if "choices" not in data:
        return {"error": str(data.get("error", data))[:120]}

    message = data["choices"][0].get("message") or {}

    return {
        "calls": [
            (c.get("function", {}).get("name", "?"),
             c.get("function", {}).get("arguments", ""))
            for c in (message.get("tool_calls") or [])
        ],
        "content": (message.get("content") or "")[:120],
        "reasoning": bool(message.get("reasoning_content")
                          or message.get("reasoning")),
    }


def _streamed(model, text, specs):
    calls = {}
    content = ""
    reasoning = False

    with requests.post(LM_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": text},
        ],
        "tools": specs,
        "max_tokens": PROBE_TOKENS,
        "stream": True,
    }, stream=True, timeout=PROBE_TIMEOUT) as response:
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:100]}"}

        for raw in response.iter_lines():
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

            if not line or not line.startswith("data:"):
                continue

            body = line[5:].strip()

            if body == "[DONE]":
                break

            try:
                delta = (json.loads(body).get("choices") or [{}])[0].get("delta") or {}
            except json.JSONDecodeError:
                continue

            content += delta.get("content") or ""
            reasoning = reasoning or bool(delta.get("reasoning_content")
                                          or delta.get("reasoning"))

            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), ["", ""])
                function = call.get("function") or {}
                slot[0] += function.get("name") or ""
                slot[1] += function.get("arguments") or ""

    return {
        "calls": [tuple(calls[k]) for k in sorted(calls)],
        "content": content[:120],
        "reasoning": reasoning,
    }


def _verdict(result, wanted):
    if "error" in result:
        return "error", result["error"]

    names = [name for name, _ in result["calls"]]

    if wanted in names:
        _name, arguments = next(c for c in result["calls"] if c[0] == wanted)

        if not arguments.strip():
            return "no args", "called it with nothing"

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return "bad json", arguments[:60]

        return "ok", ", ".join(f"{k}={v!r}" for k, v in parsed.items())[:70]

    if names:
        return "wrong tool", ", ".join(names)

    if result["reasoning"] and not result["content"]:
        return "thought only", "reasoning, then nothing"

    return "just talked", result["content"][:70] or "(nothing at all)"


def tool_calling(model, report=None):
    """Run the probes.

    `report` is called with each line as it is produced. A local model
    answering ten prompts takes long enough that a single block of
    output at the end is indistinguishable from a hang - which is
    exactly what it looked like the first time this ran.
    """
    specs = tools.specs()
    lines = []

    def say(line=""):
        lines.append(line)

        if report:
            report(line)

    say(f"Tool-calling check against {model}")
    say(f"{len(specs)} tools offered, {len(PROBES)} probes, two ways each.")
    say()

    scores = {"stream": 0, "block": 0}

    for number, (text, wanted) in enumerate(PROBES, 1):
        say(f"  [{number}/{len(PROBES)}] \"{text}\"")
        say(f"    expecting {wanted}")

        for label, runner in (("streamed", _streamed), ("blocking", _blocking)):
            started = time.monotonic()

            try:
                result = runner(model, text, specs)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"[:110]}

            verdict, detail = _verdict(result, wanted)

            if verdict == "ok":
                scores["stream" if label == "streamed" else "block"] += 1

            say(f"    {label:9} {verdict:12} {detail}"
                f"   ({time.monotonic() - started:.1f}s)")

        say()

    total = len(PROBES)
    say(f"  streamed {scores['stream']}/{total}   "
        f"blocking {scores['block']}/{total}")

    # The comparison is the point - it says whose fault it is.
    if scores["block"] > scores["stream"]:
        say()
        say("  Tools work better blocking than streamed, so the "
                     "fault is in how")
        lines.append("  this app reassembles streamed tool calls, not in "
                     "the model.")
    elif scores["stream"] == scores["block"] == 0:
        lines.append("")
        lines.append("  This model isn't choosing tools at all. Try one "
                     "with a tool")
        lines.append("  template, or turn tools off and let the keyword "
                     "fallbacks work.")
    elif scores["stream"] == total:
        lines.append("")
        lines.append("  Tool calling is healthy here.")

    return lines


# ---------------------------------------------------------------------------
# Why a model that passes the probes still doesn't call tools in practice
#
# The probes send two sentences. A real turn sends Luna's personality,
# her tone and traits and rules, the tools guidance, whatever she has
# remembered, a running summary, timestamped history, and whatever
# generation parameters config.json specifies.
#
# When the probes pass 5/5 and reminders still fail, the difference is
# in that gap. So: run the same probe with each layer added back, one
# at a time, and see which one it stops working at. Nothing here is a
# guess - the layers are built with the same functions the real turn
# uses, so a failure is the real failure.
# ---------------------------------------------------------------------------
BISECT_PROBE = ("remind me in 5 minutes to eat chocolate", "set_reminder")


def _layers():
    """Cumulative prompt layers, bare to real. Returns (label, messages,
    extra_params) tuples."""
    import config
    import history
    import llm

    text, _wanted = BISECT_PROBE
    persona = "\n".join(filter(None, [
        f"You are {config.AGENT_NAME}.",
        f"\nPersonality:\n{config.PERSONALITY}" if config.PERSONALITY else "",
        f"\nTone:\n{config.TONE}" if config.TONE else "",
        f"\nTraits:\n{', '.join(config.TRAITS)}" if config.TRAITS else "",
        f"\nRules:\n{', '.join(config.RULES)}" if config.RULES else "",
    ]))

    user = {"role": "user", "content": text}

    layers = [
        ("bare instruction", [{"role": "system", "content": INSTRUCTION}, user], {}),
        ("+ her personality", [{"role": "system", "content": persona}, user], {}),
        ("+ tools guidance",
         [{"role": "system", "content": persona + llm.build_tools_prompt()}, user], {}),
        ("+ remembered facts",
         [{"role": "system", "content": persona + llm.build_tools_prompt()
           + llm.build_memory_prompt(text)}, user], {}),
        ("+ the real system prompt",
         [{"role": "system", "content": llm.build_system_prompt(text)}, user], {}),
        ("+ generation settings",
         [{"role": "system", "content": llm.build_system_prompt(text)}, user],
         llm.build_generation_params()),
    ]

    # And finally the whole thing, history included, exactly as ask()
    # assembles it.
    # Built with llm's own helpers, not a copy of them. A diagnostic
    # that drifts from the code it diagnoses is worse than no
    # diagnostic - it was already reporting on a history format ask()
    # had stopped using.
    past = history.get_messages_full()
    messages = [{
        "role": "system",
        "content": llm.build_system_prompt(text, llm._timing_note(past)),
    }]

    replayed = []

    for message in past:
        replayed.extend(llm._replay(message))

    layers.append((
        "+ conversation history",
        messages + replayed + [user],
        llm.build_generation_params(),
    ))

    # The layer above is history as it stands; this one is what ask()
    # actually sends. The two differ only when history holds no tool
    # call of its own - which is exactly the case that was breaking
    # tool calling - so seeing them side by side says whether the
    # worked example is doing its job or whether something else is
    # wrong.
    if not llm._has_tool_example(replayed):
        example = llm._demonstration()

        if example:
            layers.append((
                "+ worked tool example",
                messages + example + replayed + [user],
                llm.build_generation_params(),
            ))

    return layers


def _unrecorded_claims():
    """How many stored turns answered a reminder request without calling
    anything. Counted rather than guessed at, because "history breaks
    it" and "these five turns break it" call for different fixes."""
    import history
    import reminders

    messages = history.get_messages_full()
    count = 0

    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or message.get("tools"):
            continue

        if index == 0 or messages[index - 1].get("role") != "user":
            continue

        asked = str(messages[index - 1].get("content", ""))

        if asked.strip().lower().startswith(reminders._DELIVERY):
            continue

        if reminders.looks_like_reminder(asked):
            count += 1

    return count


def _probe_with(model, messages, specs, extra):
    payload = dict({
        "model": model,
        "messages": messages,
        "tools": specs,
        "max_tokens": PROBE_TOKENS,
        "stream": True,
    }, **{k: v for k, v in extra.items() if k != "max_tokens"})

    calls = {}
    content = ""

    with requests.post(LM_URL, json=payload, stream=True,
                       timeout=PROBE_TIMEOUT) as response:
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}"}

        for raw in response.iter_lines():
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

            if not line or not line.startswith("data:"):
                continue

            body = line[5:].strip()

            if body == "[DONE]":
                break

            try:
                delta = (json.loads(body).get("choices") or [{}])[0].get("delta") or {}
            except json.JSONDecodeError:
                continue

            content += delta.get("content") or ""

            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), ["", ""])
                function = call.get("function") or {}
                slot[0] += function.get("name") or ""
                slot[1] += function.get("arguments") or ""

    return {"calls": [tuple(calls[k]) for k in sorted(calls)],
            "content": content[:120], "reasoning": False}


def prompt_bisect(model, report=None):
    """Add the real prompt back one layer at a time; find where it breaks."""
    specs = tools.specs()
    lines = []

    def say(line=""):
        lines.append(line)

        if report:
            report(line)

    _text, wanted = BISECT_PROBE

    say("Same request, with the real prompt rebuilt one layer at a time:")
    say(f"  \"{_text}\" -> {wanted}")
    say()

    failures = []
    layers = _layers()

    for label, messages, extra in layers:
        size = sum(len(str(m.get("content", ""))) for m in messages)
        started = time.monotonic()

        try:
            result = _probe_with(model, messages, specs, extra)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"[:90]}

        verdict, detail = _verdict(result, wanted)
        say(f"  {label:26} {size:6}ch  {verdict:12} {detail[:46]}"
            f"  ({time.monotonic() - started:.1f}s)")

        if verdict != "ok":
            failures.append(label)

    say()

    last_layer = layers[-1][0]

    if not failures:
        say("  It called the tool at every layer, so the prompt is fine.")
        say("  If reminders still fail in conversation, the difference is")
        say("  in what was actually said - check /log at the time.")
    elif last_layer in failures:
        # The full prompt is what real turns use, so its verdict is the
        # one that counts. Anything failing before it and recovering is
        # a marginal layer, not the cause - tool calling is sampled, so
        # a borderline prompt fails some of the time and passes the rest.
        say(f"  The full prompt stops it calling the tool.")

        marginal = [f for f in failures if f != last_layer]

        if marginal:
            say(f"  Also unreliable on its own: {', '.join(marginal)}.")
            say("  Those recovered at later layers, so they are marginal")
            say("  rather than broken - the sampling goes either way.")

        if "+ conversation history" in failures:
            claims = _unrecorded_claims()

            if claims:
                say("")
                say(f"  {claims} turn{'s' if claims != 1 else ''} in history "
                    "answered a reminder request in prose")
                say("  and recorded no tool call. Those are worked examples of")
                say("  the exact failure, and they outvote the prompt. /repair")
                say("  rewrites them as the calls they really were.")
    elif last_layer == "+ worked tool example" and failures == ["+ conversation history"]:
        # Not sampling noise - this is the shape the fix was written
        # for. History full of prose and nothing else suppresses tool
        # calling; one worked example ahead of it restores it.
        say("  History on its own stopped it calling the tool, and the")
        say("  worked example put it back. That is the example doing its")
        say("  job - it drops out by itself once a real tool call lands")
        say("  in history.")
    else:
        say(f"  Unreliable at: {', '.join(failures)} - but the full prompt")
        say("  worked. Tool calling is sampled, so a borderline prompt")
        say("  fails some of the time. Run it again to see if it moves.")

    return lines
