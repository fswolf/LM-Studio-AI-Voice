import json
import requests
import re

import state
from datetime import datetime

from config import LM_URL, TOOLS_ENABLED, MAX_TOOL_ROUNDS
from config import AGENT_NAME, PERSONALITY, TONE, TRAITS, RULES, GENERATION
from config import memory
import history
import longterm
import reminders
import timeutil
import tools
import vision
import websearch

# Flipped off for the rest of the session the first time LM Studio
# rejects a tools payload, so a model without a tool template falls back
# to the old keyword triggers instead of erroring on every turn.
_tools_supported = TOOLS_ENABLED


def tools_active():
    return _tools_supported


def build_memory_prompt(query=""):
    """Preferences, plus the remembered facts worth showing this turn.

    `query` is what the user just said. Under the context cap every
    fact is sent regardless; over it, the ones sharing vocabulary with
    the turn are, so a long memory doesn't turn every prompt into a
    recital of everything she's ever been told.
    """
    prompt = ""

    if "user_preferences" in memory:
        prompt += "\nUser Preferences:\n"

        for key, value in memory["user_preferences"].items():
            prompt += f"{key}: {value}\n"

    facts = longterm.relevant_facts(query)

    if facts:
        prompt += "\nThings learned over time:\n"

        for fact in facts:
            prompt += f"- {fact}\n"

    return prompt


def build_tools_prompt():
    """A short nudge about the tools. The schemas are sent separately in
    the payload; this is about *when* to reach for them, which schemas
    don't convey well to smaller models."""
    if not _tools_supported:
        return ""

    return """
You have tools. Use them instead of guessing or promising:
- What day or time it is now -> get_datetime.
- How far away a date is, or what date something falls on ->
  time_until. Do not count days or convert units in your head.
- The user asking to be reminded of something -> set_reminder. Pass
  their timing words through unchanged ("tomorrow at 9", "in ten
  minutes", "every monday"); the app works out the actual time.
  Confirm what you scheduled afterwards.
- Anything you cannot know (news, prices, live facts) -> web_search.
  Never invent an answer you would have needed to look up.
- A durable fact about the user worth recalling weeks later ->
  remember_fact. Not passing details of this conversation.
- The user correcting something you remembered, or asking you to
  forget it -> update_fact or forget_fact. Don't just agree; change it.
Call a tool only when it is needed; chat normally otherwise.
"""


def build_system_prompt(query=""):
    summary = history.get_summary()
    summary_block = f"\nEarlier conversation summary:\n{summary}\n" if summary else ""

    # A "now" anchor, plus enough calendar context that the model never
    # has to derive a date. It used to get a bare ISO timestamp and the
    # instruction to work out elapsed time from other ISO timestamps -
    # arithmetic a 9B model fails quietly and confidently.
    return f"""
You are {AGENT_NAME}.

Right now:
{timeutil.describe_now()}

Each message below is prefixed with when it was sent, already worked
out for you - e.g. [yesterday 21:04, 17 hours ago]. Read those; never
calculate a date or a duration yourself, and never repeat a timestamp
or the brackets in your own replies. If you need a date the prefixes
do not give you, call get_datetime or time_until.

Personality:
{PERSONALITY}

Tone:
{TONE}

Traits:
{', '.join(TRAITS)}

Rules:
{', '.join(RULES)}
{build_tools_prompt()}
{build_memory_prompt(query)}
{summary_block}
"""


def build_generation_params():
    gen = GENERATION
    params = {}

    max_tokens = gen.get("max_tokens")
    if max_tokens:
        params["max_tokens"] = max_tokens

    reasoning = gen.get("reasoning")
    if reasoning:
        params["reasoning"] = reasoning

    return params


def _chat_completion(payload, narrator=None):
    """POST to LM_URL and return the reply *message* (not just its text,
    because a tool-calling reply carries tool_calls and no content).

    With a narrator, the reply is streamed and fed to it token by
    token; without one this is the plain blocking request it always
    was. Both return the same message shape.

    Raises a RuntimeError with LM Studio's actual error message when
    the response doesn't have "choices" - e.g. context length
    exceeded, a bad/unsupported param, model not loaded, etc. Without
    this, a malformed response just raises a bare KeyError('choices')
    with no indication of what actually went wrong.
    """
    if narrator is not None:
        return _streamed_message(payload, narrator)

    response = requests.post(LM_URL, json=payload)

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"LM Studio returned a non-JSON response (HTTP {response.status_code}): "
            f"{response.text[:300]}"
        )

    if "choices" not in data:
        detail = data.get("error", data)
        print(f"[llm] LM Studio response missing 'choices': {detail}")
        raise RuntimeError(f"LM Studio error: {detail}")

    return data["choices"][0]["message"]


def _format_ts(when):
    """A stored timestamp as the prefix the model actually reads:
    'today 14:32, 17 minutes ago' rather than '2026-09-19T14:32:07'.

    The model used to be handed ISO timestamps and told to work out
    elapsed time from them. It was bad at it - confidently bad, calling
    yesterday "a few minutes ago" - and there was never a reason for it
    to try, since Python knows the answer exactly. Doing the
    subtraction here removes the failure and costs fewer tokens than
    the timestamp it replaces.
    """
    if isinstance(when, datetime):
        return timeutil.stamp(when)

    try:
        return timeutil.stamp(datetime.fromisoformat(str(when)))
    except (ValueError, TypeError):
        return str(when)


def _strip_leading_timestamps(text):
    """Models mimic whatever format they can see, so some of them open
    a reply with the prefix. Both forms are stripped: conversations
    saved before this change still carry the ISO one."""
    for pattern in (
        r"^(\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\]\s*)+",
        r"^(\[(?:today|tomorrow|yesterday|last \w+|\w{3} \d)[^\]]{0,48}\]\s*)+",
    ):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def _timestamped(role, content, timestamp):
    return {"role": role, "content": f"[{_format_ts(timestamp)}] {content}"}


# What the last completion actually contained, so an empty reply can be
# explained instead of apologised for.
_last_raw = {"deltas": 0, "content": 0, "reasoning": 0, "tool_rounds": 0}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# A sentence ends at .!?… plus any closing quote or bracket, followed by
# whitespace. Requiring the whitespace is what stops "3.5" and "e.g."
# being spoken as two sentences.
_SENTENCE_END = re.compile(r"[.!?…]['\"\u201d\u2019)\]]*\s")

# The shortest fragment worth sending to the TTS server on its own.
# Below this it's an interjection, and the pause around it sounds worse
# than waiting for the rest of the line - so it rides along with the
# sentence after it instead.
MIN_SPOKEN_CHARS = 16

# Words that end in a full stop without ending a sentence. Any single
# letter counts too, which is what catches "e.g.", "i.e.", "a.m." and
# "U.S." without having to list them.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "approx", "est", "fig", "dept", "inc", "ltd", "no", "pp", "al",
}


def _ends_sentence(text, stop):
    """Is the terminator at `stop` a real sentence ending?

    Only full stops are ever in doubt - nothing abbreviates with ? or !
    """
    if text[stop] != ".":
        return True

    word = re.search(r"([A-Za-z]+)$", text[:stop])

    if not word:
        return True

    word = word.group(1)

    return len(word) > 1 and word.lower() not in _ABBREVIATIONS


def _visible(raw):
    """The part of a partially-streamed reply that is safe to show and
    to speak.

    Reasoning models open with <think>, and none of that should ever
    reach the speaker. Complete blocks are cut; an unclosed one means
    everything after it is still scratch work, so it's withheld until
    the closing tag arrives. A trailing "<thi" is held back too, since
    the next token may finish the tag.
    """
    text = _THINK_BLOCK.sub("", raw)

    if "<think>" in text.lower():
        text = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)[0]

    return re.sub(r"<[a-z/]{0,7}$", "", text, flags=re.IGNORECASE)


class _Narrator:
    """Turns a token stream into two things at once: visible text for
    the screen, and complete sentences for the voice.

    The voice is why this exists. Waiting for the whole reply before
    synthesizing anything meant several seconds of silence on every
    turn; handing Kokoro each sentence as it finishes drops that to
    roughly one sentence's worth.
    """

    def __init__(self, on_text=None, on_sentence=None):
        self._on_text = on_text
        self._on_sentence = on_sentence
        self.raw = ""
        self._shown = 0
        self._spoken = 0

    def feed(self, piece):
        if not piece:
            return

        self.raw += piece
        text = _visible(self.raw)

        if self._on_text and len(text) > self._shown:
            self._on_text(text[self._shown:])
            self._shown = len(text)

        if not self._on_sentence:
            return

        cursor = self._spoken

        while True:
            match = _SENTENCE_END.search(text, cursor)

            if not match:
                break

            # Keep looking past a false ending ("e.g.") and past a
            # fragment too short to be worth speaking on its own
            # ("Hey there.") - both should join the sentence after them.
            if (not _ends_sentence(text, match.start())
                    or match.end() - self._spoken < MIN_SPOKEN_CHARS):
                cursor = match.end()
                continue

            sentence = text[self._spoken:match.end()].strip()
            self._spoken = cursor = match.end()

            if sentence:
                self._on_sentence(sentence)

    def finish(self):
        """Flush whatever didn't end in punctuation."""
        text = _visible(self.raw)

        if self._on_text and len(text) > self._shown:
            self._on_text(text[self._shown:])
            self._shown = len(text)

        if self._on_sentence:
            tail = text[self._spoken:].strip()
            self._spoken = len(text)

            if tail:
                self._on_sentence(tail)

        return text.strip()


def _stream_deltas(payload):
    """Yield delta dicts from an OpenAI-style SSE response.

    The bytes are decoded here rather than by requests. requests falls
    back to ISO-8859-1 for any text/* response that doesn't declare a
    charset, and LM Studio's text/event-stream doesn't declare one - so
    decode_unicode=True turns every emoji she streams into mojibake
    ("ð\x9f\x92\x9c" instead of a heart). Splitting on newlines first
    is safe because no byte of a multi-byte UTF-8 sequence is ever 0x0A.
    """
    with requests.post(
        LM_URL, json=dict(payload, stream=True), stream=True, timeout=600
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"LM Studio error (HTTP {response.status_code}): "
                f"{response.text[:300]}"
            )

        for raw in response.iter_lines():
            line = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, bytes) else raw
            )

            if not line or not line.startswith("data:"):
                continue

            body = line[5:].strip()

            if body == "[DONE]":
                return

            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []

            if choices:
                yield choices[0].get("delta") or {}


def _streamed_message(payload, narrator):
    """Consume a streamed completion into the same message shape the
    non-streaming path returns, so _tool_rounds can't tell the
    difference.

    Tool calls arrive in pieces too - the name in one delta and the
    arguments spread over many - and have to be reassembled by index.
    """
    calls = {}
    _last_raw.update({"deltas": 0, "content": 0, "reasoning": 0})

    for delta in _stream_deltas(payload):
        _last_raw["deltas"] += 1

        # Some servers stream a reasoning model's scratchpad in its own
        # field rather than inside <think> tags. It is never spoken, but
        # knowing it arrived is the difference between "the model said
        # nothing" and "the model thought and never concluded".
        if delta.get("reasoning_content") or delta.get("reasoning"):
            _last_raw["reasoning"] += 1

        if delta.get("content"):
            _last_raw["content"] += 1

        if state.stop_generating:
            # HOME was pressed - stop pulling tokens rather than finish
            # a reply nobody is going to hear. Deliberately not
            # stop_speaking: that also fires on a barge-in, and a false
            # barge-in should cost you the audio, never the answer.
            break

        narrator.feed(delta.get("content") or "")

        for call in delta.get("tool_calls") or []:
            slot = calls.setdefault(call.get("index", 0), {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })

            if call.get("id"):
                slot["id"] = call["id"]

            function = call.get("function") or {}

            if function.get("name"):
                slot["function"]["name"] += function["name"]

            if function.get("arguments"):
                slot["function"]["arguments"] += function["arguments"]

    message = {"role": "assistant", "content": narrator.raw}

    if calls:
        message["tool_calls"] = [calls[key] for key in sorted(calls)]

    return message




def _content(message):
    """The usable text of a reply.

    Three things get in the way on local models:
      * reasoning models wrap their scratchpad in <think>...</think>, and
        after a tool result the whole reply is sometimes *only* that -
        which read as an empty answer and, worse, got spoken aloud when
        it wasn't;
      * an unterminated <think> when the model runs out of tokens;
      * some servers return content as a list of blocks, not a string.
    """
    content = message.get("content")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )

    content = content or ""
    content = _THINK_BLOCK.sub("", content)

    # Unclosed block: keep whatever came before it, drop the rest.
    if "<think>" in content:
        content = content.split("<think>", 1)[0]

    return content.strip()


def _force_prose(payload, on_text=None, on_sentence=None):
    """Ask again with no tools offered.

    After a tool result some models keep trying to call something, or
    answer with nothing at all. Removing the tools takes that option
    away, so they have to produce words.
    """
    payload = dict(payload)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["messages"] = payload["messages"] + [{
        "role": "user",
        "content": "(Tell me what you just did, in one short sentence. "
                   "Do not call any more tools.)",
    }]

    if on_text or on_sentence:
        narrator = _Narrator(on_text, on_sentence)
        _chat_completion(payload, narrator)

        return narrator.finish()

    return _content(_chat_completion(payload))


def _tool_rounds(payload, model, on_text=None, on_sentence=None):
    """Run the model, execute any tools it asks for, repeat.

    Returns the final assistant text. Each round appends the assistant's
    tool_calls message and one `tool` message per call, which is the
    shape OpenAI-compatible servers expect on the way back in.

    Every round is streamed, including the ones that turn out to be
    tool calls - those simply produce no prose, so nothing is spoken.
    The round that finally answers starts talking while it is still
    being generated.
    """
    global _tools_supported

    streaming = bool(on_text or on_sentence)

    # A round can produce prose *and* a tool call ("let me check that
    # for you" followed by get_datetime). That preamble is spoken and
    # shown as it arrives, so it has to end up in the returned text
    # too, or replace_message would wipe it off the screen.
    said = []

    for _round in range(MAX_TOOL_ROUNDS):
        _last_raw["tool_rounds"] = _round + 1
        narrator = _Narrator(on_text, on_sentence)
        message = _chat_completion(payload, narrator)
        calls = message.get("tool_calls") or []
        text = narrator.finish() if streaming else _content(message)

        if text:
            said.append(text)

        if not calls:
            if said:
                return " ".join(said).strip()

            # Nothing usable and nothing to call: the model has finished
            # but said nothing. One more attempt with tools removed.
            return _force_prose(payload, on_text, on_sentence) or ""

        # Echo the assistant turn back verbatim - dropping it breaks the
        # pairing between tool_call_id and result.
        payload["messages"].append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:
            function = call.get("function", {})
            name = function.get("name", "")
            result = tools.call(name, function.get("arguments", "{}"))

            _log_tool_call(name, result)

            payload["messages"].append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result,
            })

        # look_at_screen has no way to put an image in a tool result -
        # the field is a string - so it stashes the capture and we hand
        # it over here as a user turn instead. A text-only model will
        # reject this payload, and the error says so plainly rather than
        # us pretending she looked.
        image = vision.take()

        if image:
            payload["messages"].append({
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "(Here is the screenshot you asked for.)"},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            })

    # Out of rounds: same treatment, so a model stuck in a call loop
    # still produces something the user can read.
    if said:
        return " ".join(said).strip()

    return _force_prose(payload, on_text, on_sentence) or ""


def _log_tool_call(name, result):
    """Surface tool use in the conversation, so it's never a mystery why
    a reminder appeared or where a fact came from."""
    try:
        import ui
    except Exception:
        return

    summary = result if len(result) <= 160 else result[:157] + "..."
    ui.add_message("system", f"{name}() -> {summary}")


def _why_empty():
    """Best explanation for a reply that came back with no words in it."""
    if state.stop_generating:
        return "you interrupted it, so generation was cut short"

    if state.stop_speaking:
        return (
            "playback was stopped mid-reply - if you didn't press HOME, "
            "barge-in is firing on her own voice. Check /barge and raise "
            "stt.barge_in_margin"
        )

    if _last_raw.get("reasoning") and not _last_raw.get("content"):
        return (
            "the model produced only reasoning and no answer - lower "
            "generation.reasoning in agent.json, or raise max_tokens"
        )

    if _last_raw.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return (
            f"it called tools {MAX_TOOL_ROUNDS} times without answering - "
            "raise tools.max_rounds, or the model is stuck in a loop"
        )

    if not _last_raw.get("deltas"):
        return "LM Studio streamed nothing at all - is the model still loaded?"

    return "the model returned no usable text"


def _generate(payload, on_text=None, on_sentence=None):
    """One completion, streamed or not depending on who's listening."""
    if on_text or on_sentence:
        narrator = _Narrator(on_text, on_sentence)
        _chat_completion(payload, narrator)

        return narrator.finish()

    return _content(_chat_completion(payload))


def ask(text, model, on_text=None, on_sentence=None):
    """One turn. Returns the finished reply.

    on_text receives visible text as it streams, for the screen.
    on_sentence receives each complete sentence, for the voice. Pass
    neither and this is the blocking request it always was, which is
    what the reminder scanner wants - it has no screen to draw on.
    """
    global _tools_supported

    now = datetime.now()

    messages = [{"role": "system", "content": build_system_prompt(text)}]

    # get_messages_full() keeps each message's stored timestamp, so
    # the model can see how much time passed between past turns too,
    # not just how old the newest message is.
    for m in history.get_messages_full():
        messages.append(_timestamped(m["role"], m["content"], m.get("timestamp", "unknown time")))

    # Folded into the *user* turn's content rather than a second
    # "system" message - some chat templates (Jinja-based, incl. this
    # one) hard-require system to be the first and only system-role
    # message, and raise a template error if another one shows up
    # anywhere else in the list.
    #
    # Only needed when tools are unavailable: with tools, the model calls
    # web_search itself instead of needing the phrase "web search" in
    # your sentence.
    user_content = text

    if not _tools_supported and websearch.looks_like_search(text):
        query = websearch.extract_query(text)
        results = websearch.search(query)
        search_block = websearch.format_results(query, results)
        user_content = f"{search_block}\n\n{text}"

    messages.append(_timestamped("user", user_content, now))

    payload = {
        "model": model,
        "messages": messages,
    }

    payload.update(build_generation_params())

    if _tools_supported:
        payload["tools"] = tools.specs()

        try:
            answer = _tool_rounds(payload, model, on_text, on_sentence)
        except RuntimeError as e:
            if "image" in str(e).lower() or "vision" in str(e).lower():
                raise RuntimeError(
                    "This model can't accept images - load a vision model "
                    f"in LM Studio, or turn vision off in agent.json. ({e})"
                ) from e

            # Most likely this model has no tool template. Drop tools for
            # the rest of the session and retry the same turn without
            # them, so the user sees an answer rather than an error.
            if "tool" not in str(e).lower():
                raise

            _tools_supported = False
            payload.pop("tools", None)
            payload["messages"] = messages

            _log_tool_call(
                "tools", "unsupported by this model - using keyword triggers"
            )

            answer = _generate(payload, on_text, on_sentence)
    else:
        answer = _generate(payload, on_text, on_sentence)

    answer = _strip_leading_timestamps(answer)

    if not answer.strip():
        # This used to be a bare apology, which told nobody anything.
        # An empty reply has a small number of distinct causes and the
        # app knows which one it hit, so say so.
        _log_tool_call("empty reply", _why_empty())
        answer = "...sorry, I got tangled up there. Say that again?"

    # Stored content itself stays clean (no timestamp prefix baked in) -
    # the prefix above is only added when building the API payload, so
    # get_messages_full()'s own "timestamp" field stays the single
    # source of truth and formatting can change later without rewriting
    # anything already saved to disk.
    history.add_message("user", text)
    history.add_message("assistant", answer)
    history.maybe_summarize(model)
    history.save()

    # With tools available the model calls remember_fact and the reminder
    # tools itself, so the keyword extractors would only duplicate work -
    # and double-schedule reminders.
    if not _tools_supported:
        longterm.extract_in_background(model, text, answer)
        reminders.extract_in_background(model, text)

    return answer
