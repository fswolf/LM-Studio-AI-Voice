import json
import requests
import re
from datetime import datetime

from config import LM_URL, TOOLS_ENABLED, MAX_TOOL_ROUNDS
from config import agent
from config import memory
import history
import longterm
import reminders
import tools
import websearch

# Flipped off for the rest of the session the first time LM Studio
# rejects a tools payload, so a model without a tool template falls back
# to the old keyword triggers instead of erroring on every turn.
_tools_supported = TOOLS_ENABLED


def tools_active():
    return _tools_supported


def build_memory_prompt():
    prompt = ""

    if "user_preferences" in memory:
        prompt += "\nUser Preferences:\n"

        for key,value in memory["user_preferences"].items():
            prompt += f"{key}: {value}\n"

    facts = longterm.get_facts()
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
- Anything about the current date or time -> get_datetime.
- The user asking to be reminded of something -> set_reminder or
  set_reminder_at. Confirm what you scheduled afterwards.
- Anything you cannot know (news, prices, live facts) -> web_search.
  Never invent an answer you would have needed to look up.
- A durable fact about the user worth recalling weeks later ->
  remember_fact. Not passing details of this conversation.
Call a tool only when it is needed; chat normally otherwise.
"""


def build_system_prompt():
    summary = history.get_summary()
    summary_block = f"\nEarlier conversation summary:\n{summary}\n" if summary else ""

    # Gives the model a "now" anchor to compare message timestamps
    # against - without this, it only sees relative gaps between
    # messages with no idea how long ago the most recent one was.
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
You are {agent['name']}.

Current date and time: {now_str}
Each message below is prefixed with the time it was sent, e.g.
[2026-07-30 15:12:17]. Use these ONLY internally to judge how much
time has passed - never include a timestamp, brackets, or that
[YYYY-MM-DD HH:MM:SS] format anywhere in your own replies.

Personality:
{agent['personality']}

Tone:
{agent['tone']}

Traits:
{', '.join(agent['traits'])}

Rules:
{', '.join(agent['rules'])}
{build_tools_prompt()}
{build_memory_prompt()}
{summary_block}
"""


def build_generation_params():
    gen = agent.get("generation", {})
    params = {}

    max_tokens = gen.get("max_tokens")
    if max_tokens:
        params["max_tokens"] = max_tokens

    reasoning = gen.get("reasoning")
    if reasoning:
        params["reasoning"] = reasoning

    return params


def _chat_completion(payload):
    """POST to LM_URL and return the reply *message* (not just its text,
    because a tool-calling reply carries tool_calls and no content).

    Raises a RuntimeError with LM Studio's actual error message when
    the response doesn't have "choices" - e.g. context length
    exceeded, a bad/unsupported param, model not loaded, etc. Without
    this, a malformed response just raises a bare KeyError('choices')
    with no indication of what actually went wrong.
    """
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


def _format_ts(ts: str) -> str:
    """Normalize a stored timestamp to the space-separated
    'YYYY-MM-DD HH:MM:SS' format described in the system prompt.

    history.add_message() stores timestamps via .isoformat(), which
    produces a "T" between date and time (e.g. "2026-08-01T23:20:44").
    The system prompt tells the model to expect a space-separated
    format instead - that mismatch was the root cause of the model
    echoing "[2026-08-01T23:20:44]" back at the start of replies: it
    was just mimicking the only timestamp format it actually saw.
    Routing every timestamp through this one function keeps the
    prompt's description and the actual payload in sync.
    """
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def _strip_leading_timestamps(text: str) -> str:
    # Tolerates both "T" and space separators as a safety net, in case
    # a differently-formatted timestamp ever makes it into a reply.
    pattern = r"^(\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\]\s*)+"
    return re.sub(pattern, "", text).strip()


def _timestamped(role, content, timestamp):
    return {"role": role, "content": f"[{_format_ts(timestamp)}] {content}"}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


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


def _force_prose(payload):
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

    return _content(_chat_completion(payload))


def _tool_rounds(payload, model):
    """Run the model, execute any tools it asks for, repeat.

    Returns the final assistant text. Each round appends the assistant's
    tool_calls message and one `tool` message per call, which is the
    shape OpenAI-compatible servers expect on the way back in.
    """
    global _tools_supported

    for _round in range(MAX_TOOL_ROUNDS):
        message = _chat_completion(payload)
        calls = message.get("tool_calls") or []

        if not calls:
            text = _content(message)

            if text:
                return text

            # Nothing usable and nothing to call: the model has finished
            # but said nothing. One more attempt with tools removed.
            return _force_prose(payload) or ""

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

    # Out of rounds: same treatment, so a model stuck in a call loop
    # still produces something the user can read.
    return _force_prose(payload) or ""


def _log_tool_call(name, result):
    """Surface tool use in the conversation, so it's never a mystery why
    a reminder appeared or where a fact came from."""
    try:
        import ui
    except Exception:
        return

    summary = result if len(result) <= 160 else result[:157] + "..."
    ui.add_message("system", f"{name}() -> {summary}")


def ask(text, model):
    global _tools_supported

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    messages = [{"role": "system", "content": build_system_prompt()}]

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

    messages.append(_timestamped("user", user_content, now_str))

    payload = {
        "model": model,
        "messages": messages,
    }

    payload.update(build_generation_params())

    if _tools_supported:
        payload["tools"] = tools.specs()

        try:
            answer = _tool_rounds(payload, model)
        except RuntimeError as e:
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

            answer = _chat_completion(payload).get("content") or ""
    else:
        answer = _chat_completion(payload).get("content") or ""

    answer = _strip_leading_timestamps(answer)

    if not answer.strip():
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
