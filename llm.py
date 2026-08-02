import requests
import re
from datetime import datetime

from config import LM_URL
from config import agent
from config import memory
import history
import longterm
import reminders
import websearch


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


def _strip_leading_timestamps(text: str) -> str:
    pattern = r"^(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*)+"
    return re.sub(pattern, "", text).strip()


def _timestamped(role, content, timestamp):
    return {"role": role, "content": f"[{timestamp}] {content}"}


def ask(text, model):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    messages = [{"role": "system", "content": build_system_prompt()}]

    # get_messages_full() keeps each message's stored timestamp, so
    # the model can see how much time passed between past turns too,
    # not just how old the newest message is.
    for m in history.get_messages_full():
        messages.append(_timestamped(m["role"], m["content"], m.get("timestamp", "unknown time")))

    if websearch.looks_like_search(text):
        query = websearch.extract_query(text)
        results = websearch.search(query)
        messages.append({"role": "system", "content": websearch.format_results(query, results)})

    messages.append(_timestamped("user", text, now_str))

    payload = { 
        "model": model,
        "messages": messages,
    }

    payload.update(build_generation_params())

    response = requests.post(LM_URL,json=payload)

    answer = response.json()["choices"][0]["message"]["content"]
    answer = _strip_leading_timestamps(answer)

    # Stored content itself stays clean (no timestamp prefix baked in) -
    # the prefix above is only added when building the API payload, so
    # get_messages_full()'s own "timestamp" field stays the single
    # source of truth and formatting can change later without rewriting
    # anything already saved to disk.
    history.add_message("user", text)
    history.add_message("assistant", answer)
    history.maybe_summarize(model)
    history.save()

    longterm.extract_in_background(model, text, answer)
    reminders.extract_in_background(model, text)

    return answer
