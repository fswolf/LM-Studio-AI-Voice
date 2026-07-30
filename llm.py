import requests

from config import LM_URL
from config import agent
from config import memory
import history
import longterm
import reminders


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

    return f"""
You are {agent['name']}.

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
    """
    Pull max_tokens/reasoning from agent.json's "generation" block.
    reasoning is only included if set (not null/empty) - sending it to
    a model that doesn't support reasoning causes LM Studio to error,
    so leaving it null in agent.json just omits it from the request.
    """
    gen = agent.get("generation", {})
    params = {}

    max_tokens = gen.get("max_tokens")
    if max_tokens:
        params["max_tokens"] = max_tokens

    reasoning = gen.get("reasoning")
    if reasoning:
        params["reasoning"] = reasoning

    return params


def ask(text,model):
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(history.get_messages())
    messages.append({"role": "user", "content": text})

    payload = {
        "model": model,
        "messages": messages,
    }
    payload.update(build_generation_params())

    response = requests.post(
        LM_URL,
        json=payload
    )

    answer = response.json()["choices"][0]["message"]["content"]

    history.add_message("user", text)
    history.add_message("assistant", answer)
    history.maybe_summarize(model)
    history.save()

    longterm.extract_in_background(model, text, answer)
    reminders.extract_in_background(model, text)

    return answer