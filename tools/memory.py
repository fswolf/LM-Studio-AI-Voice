"""What she knows about you, and what she can look up.

The editing tools take a loose description rather than an index,
for the same reason set_reminder takes words: the model is good
at repeating what was said and bad at bookkeeping.
"""
import config
import longterm
import transcript

from . import tool


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@tool(
    "remember_fact",
    "Save one durable fact about the user for future conversations - a "
    "preference, a project, something about their life. Only for things "
    "worth recalling weeks later, not passing details of this chat.",
    {
        "fact": {
            "type": "string",
            "description": "The fact, written as a short third-person "
                           "statement, e.g. 'Ryan streams as a VTuber'.",
        },
    },
    required=("fact",),
)
def _remember_fact(fact):
    if longterm.add_fact(fact):
        return f"Saved: {fact}"

    return "Not saved - too long, too short, or already known."

# `about` arrived with the sqlite backend. Without it this tool was a
# SELECT * - fine at thirty facts, a context bomb at three thousand.
# It stays optional so "what do you know about me" still works, but a
# bare call now returns the newest slice rather than everything.
@tool(
    "recall_facts",
    "Search what has been remembered about the user. Pass `about` to "
    "look something up ('his gpu', 'streaming'); omit it for the most "
    "recently learned facts.",
    {
        "about": {
            "type": "string",
            "description": "What to look for, in plain words. Optional.",
        },
    },
)
def _recall_facts(about=""):
    about = str(about or "").strip()

    if about:
        facts = longterm.relevant_facts(about, limit=10)

        if not facts:
            return f"Nothing remembered about {about!r}."
    else:
        facts = longterm.get_facts()[-15:]

        if not facts:
            return "Nothing saved yet."

    return "\n".join(f"- {f}" for f in facts)

@tool(
    "forget_fact",
    "Delete something remembered about the user, when they say it is "
    "wrong or ask you to forget it. Describe the fact in your own "
    "words - you do not need to quote it exactly.",
    {
        "about": {
            "type": "string",
            "description": "Roughly what the fact says, e.g. 'that he "
                           "works at the bakery'.",
        },
    },
    required=("about",),
)
def _forget_fact(about):
    removed, candidates = longterm.forget_fact(about)

    if removed:
        return f"Forgotten: {removed}"

    if candidates:
        listing = "\n".join(f"- {c}" for c in candidates)

        return (
            f"More than one could be it:\n{listing}\nAsk the user which "
            "one, then call this again with wording closer to theirs."
        )

    return f"Nothing remembered matches {about!r}. Nothing was deleted."

@tool(
    "update_fact",
    "Correct something remembered about the user when the old version "
    "is out of date - they moved, changed jobs, finished the project.",
    {
        "about": {
            "type": "string",
            "description": "Roughly what the old fact says.",
        },
        "corrected": {
            "type": "string",
            "description": "The replacement, as one short third-person "
                           "sentence.",
        },
    },
    required=("about", "corrected"),
)
def _update_fact(about, corrected):
    changed, candidates = longterm.update_fact(about, corrected)

    if changed:
        return f"Updated: {changed[0]!r} is now {changed[1]!r}"

    if candidates:
        listing = "\n".join(f"- {c}" for c in candidates)

        return f"More than one could be it:\n{listing}\nAsk which one."

    return (
        f"Nothing remembered matches {about!r}, and {corrected!r} was not "
        "saved. Use remember_fact if this is new."
    )

@tool(
    "search_history",
    "Look through past conversations for something the user is "
    "referring to. Use this whenever they mention something you can't "
    "see in the messages above - an earlier decision, a plan, a name. "
    "Search before saying you don't remember.",
    {
        "about": {
            "type": "string",
            "description": "What to look for, in their words - 'the "
                           "deploy window', 'what we decided about the "
                           "migration'.",
        },
    },
    required=("about",),
    available=lambda: config.TRANSCRIPT_ENABLED,
    why=lambda: "the transcript is off - set history.transcript true",
)
def _search_history(about):
    hits = transcript.search(about)

    if not hits:
        return (
            f"Nothing in past conversations matches {about!r}. Say so "
            "plainly rather than inventing a memory of it."
        )

    return (
        transcript.describe(hits, config.AGENT_NAME)
        + "\n\n(These were matched on wording alone, so some may be "
        "coincidence. If none of them actually answers the question, "
        "say you don't recall rather than stretching one to fit.)"
    )
