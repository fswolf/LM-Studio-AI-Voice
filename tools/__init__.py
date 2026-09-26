"""Tools the model can call for itself.

Before this, capabilities were bolted on with keyword matching: `"remind"
in text` fired a second model call to extract a reminder, `"web search"`
had to appear literally in your sentence, and a memory extractor ran
after every single turn whether or not there was anything to learn. The
model never chose any of it.

With tool calling the model decides, which means it can ask a follow-up
before scheduling, search only when a question actually needs it, and
chain steps together - look something up, then set a reminder about it.

Each tool is a JSON schema the model sees plus a Python function it
never sees. Everything returns a string; exceptions become error strings
rather than propagating, because a failed tool should be something the
model can talk about, not something that kills the turn.
"""
import json

import config

# Deliberately nothing else. Anything imported here becomes an
# attribute of the package, and a name that matches a submodule -
# `reminders`, `desktop` - wins over it in `from . import reminders`.
# That doesn't raise; the tools in that module just never register.
# Nine of them went missing exactly once, which was enough.

_REGISTRY = {}


def tool(name, description, properties, required=(), available=None, why=None):
    """Register a function as a callable tool.

    `available` is checked each turn - a tool whose dependencies aren't
    installed is simply not offered.
    """
    def decorator(fn):
        _REGISTRY[name] = {
            "run": fn,
            "available": available or (lambda: True),
            "why": why or (lambda: ""),
            "spec": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(required),
                    },
                },
            },
        }
        return fn

    return decorator


def specs(only=None):
    """The tool list sent to the model.

    A tool that can't work is left out rather than offered and failed.
    Telling a model it can see, when grim isn't installed, gets you an
    assistant that confidently describes a screen it never looked at.

    `only` narrows it to a named set. That exists for turns that came
    from somewhere other than the person sitting here - stream chat,
    most of all. Everything in this file acts on Ryan's machine or
    Ryan's data, so a turn typed by a stranger must not reach most of
    it, and "the model knows not to" is not a control. Leaving the
    tool out of the request is.
    """
    allowed = set(only) if only is not None else None

    return [
        entry["spec"] for name, entry in _REGISTRY.items()
        if entry["available"]() and enabled(name)
        and (allowed is None or name in allowed)
    ]


def names():
    return sorted(
        name for name, entry in _REGISTRY.items()
        if entry["available"]() and enabled(name)
    )


# ---------------------------------------------------------------------------
# Switching tools off
#
# Separate from `available` on purpose: that answers "can this tool
# work here" (is grim installed, is the transcript on), which is about
# the machine. This answers "do you want it offered", which is about
# the context window - every schema costs a few hundred tokens of it,
# every turn, whether or not it is ever called.
#
# Stored as an off-list rather than an on-list so a tool added later is
# offered by default. Opting out is a decision; opting in shouldn't be
# a chore.
# ---------------------------------------------------------------------------
# Which family each tool belongs to, for the tools pane. A dict rather
# than an argument on @tool because the decorator calls have half a
# dozen shapes between them and this is one place to read. A tool that
# isn't listed lands in "Other", which is visible rather than silent -
# so the drift announces itself the first time you open the pane.
_GROUPS = (
    ("Time", ("get_datetime", "time_until")),
    ("Reminders", ("set_reminder", "set_alarm", "list_reminders",
                   "cancel_reminder")),
    ("Memory", ("remember_fact", "recall_facts", "forget_fact",
                "update_fact", "search_history")),
    ("Files", ("list_files", "read_file", "write_file", "edit_file")),
    ("Desktop", ("look_at_screen", "control_audio", "clipboard",
                 "focus_window", "system_status")),
    ("Web", ("web_search", "read_page")),
)


def group_of(name):
    for label, members in _GROUPS:
        if name in members:
            return label

    return "Other"


def grouped():
    """[(group, [(name, on, available, why, cost), ...], live, total)].

    Ordered as _GROUPS declares, tools within a group by cost so the
    expensive ones are the ones you see first - that being the whole
    reason for looking. `live` is what this group currently costs,
    `total` what it would cost with everything in it switched on.
    """
    rows = {}

    for row in inventory():
        rows.setdefault(group_of(row[0]), []).append(row)

    order = [label for label, _members in _GROUPS] + ["Other"]
    out = []

    for label in order:
        members = rows.get(label)

        if not members:
            continue

        members.sort(key=lambda r: (-r[4], r[0]))
        live = sum(r[4] for r in members if r[1] and r[2])
        total = sum(r[4] for r in members if r[2])
        out.append((label, members, live, total))

    return out


def enabled(name):
    return name not in config.TOOLS_DISABLED


def set_enabled(name, on):
    """Turn one tool on or off, saved to config.json. Returns the new
    state, or None if there's no such tool."""
    if name not in _REGISTRY:
        return None

    off = set(config.TOOLS_DISABLED)
    off.discard(name) if on else off.add(name)
    config.TOOLS_DISABLED = sorted(off)

    try:
        # _store, not save_setting: a list isn't a /set-able scalar, so
        # it stays out of SETTINGS - same escape hatch plugin switches use.
        config._store("tools.disabled", config.TOOLS_DISABLED)
    except Exception:
        pass  # a failed save costs the preference, not the session

    return on


def cost(name):
    """Roughly what this tool's schema costs in tokens, every turn."""
    entry = _REGISTRY.get(name)

    return int(len(json.dumps(entry["spec"])) / 3.5) if entry else 0


def inventory():
    """(name, enabled, available, why, cost) for every registered tool,
    for the tools pane and /tools."""
    rows = []

    for name, entry in sorted(_REGISTRY.items()):
        ready = entry["available"]()
        rows.append((
            name,
            enabled(name),
            ready,
            "" if ready else entry["why"](),
            cost(name),
        ))

    return rows


def budget():
    """(tokens actually being sent, tokens if everything were on)."""
    live = sum(
        cost(name) for name, entry in _REGISTRY.items()
        if entry["available"]() and enabled(name)
    )
    possible = sum(
        cost(name) for name, entry in _REGISTRY.items() if entry["available"]()
    )

    return live, possible


def unavailable():
    """(name, why) for each tool that isn't being offered."""
    return [
        (name, entry["why"]())
        for name, entry in sorted(_REGISTRY.items())
        if not entry["available"]()
    ]


def call(name, arguments):
    """Run a tool by name. `arguments` is the JSON string the model sent."""
    entry = _REGISTRY.get(name)

    if entry is None:
        return f"Error: no tool called {name!r}. Available: {', '.join(names())}"

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: arguments for {name} were not valid JSON: {arguments!r}"

    if not isinstance(arguments, dict):
        return f"Error: arguments for {name} must be an object"

    try:
        return str(entry["run"](**arguments))
    except TypeError as e:
        return f"Error: wrong arguments for {name}: {e}"
    except Exception as e:
        return f"Error running {name}: {e}"


# ---------------------------------------------------------------------------
# The tools themselves
#
# Imported last, and for their side effects: each module calls @tool at
# import time to register what it defines. They can't be imported any
# earlier than this, because `tool` has to exist before they run.
#
# One module per group, matching the grouping the tools pane shows -
# so "what's in Desktop" has one answer rather than two that can drift.
# ---------------------------------------------------------------------------
from . import time      # noqa: E402,F401
from . import reminders  # noqa: E402,F401
from . import memory     # noqa: E402,F401
from . import files      # noqa: E402,F401
from . import desktop    # noqa: E402,F401
from . import web        # noqa: E402,F401
