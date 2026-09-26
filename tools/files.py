"""Reading and writing your files - with you in the loop for the writing.

"Luna, write me a bash script in ~/scripts that backs up my configs"
should just work, and it can't be allowed to just work: a 9B model
given a write tool will, sooner or later, write the wrong thing to the
wrong place with total confidence. So the split is:

  * Reads are free (list_files, read_file) - under your home, minus a
    deny list of places nothing should ever paste into a prompt.
  * Writes ask (write_file, edit_file). Every one pops the approval
    window in the TUI with the path, what's about to happen, and the
    full content or a diff. You answer on the keyboard. No answer in
    time is a no.

The deny list is not negotiable by the model or by the popup: keys,
credentials, shell startup files. There is no version of "she edits my
.bashrc" that ends well, and a popup you'd approve on reflex is not a
safeguard.

None of these tools are in chatroom.TOOL_CEILING, so a stream-chat turn
can't reach them at all - the containment there is by omission, not
instruction, and this module doesn't change that.
"""
import difflib
import os
import stat

import config
import logbook
import state

from . import tool

HOME = os.path.expanduser("~")

# Places a model must never read or write, however nicely it asks. Read
# matters as much as write here: a read puts the file's contents into
# the prompt, and from there into history, the transcript, and the log.
_DENY_DIRS = (
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
    ".config/ai-voice",          # pomf.json and friends
    ".local/share/keyrings", ".password-store", ".mozilla", ".thunderbird",
)
_DENY_FILES = (
    ".bashrc", ".zshrc", ".profile", ".bash_profile", ".bash_history",
    ".zsh_history", ".netrc", ".git-credentials", ".pgpass", ".npmrc",
    ".pypirc", ".env",
)
_DENY_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".kdbx", ".gpg", ".asc")

MAX_READ_BYTES = 200 * 1024
MAX_PROMPT_CHARS = 12000  # what read_file hands back; the rest is a note
MAX_WRITE_BYTES = 512 * 1024


class Refused(Exception):
    """A path or an operation the policy won't allow, with the reason."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def resolve(path):
    """Expand ~ and relative pieces; refuse anything the policy forbids.

    Returns the absolute path. Raises Refused with a reason a model can
    relay to the user without inventing one.
    """
    raw = str(path or "").strip()

    if not raw:
        raise Refused("no path given")

    full = os.path.realpath(os.path.expanduser(raw))

    if full != HOME and not full.startswith(HOME + os.sep):
        raise Refused(f"{raw} is outside your home folder ({HOME}) - "
                      "files can only be touched under it")

    rel = os.path.relpath(full, HOME)
    parts = rel.split(os.sep)

    for deny in _DENY_DIRS:
        deny_parts = deny.split("/")

        if parts[:len(deny_parts)] == deny_parts:
            raise Refused(f"~/{deny} is off limits (keys, credentials or "
                          "browser data)")

    if parts[-1] in _DENY_FILES:
        raise Refused(f"{parts[-1]} is off limits (shell startup or "
                      "credentials)")

    if parts[-1].lower().endswith(_DENY_SUFFIXES):
        raise Refused(f"{parts[-1]} looks like a key or secret - off limits")

    return full


def _display(full):
    """~/relative form for the popup and the tool results."""
    if full == HOME:
        return "~"

    if full.startswith(HOME + os.sep):
        return "~/" + os.path.relpath(full, HOME)

    return full


def _is_text(data):
    """Binary sniff: a NUL byte in the first 8KB means not text."""
    return b"\x00" not in data[:8192]


# ---------------------------------------------------------------------------
# Approval - the popup lives in ui.py; this is the request shape
# ---------------------------------------------------------------------------
def _note_denied():
    try:
        import mood

        mood.note_denied()
    except Exception:
        pass


def _ask(action, full, body_lines, note=""):
    """Block until the user answers the popup. False on timeout or when
    there's no TUI to ask (headless = deny; never write unasked)."""
    import ui

    if state.turn_source != "typed":
        # Voice: a one-liner so you know to look at the screen. The
        # details stay in the popup; reading a diff aloud helps nobody.
        try:
            from speech import speak

            speak(f"Can I {action} {os.path.basename(full)}? "
                  "It's on screen.")
        except Exception as e:
            logbook.warn("files", "couldn't voice the request: %s", e)

    return ui.ask_approval(
        title=f"{action} {_display(full)}",
        note=note,
        body=body_lines,
        timeout=config.FILES_APPROVAL_TIMEOUT,
    )


def _diff(old, new, name):
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"{name} (now)", tofile=f"{name} (after)", lineterm="",
    ))

    return lines or ["(no change)"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _enabled():
    return config.FILES_ENABLED


def _why():
    return "files.enabled is false"


@tool(
    "list_files",
    "List what's in a folder under the user's home - names, with a / "
    "after folders. Use it before writing somewhere you haven't seen, "
    "and to answer 'what's in my X folder'.",
    {
        "path": {
            "type": "string",
            "description": "Folder path. ~ means home. e.g. '~/scripts'.",
        },
    },
    required=("path",),
    available=_enabled,
    why=_why,
)
def _list_files(path):
    try:
        full = resolve(path)
    except Refused as e:
        return f"Refused: {e}"

    if not os.path.isdir(full):
        return f"{_display(full)} isn't a folder" + (
            "" if os.path.exists(full) else " - it doesn't exist")

    try:
        names = sorted(os.listdir(full), key=str.lower)
    except OSError as e:
        return f"Couldn't list {_display(full)}: {e.strerror}"

    shown = []

    for name in names[:200]:
        if name.startswith(".") and name not in (".", ".."):
            continue  # dotfiles are noise here and often off limits anyway

        shown.append(name + ("/" if os.path.isdir(os.path.join(full, name)) else ""))

    if not shown:
        return f"{_display(full)} is empty"

    more = f"\n... and {len(names) - 200} more" if len(names) > 200 else ""

    return f"{_display(full)}:\n" + "\n".join(shown) + more


@tool(
    "read_file",
    "Read a text file under the user's home. Do this before editing "
    "anything, and to answer questions about a file's contents.",
    {
        "path": {
            "type": "string",
            "description": "File path. ~ means home.",
        },
    },
    required=("path",),
    available=_enabled,
    why=_why,
)
def _read_file(path):
    try:
        full = resolve(path)
    except Refused as e:
        return f"Refused: {e}"

    if not os.path.isfile(full):
        return f"{_display(full)} doesn't exist" if not os.path.exists(full) \
            else f"{_display(full)} isn't a regular file"

    size = os.path.getsize(full)

    if size > MAX_READ_BYTES:
        return f"{_display(full)} is {size // 1024} KB - too big to read " \
               f"(limit {MAX_READ_BYTES // 1024} KB)"

    with open(full, "rb") as handle:
        data = handle.read()

    if not _is_text(data):
        return f"{_display(full)} is binary, not text"

    text = data.decode("utf-8", errors="replace")

    if len(text) > MAX_PROMPT_CHARS:
        return (text[:MAX_PROMPT_CHARS]
                + f"\n... [{len(text) - MAX_PROMPT_CHARS} more characters "
                  "not shown]")

    return text if text else "(empty file)"


@tool(
    "write_file",
    "Create a file, or replace one, under the user's home. The user "
    "sees the full content in an approval window and must say yes "
    "first - if the result says denied, tell them and do not try again. "
    "For a script, set executable true.",
    {
        "path": {
            "type": "string",
            "description": "Where to write. ~ means home. Folders that "
                           "don't exist yet are created.",
        },
        "content": {
            "type": "string",
            "description": "The complete file contents.",
        },
        "executable": {
            "type": "boolean",
            "description": "true for scripts, so they can be run directly.",
        },
    },
    required=("path", "content"),
    available=_enabled,
    why=_why,
)
def _write_file(path, content, executable=False):
    try:
        full = resolve(path)
    except Refused as e:
        return f"Refused: {e}"

    content = str(content if content is not None else "")

    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return "Refused: that's over the write size limit"

    if os.path.isdir(full):
        return f"Refused: {_display(full)} is a folder"

    exists = os.path.isfile(full)
    executable = bool(executable) or (
        not exists and content.startswith("#!")
    )
    parent = os.path.dirname(full)
    notes = []

    if not os.path.isdir(parent):
        notes.append(f"creates folder {_display(parent)}/")

    if executable:
        notes.append("marked executable")

    if exists:
        try:
            with open(full, "rb") as handle:
                old_data = handle.read()

            old = old_data.decode("utf-8", errors="replace") if _is_text(old_data) else None
        except OSError as e:
            return f"Couldn't read the existing file: {e.strerror}"

        action = "overwrite"
        body = _diff(old, content, os.path.basename(full)) if old is not None \
            else ["(existing file is binary - it will be replaced entirely)"]
        notes.insert(0, f"replaces {len(old_data)} bytes")
    else:
        action = "create"
        body = content.splitlines() or ["(empty file)"]

    lines = len(content.splitlines())
    notes.insert(0, f"{lines} line{'s' if lines != 1 else ''}")

    if not _ask(action, full, body, note=", ".join(notes)):
        _note_denied()

        return f"Denied by the user - {_display(full)} was not written. Do not retry."

    try:
        os.makedirs(parent, exist_ok=True)

        with open(full, "w", encoding="utf-8") as handle:
            handle.write(content)

        if executable:
            mode = os.stat(full).st_mode
            os.chmod(full, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as e:
        return f"Couldn't write {_display(full)}: {e.strerror}"

    logbook.info("files", "%s %s (%d lines%s)", action, full, lines,
                 ", +x" if executable else "")

    return f"{'Replaced' if exists else 'Wrote'} {_display(full)} " \
           f"({lines} lines{', executable' if executable else ''})"


@tool(
    "edit_file",
    "Change part of an existing file: replace one exact passage with "
    "another. `find` must appear exactly once - read the file first and "
    "copy it verbatim. The user sees the diff and must approve; if the "
    "result says denied, tell them and do not try again.",
    {
        "path": {
            "type": "string",
            "description": "The file to change. ~ means home.",
        },
        "find": {
            "type": "string",
            "description": "The exact existing text to replace - enough "
                           "of it to be unique in the file.",
        },
        "replace": {
            "type": "string",
            "description": "What it becomes. Empty string deletes it.",
        },
    },
    required=("path", "find", "replace"),
    available=_enabled,
    why=_why,
)
def _edit_file(path, find, replace):
    try:
        full = resolve(path)
    except Refused as e:
        return f"Refused: {e}"

    if not os.path.isfile(full):
        return f"{_display(full)} doesn't exist - use write_file to create it"

    find = str(find or "")
    replace = str(replace if replace is not None else "")

    if not find:
        return "Refused: `find` is empty - say what to replace"

    try:
        with open(full, "rb") as handle:
            data = handle.read()
    except OSError as e:
        return f"Couldn't read {_display(full)}: {e.strerror}"

    if not _is_text(data):
        return f"{_display(full)} is binary - can't edit it"

    old = data.decode("utf-8", errors="replace")
    hits = old.count(find)

    if hits == 0:
        return ("That text isn't in the file. Read it again and copy the "
                "passage exactly, including whitespace.")

    if hits > 1:
        return f"That text appears {hits} times - include more of the " \
               "surrounding lines so it's unique."

    new = old.replace(find, replace, 1)
    body = _diff(old, new, os.path.basename(full))
    changed = sum(1 for l in body if l[:1] in "+-" and not l.startswith(("+++", "---")))

    if not _ask("edit", full, body, note=f"{changed} line{'s' if changed != 1 else ''} change"):
        _note_denied()

        return f"Denied by the user - {_display(full)} was not changed. Do not retry."

    try:
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(new)
    except OSError as e:
        return f"Couldn't write {_display(full)}: {e.strerror}"

    logbook.info("files", "edit %s (%d changed lines)", full, changed)

    return f"Edited {_display(full)} ({changed} lines changed)"
