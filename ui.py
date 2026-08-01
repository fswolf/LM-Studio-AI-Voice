import os
import shutil
import threading
from rich.console import Console

try:
    from wcwidth import wcswidth
except ImportError:  # pragma: no cover - fallback if wcwidth isn't installed
    wcswidth = None

console = Console()
_lock = threading.Lock()
WIDTH = 62  # fallback width if terminal size can't be read
# ---------------------------------------------------------------------------
# Shared UI state (mutated by assistant/keyboard threads, read on render)
# ---------------------------------------------------------------------------
state = {
    "agent_name": "Luna",
    "model": "unknown",
    "voice": "af_bella",
    "status": "Idle",
    "memory": "Loaded",
}
conversation = []  # list of (speaker, text) tuples
def set_status(status: str):
    with _lock:
        state["status"] = status
    render()
def add_message(speaker: str, text: str):
    with _lock:
        conversation.append((speaker, text))
    render()
def init(agent_name, model, voice, memory_status="Loaded"):
    with _lock:
        state["agent_name"] = agent_name
        state["model"] = model
        state["voice"] = voice
        state["memory"] = memory_status
    render()
# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _size():
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except Exception:
        return WIDTH, 24
def _width():
    return _size()[0]
def _clear():
    # \x1b[H   -> move cursor home
    # \x1b[2J  -> clear the visible screen
    # \x1b[3J  -> clear scrollback too, so old frames don't pile up
    #             when the user scrolls up or copies the buffer
    # Writing directly avoids spawning a subprocess (os.system) on
    # every single render, which was also a source of flicker/races
    # when multiple threads triggered renders back to back.
    print("\x1b[H\x1b[2J\x1b[3J", end="")
# Fixed chrome that's always shown, used to figure out how many rows
# are left over for the conversation area.
_HEADER_ROWS = 7      # title(2) + === + status block(4)
_CONV_LABEL_ROWS = 3  # --- + " Conversation" + --- (this was missing before, causing overflow)
_COMMANDS_ROWS = 3    # --- + " Commands" + --- + 3 command lines + ---
_PROMPT_ROWS = 1       # the "> " input line itself
def vwidth(s: str) -> int:
    """Visible width of a string in terminal cells.

    Emoji and other wide/zero-width characters break plain len()-based
    wrapping (an emoji is usually 2 cells wide but len() counts it as 1,
    sometimes 2 for multi-codepoint ones), which is what was pushing
    the conversation display down/misaligned. wcwidth measures real
    display width; if it isn't installed, fall back to len() rather
    than crashing.
    """
    if wcswidth is None:
        return len(s)
    w = wcswidth(s)
    return w if w is not None and w >= 0 else len(s)
def _wrap_text(text: str, width: int):
    """Word-wrap using visible width instead of character count.

    A drop-in replacement for textwrap.wrap() that won't misjudge
    lines containing emoji/wide characters.
    """
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = []
    current_width = 0
    for word in words:
        w = vwidth(word)
        space = 1 if current else 0
        if current_width + space + w > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_width = w
        else:
            current.append(word)
            current_width += space + w
    if current:
        lines.append(" ".join(current))
    return lines or [""]
def _wrapped_conversation_lines(width):
    """Wrap every message to `width`, tagged with its speaker label."""
    wrapped = []
    for speaker, text in conversation:
        label = "You " if speaker == "user" else state["agent_name"][:5]
        prefix = f"{label:<5}> "
        prefix_w = vwidth(prefix)
        indent = " " * prefix_w
        wrapped_msg = _wrap_text(text, width=max(10, width - prefix_w))
        wrapped.append(prefix + wrapped_msg[0])
        for cont in wrapped_msg[1:]:
            wrapped.append(indent + cont)
    return wrapped
def render():
    with _lock:
        w, h = _size()
        lines = []
        title = f" {state['agent_name']} AI Assistant"
        lines.append("=" * w)
        lines.append(title.center(w))
        lines.append("=" * w)
        lines.append(f"Status : {state['status']}")
        lines.append(f"Voice  : {state['voice']}")
        lines.append(f"Memory : {state['memory']}")
        lines.append(f"Model  : {state['model']}")
        lines.append("-" * w)
        lines.append(" Conversation")
        lines.append("-" * w)
        # Only show as many conversation lines as actually fit, so the
        # footer/prompt never get pushed off screen and the UI stays
        # pinned as a single "dashboard" instead of scrolling.
        available = max(1, h - _HEADER_ROWS - _CONV_LABEL_ROWS - _COMMANDS_ROWS - _PROMPT_ROWS)
        conv_lines = _wrapped_conversation_lines(w)[-available:]
        lines.extend(conv_lines)
        # pad up to `available` so the footer stays anchored to the
        # bottom instead of jumping around as the conversation grows
        for _ in range(available - len(conv_lines)):
            lines.append("")
        lines.append("-" * w)
        lines.append(" HOME = Push To Talk / Halt Voice Output, ESC = Quit")
        lines.append("-" * w)
        _clear()
        console.print("\n".join(lines))
def prompt() -> str:
    try:
        return console.input("> ")
    except (EOFError, KeyboardInterrupt):
        return "/quit"