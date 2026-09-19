"""Self-registering Hyprland hotkey.

Wayland gives applications no global key grabs - that's a protocol
decision, not a Python one, so no rewrite in any language dodges it. The
compositor has to own the binding.

Two ways to hand Hyprland a bind at runtime, depending on which config
parser is in use:

  legacy (hyprland.conf):
      hyprctl keyword bind SUPER,HOME,exec,<command>

  Lua config (hyprland.lua):
      keyword is refused - "keyword can't work with non-legacy parsers.
      Use eval." - so we evaluate the same bind as Lua instead:
      hyprctl eval '__ai_voice_bind = hl.bind("SUPER + HOME", ...)'

Either way the bind is runtime-only: it isn't written to your config and
it's removed on exit. Runtime binds don't survive a config reload, so if
you reload often, bind the key in your own config and set
"hotkey": "none" in agent.json.
"""
import os
import shutil
import subprocess
import sys

from config import BASE_DIR, HOTKEY, AUTO_BIND

CTL = os.path.join(BASE_DIR, "ai-voice-ctl.py")

# Lua global we stash the bind handle in, so we can disable it on exit.
_LUA_HANDLE = "__ai_voice_bind"

_ERROR_WORDS = (
    "error", "invalid", "can't", "cannot", "unknown", "failed",
    "no such", "nil value", "expected",
)

# Set once a bind is live, so the UI and the keyboard fallback know a
# working hotkey already exists.
bound = ""
method = ""  # "keyword" or "eval"

# Why the last bind() failed, surfaced by /keys.
last_error = ""


def available():
    """True when we're running under a Hyprland session with hyprctl."""
    return bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")) and \
        shutil.which("hyprctl") is not None


def _hyprctl(*args):
    """Run hyprctl. Returns (ok, output) - ok means it didn't obviously
    complain, since `keyword` answers "ok" but `eval` answers with the
    expression's result."""
    try:
        result = subprocess.run(
            ["hyprctl", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)

    output = (result.stdout + result.stderr).strip()
    low = output.lower()

    ok = result.returncode == 0 and not any(w in low for w in _ERROR_WORDS)

    return ok, output


def _normalize(hotkey):
    """'SUPER, HOME' / 'super+home' -> ('SUPER', 'HOME')."""
    parts = [p.strip() for p in hotkey.replace("+", ",").split(",") if p.strip()]

    if not parts:
        return "SUPER", "HOME"

    key = parts[-1]
    mods = " ".join(p.upper() for p in parts[:-1])

    return mods, key


def _lua_string(text):
    """Escape a Python string for embedding in a Lua double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _command(action):
    # ai-voice-ctl.py is stdlib-only, but sys.executable is guaranteed to
    # exist regardless of what's on PATH when the compositor runs it.
    return f"{sys.executable} {CTL} {action}"


def bind(action="ptt"):
    """Register the push-to-talk hotkey. Returns the human-readable combo
    on success, or "" (with last_error set) if it couldn't be bound."""
    global bound, method, last_error

    if not AUTO_BIND:
        last_error = "auto-bind disabled (hotkey set to none)"
        return ""

    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        last_error = "not a Hyprland session (HYPRLAND_INSTANCE_SIGNATURE unset)"
        return ""

    if shutil.which("hyprctl") is None:
        last_error = "hyprctl not found on PATH"
        return ""

    mods, key = _normalize(HOTKEY)
    command = _command(action)
    combo = f"{mods}+{key}" if mods else key

    # Path 1: legacy parser.
    ok, output = _hyprctl("keyword", "bind", f"{mods},{key},exec,{command}")

    if ok:
        bound, method, last_error = f"{mods},{key}", "keyword", ""
        return combo

    keyword_error = output

    # Path 2: Lua parser. Hyprland refuses `keyword` outright and points
    # at eval, so run the equivalent hl.bind() call.
    lua_combo = " + ".join(mods.split() + [key]) if mods else key
    lua = '{handle} = hl.bind("{combo}", hl.dsp.exec_cmd("{cmd}"))'.format(
        handle=_LUA_HANDLE,
        combo=_lua_string(lua_combo),
        cmd=_lua_string(command),
    )

    ok, output = _hyprctl("eval", lua)

    if ok:
        bound, method, last_error = lua_combo, "eval", ""
        return combo

    last_error = f"keyword: {keyword_error} | eval: {output}"

    return ""


def unbind():
    """Drop the bind so the key goes back to whatever it was."""
    global bound, method

    if not bound:
        return

    if method == "keyword":
        _hyprctl("keyword", "unbind", bound)
    else:
        # pcall so an older Lua API without :set_enabled() doesn't turn
        # shutdown into an error.
        _hyprctl(
            "eval",
            "if {h} then pcall(function() {h}:set_enabled(false) end) end".format(
                h=_LUA_HANDLE
            ),
        )

    bound, method = "", ""
