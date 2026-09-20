"""The log nobody had when they needed it.

Everything this app knows goes to the TUI and then scrolls away. That is
fine while it works and useless the moment it doesn't: a reply comes
back empty, or she cuts herself off mid-sentence, and the only evidence
is a screenshot of a pane that has already moved on.

So: a rotating file under ~/.cache/ai-voice/, and `/log` to read the
tail of it without leaving the app.

Two rules hold this together.

Logging never breaks a turn. Every call here is wrapped - a full disk,
a read-only home, a weird locale, none of it is allowed to take down a
conversation. A missing log line costs you a debugging session; an
exception in the logging path costs you the reply.

And it records decisions, not just errors. "TTS failed" is a log line
anyone would write. "barge-in fired: level 0.041 over baseline 0.004 x
2.5, speech 0.96" is the one that actually tells you the calibration
window measured silence - which is the bug it took a screenshot and an
hour to find.
"""
import logging
import os
import sys
import traceback

from logging.handlers import RotatingFileHandler

from config import LOG_ENABLED, LOG_LEVEL, LOG_MAX_KB, LOG_KEEP

LOG_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "ai-voice",
)
LOG_FILE = os.path.join(LOG_DIR, "ai-voice.log")

_log = None
_broken = ""


def path():
    return LOG_FILE


def unavailable():
    return _broken


def start():
    """Open the log. Safe to call twice; never raises."""
    global _log, _broken

    if _log is not None or not LOG_ENABLED:
        return _log

    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=max(64, LOG_MAX_KB) * 1024,
            backupCount=max(0, LOG_KEEP),
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)-9s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        logger = logging.getLogger("ai-voice")
        logger.setLevel(getattr(logging, str(LOG_LEVEL).upper(), logging.INFO))
        logger.handlers.clear()
        logger.addHandler(handler)
        # Nothing may reach stderr - it would paint over the TUI.
        logger.propagate = False

        _log = logger
        _log.info("=" * 60)
        _log.info("started | python %s | pid %s", sys.version.split()[0], os.getpid())
    except Exception as e:
        _broken = str(e)
        _log = None

    return _log


def _write(level, where, message, *args):
    if _log is None:
        return

    try:
        _log.log(level, "[%s] " + message, where, *args)
    except Exception:
        pass  # a broken log line is not worth a broken turn


def debug(where, message, *args):
    _write(logging.DEBUG, where, message, *args)


def info(where, message, *args):
    _write(logging.INFO, where, message, *args)


def warn(where, message, *args):
    _write(logging.WARNING, where, message, *args)


def error(where, message, *args):
    _write(logging.ERROR, where, message, *args)


def exception(where, message, *args):
    """An error plus the traceback that caused it.

    The traceback is the whole point. Every `except Exception as e` in
    this app turns a crash into a one-line message for the user, which
    is right for them and hopeless for anyone trying to fix it.
    """
    _write(logging.ERROR, where, message, *args)

    if _log is None:
        return

    try:
        for line in traceback.format_exc().rstrip().splitlines():
            _log.error("[%s]     %s", where, line)
    except Exception:
        pass


def tail(lines=25):
    """The last few lines, for /log."""
    if _broken:
        return [f"logging is off: {_broken}"]

    if not os.path.exists(LOG_FILE):
        return ["nothing logged yet"]

    try:
        # Read the end rather than the whole file - it can be megabytes.
        with open(LOG_FILE, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            window = min(size, 64 * 1024)
            handle.seek(size - window)
            text = handle.read().decode("utf-8", errors="replace")

        found = text.splitlines()

        # A partial first line if we landed mid-way into one.
        if window < size and found:
            found = found[1:]

        return found[-lines:] or ["nothing logged yet"]
    except OSError as e:
        return [f"couldn't read the log: {e}"]
