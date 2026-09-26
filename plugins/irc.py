"""IRC. Libera.Chat and #gameranger by default.

Transport only, the same as pomf.py: who gets answered, how often, and
what she is allowed to do about it all live in chatroom.py. What is
specific to IRC is a socket, a handshake, and the two things that will
get you disconnected if you get them wrong - PING and flooding.

Stdlib only. IRC is a line protocol over a socket, and the libraries
that wrap it are bigger than the part of it this needs.

## Unlike pomf, she talks back

pomf is one-way: she reads the room and answers out loud. Here she also
PRIVMSGs the answer into the channel, which makes her a visible bot in
somebody else's room. That is what the send queue and the line splitter
below are for. A bot that floods is a bot that gets banned, and the
channel did not ask for a paragraph.

## Settings, under "plugins" in config.json

    "irc": {
        "enabled": false,
        "server": "irc.libera.chat",
        "port": 6697,
        "tls": true,
        "channel": "#gameranger",
        "nick": "",              defaults to her name, lowercased
        "post_replies": true,    false = out loud only, never posts
        "max_reply_lines": 3,
        "ignore": ["SomeBot"]
    }

## NickServ, if the channel needs you registered

Credentials are not settings - config.json is in the repo. They go in
~/.config/ai-voice/irc.json, beside the pomf ones:

    {"sasl_user": "luna", "sasl_password": "..."}

SASL PLAIN is used when both are present, which authenticates during
the handshake rather than after it. That ordering matters: a channel
with +r set rejects the JOIN of an unidentified nick, and a NickServ
message sent after JOIN is a message sent after it was refused.
"""
import base64
import json
import os
import random
import re
import socket
import ssl
import threading
import time

from datetime import datetime

import chatroom
import config
import logbook
import timeutil

NAME = "irc"
SUMMARY = "answers a channel on IRC, out loud and in the room"

CREDENTIALS_FILE = os.path.expanduser("~/.config/ai-voice/irc.json")

DEFAULTS = {
    "server": "irc.libera.chat",
    "port": 6697,
    "tls": True,
    "channel": "#gameranger",
    "nick": "",
    "post_replies": True,
    "max_reply_lines": 3,
}

# A PRIVMSG goes out as ":nick!user@host PRIVMSG #chan :text\r\n" and
# the whole thing has to fit in 512 bytes. We don't know the host mask
# the server gives us, so budget generously for it and measure the text
# in BYTES - a 400-character reply with an emoji in it is not 400 bytes,
# and the server truncates by bytes without asking.
MAX_LINE_BYTES = 400

# Libera's flood limit is roughly a line every two seconds with a small
# burst. Undercutting it costs nothing; exceeding it gets you killed
# with "Excess Flood" and a temporary ban.
SEND_INTERVAL = 2.0

# recv() timeout. Quiet channels are normal, so this only bounds how
# long we wait before checking whether the server has gone silent.
RECV_TIMEOUT = 30

# No data at all for this long means the connection is dead even though
# the socket still looks open - the usual way an IRC connection ends.
# The server pings well inside this; so do we, if it doesn't.
DEAD_AFTER = 300
PING_AFTER = 120

# mIRC colour, bold, italic, underline, reverse, reset.
_FORMATTING = re.compile(r"\x03(?:\d{1,2}(?:,\d{1,2})?)?|[\x02\x0f\x11\x16\x1d\x1e\x1f]")

_room = None
_thread = None
_stop = threading.Event()
_lock = threading.Lock()

_running = False
_connected = False
_last_error = ""
_nick_in_use = ""

_sock = None
_send_lock = threading.Lock()
_last_send = 0.0

_lines = 0
_last_line = ""
_last_line_at = 0.0
_posted = 0


def settings():
    merged = dict(DEFAULTS)
    merged.update(config.plugin_settings(NAME))

    return merged


def _credentials():
    """(sasl_user, sasl_password) from outside the repo, or ("", "")."""
    try:
        with open(CREDENTIALS_FILE) as handle:
            parsed = json.load(handle)
    except FileNotFoundError:
        return "", ""
    except (ValueError, OSError) as e:
        logbook.warn(NAME, "couldn't read %s: %s", CREDENTIALS_FILE, e)

        return "", ""

    if not isinstance(parsed, dict):
        return "", ""

    return (str(parsed.get("sasl_user", "")).strip(),
            str(parsed.get("sasl_password", "")).strip())


def nick():
    configured = str(settings().get("nick", "")).strip()

    # IRC nicks can't contain spaces and traditionally start with a
    # letter. Her name is normally fine; this is for the cases where
    # somebody has renamed her to something with a space in it.
    fallback = re.sub(r"[^A-Za-z0-9_\[\]\\^{}|`-]", "", config.AGENT_NAME) or "luna"

    return configured or fallback.lower()


def available():
    block = settings()

    return bool(str(block.get("server", "")).strip()
                and str(block.get("channel", "")).strip())


def why_unavailable():
    block = settings()

    if not str(block.get("server", "")).strip():
        return 'no server set - "irc": {"server": "irc.libera.chat"} under plugins'

    if not str(block.get("channel", "")).strip():
        return 'no channel set - "irc": {"channel": "#gameranger"} under plugins'

    return ""


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
def _raw(line):
    """One protocol line, rate limited, never more than 512 bytes."""
    global _last_send

    with _send_lock:
        connection = _sock

        if connection is None:
            return False

        wait = SEND_INTERVAL - (time.time() - _last_send)

        if wait > 0:
            time.sleep(wait)

        payload = line.encode("utf-8", "replace")[:510] + b"\r\n"

        try:
            connection.sendall(payload)
        except OSError as e:
            logbook.warn(NAME, "send failed: %s", e)

            return False

        _last_send = time.time()

        return True


def _handshake_raw(connection, line):
    """Send during the handshake, before _sock exists and unthrottled.

    The registration burst is expected by the server and exempt from
    flood limits; throttling it to one line every two seconds makes
    connecting take ten seconds for no reason.
    """
    connection.sendall(line.encode("utf-8", "replace")[:510] + b"\r\n")


def _split_for_irc(text, limit=MAX_LINE_BYTES):
    """One reply as a list of lines that will survive the wire.

    Splits on words, measured in bytes, because the server truncates by
    bytes and a chopped multi-byte character arrives as a mojibake
    question mark.
    """
    out = []

    for paragraph in str(text or "").replace("\r", "").split("\n"):
        paragraph = _FORMATTING.sub("", paragraph).strip()

        if not paragraph:
            continue

        current = ""

        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()

            if len(candidate.encode("utf-8")) <= limit:
                current = candidate
                continue

            if current:
                out.append(current)

            # A single word longer than the limit - a pasted URL,
            # usually. Cut it by bytes and keep going rather than
            # dropping it.
            while len(word.encode("utf-8")) > limit:
                chunk = word.encode("utf-8")[:limit].decode("utf-8", "ignore")

                if not chunk:
                    break

                out.append(chunk)
                word = word[len(chunk):]

            current = word

        if current:
            out.append(current)

    return out


def _say(channel, text, max_lines):
    """Post a reply into the channel, capped."""
    global _posted

    lines = _split_for_irc(text)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

        # Better a visibly clipped answer than a wall of text, and
        # better still that the room can tell it was clipped.
        lines[-1] = lines[-1].rstrip(" .,") + " ..."

    for line in lines:
        # Strip control characters: CR or LF in the payload is command
        # injection on a line protocol, and \x01 is CTCP.
        clean = "".join(c for c in line if ord(c) >= 32 or c == "\t")

        if not _raw(f"PRIVMSG {channel} :{clean}"):
            return

        _posted += 1


# ---------------------------------------------------------------------------
# Receiving
# ---------------------------------------------------------------------------
def _parse(line):
    """(prefix, command, params) for one raw IRC line."""
    prefix = ""

    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")

    # The trailing parameter is whatever follows the first " :", and it
    # is the only one allowed to contain spaces. `sep` rather than
    # `trailing` decides whether there was one: "PRIVMSG #chan :" has an
    # empty trailing parameter, which is not the same as having none.
    head, sep, trailing = line.partition(" :")
    parts = head.split()
    command = parts[0].upper() if parts else ""
    params = parts[1:]

    if sep:
        params.append(trailing)

    return prefix, command, params


def _nick_of(prefix):
    return prefix.split("!", 1)[0]


def _handle(prefix, command, params, channel, me):
    global _lines, _last_line, _last_line_at

    _lines += 1
    _last_line = command[:20]
    _last_line_at = time.time()

    if command != "PRIVMSG" or len(params) < 2:
        return

    target, message = params[0], params[1]

    # Channel traffic only. Answering DMs would make her a private
    # oracle for anyone who opens a query window, with none of the
    # social pressure of a room watching.
    if not target.startswith(("#", "&")):
        logbook.info(NAME, "ignoring a private message from %s", _nick_of(prefix))
        return

    if target.lower() != channel.lower():
        return

    # CTCP. ACTION is "/me waves" and reads as speech; anything else is
    # a client-to-client request we have no business answering.
    if message.startswith("\x01"):
        body = message.strip("\x01")

        if not body.upper().startswith("ACTION "):
            return

        message = body[7:]

    message = _FORMATTING.sub("", message)
    who = _nick_of(prefix)

    if who.lower() == me.lower():
        return

    _room.saw(who, message)


# ---------------------------------------------------------------------------
# The connection
# ---------------------------------------------------------------------------
def _authenticate(connection, reader, want_nick, user, password):
    """SASL PLAIN during registration. Returns a problem string, or "".

    Best effort: a server that doesn't offer SASL, or a password that
    is wrong, is logged and connected through anyway. Refusing to
    connect over it would leave her out of an open channel because of a
    credential the channel never wanted.
    """
    _handshake_raw(connection, "CAP LS 302")
    _handshake_raw(connection, f"NICK {want_nick}")
    _handshake_raw(connection, f"USER {want_nick} 0 * :{config.AGENT_NAME}")

    if not (user and password):
        _handshake_raw(connection, "CAP END")

        return ""

    _handshake_raw(connection, "CAP REQ :sasl")
    deadline = time.time() + 20
    problem = ""

    while time.time() < deadline:
        line = reader()

        if line is None:
            break

        prefix, command, params = _parse(line)

        if command == "PING":
            _handshake_raw(connection, f"PONG :{params[-1] if params else ''}")
        elif command == "CAP" and len(params) >= 2 and params[1].upper() == "ACK":
            _handshake_raw(connection, "AUTHENTICATE PLAIN")
        elif command == "CAP" and len(params) >= 2 and params[1].upper() == "NAK":
            problem = "the server doesn't offer SASL"
            break
        elif command == "AUTHENTICATE" and params and params[0] == "+":
            blob = base64.b64encode(
                f"{user}\0{user}\0{password}".encode()
            ).decode()
            _handshake_raw(connection, f"AUTHENTICATE {blob}")
        elif command == "903":
            logbook.info(NAME, "authenticated as %s", user)
            break
        elif command in ("902", "904", "905", "906", "907"):
            problem = f"SASL was refused ({command})"
            break

    _handshake_raw(connection, "CAP END")

    if problem:
        logbook.warn(NAME, "%s - connecting unauthenticated", problem)

    return problem


def _connect_once(block):
    global _sock, _connected, _last_error, _nick_in_use

    server = str(block["server"]).strip()
    port = int(block["port"])
    channel = str(block["channel"]).strip()
    want = nick()

    connection = socket.create_connection((server, port), timeout=30)

    if block.get("tls", True):
        context = ssl.create_default_context()
        connection = context.wrap_socket(connection, server_hostname=server)

    connection.settimeout(RECV_TIMEOUT)

    buffer = b""

    def read_line():
        """One line, or None on timeout/close. Shared with the handshake."""
        nonlocal buffer

        while b"\r\n" not in buffer:
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                return None
            except OSError:
                return None

            if not chunk:
                return None

            buffer += chunk

        line, _, buffer = buffer.partition(b"\r\n")

        return line.decode("utf-8", "replace")

    try:
        user, password = _credentials()
        _authenticate(connection, read_line, want, user, password)

        _sock = connection
        joined = False
        last_traffic = time.time()
        pinged = False

        while not _stop.is_set():
            line = read_line()

            if line is None:
                idle = time.time() - last_traffic

                if idle > DEAD_AFTER:
                    raise OSError(f"no traffic for {int(idle)}s")

                # Ask the server whether it's still there rather than
                # waiting out the full timeout. A connection that has
                # silently died otherwise looks exactly like a quiet
                # channel for five minutes.
                if idle > PING_AFTER and not pinged:
                    _raw(f"PING :{server}")
                    pinged = True

                continue

            last_traffic = time.time()
            pinged = False
            prefix, command, params = _parse(line)

            if command == "PING":
                # Unthrottled and before anything else. The rate limit
                # is for the things we choose to say; a late PONG is a
                # disconnect.
                try:
                    connection.sendall(
                        f"PONG :{params[-1] if params else ''}\r\n".encode()
                    )
                except OSError:
                    break

                continue

            if command == "433":  # nick in use
                want = f"{want}_"
                _nick_in_use = want
                logbook.warn(NAME, "nick taken, trying %s", want)
                _raw(f"NICK {want}")
                continue

            if command == "001" and not joined:
                _connected = True
                _last_error = ""
                logbook.info(NAME, "registered as %s, joining %s", want, channel)
                _raw(f"JOIN {channel}")
                joined = True
                continue

            if command == "473":  # invite only
                _last_error = f"{channel} is invite-only"
                logbook.warn(NAME, "%s", _last_error)
                continue

            if command in ("474", "475", "477"):
                _last_error = (f"can't join {channel} ({command}) - it may "
                               "need a registered nick")
                logbook.warn(NAME, "%s", _last_error)
                continue

            _handle(prefix, command, params, channel, want)
    finally:
        _connected = False
        _sock = None

        try:
            connection.close()
        except Exception:
            pass


def _listen_loop():
    global _last_error

    block = settings()
    delay = 5

    while not _stop.is_set():
        try:
            _connect_once(block)
            delay = 5
        except Exception as e:
            _last_error = f"{type(e).__name__}: {e}"[:140]
            logbook.warn(NAME, "connection failed: %s", _last_error)

        if _stop.is_set():
            break

        # Backoff with jitter. Reconnecting hard in a loop against
        # somebody else's IRC server is how a host gets K-lined, and a
        # dropped connection is usually the network rather than
        # anything trying faster would fix.
        time.sleep(delay + random.uniform(0, 2))
        delay = min(delay * 2, 300)


# ---------------------------------------------------------------------------
# The plugin interface
# ---------------------------------------------------------------------------
def start(model):
    global _room, _thread, _running

    with _lock:
        if _running:
            return False, "Already on IRC."

        problem = why_unavailable()

        if problem:
            return False, problem

        block = settings()
        channel = str(block["channel"]).strip()
        post = bool(block.get("post_replies", True))
        max_lines = int(block.get("max_reply_lines", 3))
        me = nick()

        def reply(_who, answer):
            if post:
                _say(channel, answer, max_lines)

        # Her own nick goes in the ignore list. pomf doesn't need this
        # because she never posts there; here she does, and one reply
        # containing her own name is an endless loop in a public room.
        merged = dict(block)
        merged["ignore"] = tuple(block.get("ignore", ())) + (me,)

        _room = chatroom.ChatRoom(
            NAME, owner=config.AGENT_NAME, settings=merged,
            where=f"the IRC channel {channel} on {block['server']}",
            reply=reply if post else None,
        )
        _room.start(model)

        _stop.clear()
        _thread = threading.Thread(target=_listen_loop, daemon=True)
        _thread.start()
        _running = True

    how = "out loud and in the channel" if post else "out loud only"

    return True, (
        f"Joining {channel} on {block['server']} as {me}. "
        f"She'll answer anything with \"{config.AGENT_NAME}\" in it, {how}."
    )


def stop():
    global _running

    with _lock:
        if not _running:
            return False, "Not on IRC."

        _stop.set()

        if _room is not None:
            _room.stop()

        # Leave properly. A QUIT is the difference between "left" and
        # a ghost sitting in the userlist until the server times it out.
        _raw("QUIT :bye")
        _running = False

    return True, "Left IRC."


def running():
    return _running


def status():
    if not _running:
        problem = why_unavailable()

        return f"irc: off ({problem})" if problem else "irc: off"

    block = settings()
    where = "connected" if _connected else f"reconnecting ({_last_error})"
    me = _nick_in_use or nick()

    lines = [f"irc: {where} to {block['server']} as {me}, in {block['channel']}"]

    if _lines:
        ago = timeutil.relative(
            datetime.fromtimestamp(_last_line_at), datetime.now()
        )
        lines.append(f"  {_lines} protocol lines, last was {_last_line} {ago}")
    else:
        lines.append("  nothing has arrived from the server yet")

    if block.get("post_replies", True):
        lines.append(f"  posts replies in-channel ({_posted} lines sent, "
                     f"max {block.get('max_reply_lines', 3)} per answer)")
    else:
        lines.append("  answers out loud only - never posts")

    lines.extend(_room.summary())

    # Same decision tree as pomf: three counters that between them say
    # where a silence is coming from, and nothing else does.
    if not _lines:
        lines.append("  -> nothing from the server. Check the host and port, "
                     "and whether TLS is right for that port.")
    elif not _room.seen:
        lines.append(f"  -> connected, but no channel messages. Is the JOIN "
                     f"to {block['channel']} succeeding? Check /log.")
    elif not _room.answered:
        lines.append(f"  -> messages are arriving, but none said "
                     f"\"{config.AGENT_NAME}\" (or they hit a cooldown).")

    return "\n".join(lines)
