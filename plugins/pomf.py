"""pomf.tv stream chat.

Transport only. Everything about who gets answered, how often, and what
she is allowed to do about it lives in chatroom.py, so this file is the
part that is actually specific to pomf: a websocket, a connect frame,
and a reconnect loop.

The protocol, taken from pomf's own chat client rather than from the
third-party bot this was first written against - that bot sends
"userName" and has not worked for some time:

    wss://pomf.tv/websocket/          Origin: https://pomf.tv
    ->  {"roomId":"<streamer>","userId":"<numeric id>",
         "apikey":"<key>","action":"connect"}
    ->  {"action":"ping"}                    every 30s, or it times out
    <-  {"type":"message","from":{"name":"viewer"},"message":"...",
         "timestamp":0,"msgid":1}
    <-  {"type":"message-emote", ...}        same shape, emote-only

`userId` is the number pomf knows your account by, not your display
name. Find it in your own chat popout: view source and look for
connectToChat("YourName", "12345") - the second argument is it.

That id and the API key are one credential - the key belongs to that
account - so they live together, outside the repo, in
~/.config/ai-voice/pomf.json:

    {"user_id": "94652", "apikey": "..."}

config.json is in the repo, and a key pasted into it is a key on
GitHub. Which room to watch is not a credential and stays there.
"""
import json
import os
import random
import threading
import time

from datetime import datetime

import chatroom
import config
import logbook
import timeutil

NAME = "pomf"
SUMMARY = "answers pomf.tv stream chat out loud"

WEBSOCKET_URL = "wss://pomf.tv/websocket/"
ORIGIN = "https://pomf.tv"

# Keepalive. recv() has a 30s timeout, so this fires between reads
# rather than on a timer of its own.
PING_SECONDS = 25
RECV_TIMEOUT = 30
CREDENTIALS_FILE = os.path.expanduser("~/.config/ai-voice/pomf.json")

_room = None
_thread = None
_stop = threading.Event()
_lock = threading.Lock()

_running = False
_connected = False
_last_error = ""

# Raw frames off the socket, before any filtering. Counted separately
# from the room's own tallies because the three numbers together say
# where a silence is coming from, and nothing else does:
#
#   frames 0            nothing is arriving - the socket or the key
#   frames >0, seen 0   arriving but discarded - wrong room id
#   seen >0, answered 0 heard, but not addressed to her, or rate limited
_frames = 0
_last_frame = ""
_last_frame_at = 0.0

# Anything that isn't a real key. A placeholder passes every "is it
# empty" check, sails through start(), and then fails at the far end
# where the only symptom is silence.
_PLACEHOLDERS = (
    "put_your_pomf_api_key_here", "your key from pomf", "changeme",
    "xxx", "...", "none", "null", "guest",
)


def settings():
    return config.plugin_settings(NAME)


def _stored():
    """The credentials file, as (fields, problem).

    A broken file is told apart from a missing one. Swallowing a JSON
    syntax error here makes a misplaced comma look exactly like "no API
    key", which sends you hunting for a key that was there all along.
    """
    try:
        with open(CREDENTIALS_FILE) as handle:
            parsed = json.load(handle)
    except FileNotFoundError:
        return {}, ""
    except ValueError as e:
        return {}, f"{CREDENTIALS_FILE} isn't valid JSON ({e})"
    except OSError as e:
        return {}, f"can't read {CREDENTIALS_FILE} ({e})"

    if not isinstance(parsed, dict):
        return {}, f"{CREDENTIALS_FILE} should contain an object"

    return parsed, ""


def credentials():
    """(channel, user_id, apikey).

    The id and the key are one credential, not two settings: the key
    belongs to that pomf account, and identifying as anybody else is
    how a connect gets refused with nothing at all to see. So they live
    together in one file, and splitting them - which is what this used
    to do - buys you a mismatch indistinguishable from a dead socket.

    `channel` is not part of it. That is which room to watch, not who
    you are, so it stays in config.json with the rest of the behaviour.

    There is no sensible fallback for the id. A missing one used to
    default to the channel name, which the server silently ignores;
    better to refuse to start and say where to find the number.
    """
    block = settings()
    channel = str(block.get("channel", "")).strip()

    stored, _problem = _stored()

    user_id = (str(stored.get("user_id", "")).strip()
               or str(stored.get("userId", "")).strip()
               or os.environ.get("POMF_USER_ID", "").strip())

    key = (os.environ.get("POMF_APIKEY", "").strip()
           or str(stored.get("apikey", "")).strip())

    return channel, user_id, key


def _library():
    try:
        import websocket  # websocket-client

        return websocket
    except ImportError:
        return None


def _is_placeholder(key):
    lowered = str(key or "").strip().lower()

    return any(mark in lowered for mark in _PLACEHOLDERS)


def available():
    channel, user_id, key = credentials()

    return bool(channel and key and user_id and user_id.isdigit()
                and not _is_placeholder(key) and _library())


def why_unavailable():
    if _library() is None:
        return "websocket-client isn't installed - pip install websocket-client"

    # Before anything about what's missing: a file that didn't parse
    # has nothing missing from it, it just didn't load.
    _fields, problem = _stored()

    if problem:
        return problem

    channel, _user, key = credentials()

    if not channel:
        return 'no channel set - "pomf": {"channel": "YourName"} under plugins'

    if not key:
        return (f"no API key - set POMF_APIKEY, or put "
                f'{{"user_id": "...", "apikey": "..."}} in {CREDENTIALS_FILE}')

    if _is_placeholder(key):
        return (f"the API key in {CREDENTIALS_FILE} is still the "
                "placeholder - put your real one in")

    _c, user_id, _k = credentials()

    if not user_id:
        return (f'no user_id - add {{"user_id": "12345"}} to '
                f"{CREDENTIALS_FILE}. It is the number in your own chat "
                'popout\'s source: connectToChat("YourName", "12345")')

    if not user_id.isdigit():
        return (f"user_id should be the number pomf knows your account "
                f"by, not a name ({user_id!r})")

    return ""


# ---------------------------------------------------------------------------
# The socket
# ---------------------------------------------------------------------------
def _handle(payload, channel):
    global _frames, _last_frame, _last_frame_at

    if not isinstance(payload, dict):
        return

    _frames += 1
    _last_frame = str(payload.get("type", "?"))[:30]
    _last_frame_at = time.time()

    # Every frame, at info. A chat plugin that goes quiet is a plugin
    # you cannot debug from the outside, and "what is actually coming
    # down the socket" is the only question worth asking first.
    logbook.info(
        NAME, "frame type=%s from=%s room=%s: %s",
        _last_frame, (payload.get("from") or {}).get("name"),
        payload.get("roomid"), str(payload.get("message", ""))[:120],
    )

    if payload.get("type") not in ("message", "message-emote"):
        return

    # A room id that isn't ours means the server is echoing something
    # we didn't ask for. Absent is fine; wrong is not - but say so,
    # because a channel name with the wrong case would drop every
    # message in silence otherwise.
    room = payload.get("roomid")

    if room and str(room).lower() != str(channel).lower():
        logbook.info(NAME, "ignoring a message for %s (we want %s)", room, channel)
        return

    _room.saw((payload.get("from") or {}).get("name"), payload.get("message"))


def _connect_once(websocket, channel, user_id, key):
    global _connected, _last_error

    connection = websocket.create_connection(
        WEBSOCKET_URL, origin=ORIGIN, timeout=RECV_TIMEOUT,
    )

    try:
        connection.send(json.dumps({
            "roomId": channel,
            # userId, not userName. The old third-party bot sends a
            # name here; the server ignores the whole frame when it
            # does, and the only symptom is a socket that stays open
            # and never delivers anything.
            "userId": user_id,
            "apikey": key,
            "action": "connect",
        }) + "\n")

        _connected = True
        _last_error = ""
        # The key is the one thing that never goes in the log, but
        # everything else about the handshake does - announcing
        # yourself as the wrong account is refused silently, and the
        # frame we sent is the first thing worth checking.
        logbook.info(
            NAME, "connected: roomId=%s userId=%s key=%s...%s (%d chars)",
            channel, user_id, key[:2], key[-2:], len(key),
        )

        last_ping = time.time()

        while not _stop.is_set():
            # pomf's own client has a notimeout() that sends this, so
            # the server evidently drops connections that go quiet.
            # Nothing else here would notice: a chat with no traffic
            # and a connection that was dropped look identical from
            # this side.
            if time.time() - last_ping >= PING_SECONDS:
                connection.send(json.dumps({"action": "ping"}))
                last_ping = time.time()

            try:
                frame = connection.recv()
            except websocket.WebSocketTimeoutException:
                # Quiet chat is normal, not a dead socket.
                continue

            if not frame:
                break  # a clean close from the far end

            if isinstance(frame, bytes):
                frame = frame.decode("utf-8", "replace")

            for line in frame.splitlines():
                line = line.strip()

                if not line:
                    continue

                try:
                    _handle(json.loads(line), channel)
                except ValueError:
                    continue
    finally:
        _connected = False

        try:
            connection.close()
        except Exception:
            pass


def _listen_loop():
    global _last_error

    websocket = _library()
    channel, user_id, key = credentials()
    delay = 2

    while not _stop.is_set():
        try:
            _connect_once(websocket, channel, user_id, key)
            delay = 2
        except Exception as e:
            _last_error = str(e)[:120]
            logbook.warn(NAME, "connection failed: %s", _last_error)

        if _stop.is_set():
            break

        # Backoff with jitter, capped. Reconnecting hard in a loop
        # against someone else's server is how an API key gets blocked,
        # and a dropped websocket is usually the network rather than
        # anything we can fix by trying faster.
        time.sleep(delay + random.uniform(0, 1))
        delay = min(delay * 2, 60)


# ---------------------------------------------------------------------------
# The plugin interface
# ---------------------------------------------------------------------------
def start(model):
    global _room, _thread, _running

    with _lock:
        if _running:
            return False, "Already listening to pomf chat."

        problem = why_unavailable()

        if problem:
            return False, problem

        channel, user_id, _key = credentials()

        _room = chatroom.ChatRoom(
            NAME, owner=channel, settings=settings(),
            where=f"{channel}'s pomf.tv stream chat",
        )

        # Nothing is added to the ignore list for her own account.
        # Incoming messages carry a display name and we connect with a
        # numeric id, so there is nothing to match on - and she doesn't
        # post, so the loop it would guard cannot happen. Bots go in
        # the configured "ignore" list by name.
        _room.start(model)

        _stop.clear()
        _thread = threading.Thread(target=_listen_loop, daemon=True)
        _thread.start()
        _running = True

    return True, (
        f"Listening to {channel}'s pomf chat (account {user_id}). "
        f"She'll answer "
        f"anything with \"{config.AGENT_NAME}\" in it, out loud."
    )


def stop():
    global _running

    with _lock:
        if not _running:
            return False, "Not listening to pomf chat."

        _stop.set()

        if _room is not None:
            _room.stop()

        _running = False

    return True, "Stopped listening to pomf chat."


def running():
    return _running


def status():
    if not _running:
        problem = why_unavailable()

        return f"pomf: off ({problem})" if problem else "pomf: off"

    channel, user_id, _key = credentials()
    where = "connected" if _connected else f"reconnecting ({_last_error})"

    if _frames:
        ago = timeutil.relative(
            datetime.fromtimestamp(_last_frame_at), datetime.now()
        )
        traffic = f"  {_frames} frames received, last was {_last_frame} {ago}"
    else:
        traffic = "  nothing has arrived down the socket yet"

    lines = [f"pomf: {where} to {channel} as account {user_id}", traffic]
    lines.extend(_room.summary())

    # The three counters read as a decision tree, so say which branch
    # we're on rather than leaving it to be worked out. A chat plugin
    # that goes quiet is otherwise undebuggable from the outside.
    if not _frames:
        lines.append("  -> nothing is arriving. Either the room is "
                     "silent, or the connect was refused - check that "
                     "user_id and apikey in pomf.json are the same account.")
    elif not _room.seen:
        lines.append("  -> frames are arriving but none are chat messages "
                     "for this room - check /log for the room id they "
                     "carry.")
    elif not _room.answered:
        lines.append(f"  -> messages are arriving, but none said "
                     f"\"{config.AGENT_NAME}\" (or they hit a cooldown).")

    return "\n".join(lines)
