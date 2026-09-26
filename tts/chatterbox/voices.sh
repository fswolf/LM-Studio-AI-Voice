#!/usr/bin/env bash
# Audition voices without restarting anything.
#
# The slow part of picking a voice isn't making the reference clips -
# it's hearing what she actually sounds like with one. Editing
# config.json and restarting the assistant for every candidate turns a
# two-minute job into an evening, so `test` asks the running Chatterbox
# to say a line with a given voice and plays it back. Same model, same
# settings, same everything - just without the round trip.
#
#   ./voices.sh grab af_sky af_nicole af_bella
#   ./voices.sh test af_sky "Mrrp~ Senpai! Your build finished."
#   ./voices.sh use af_sky
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Empty means "go and look". An explicit override is honoured exactly,
# never quietly swapped for a different port that happens to answer.
CHATTERBOX="${CHATTERBOX_URL:-}"
SOURCE="${SOURCE_URL:-}"
CONFIG="../config.json"
LINE="Mrrp~ Senpai! I'm all set up and ready when you are."

die() { printf '%s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Finding things
# ---------------------------------------------------------------------------
engine_at() {
    curl -fsS --max-time 3 "$1/health" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("engine",""))' \
          2>/dev/null || true
}

# Both servers default to 8899, so neither port says which is which.
# Ask each one what it is rather than guessing from the number.
#
# `want` is an engine name, or "source" for anything that isn't
# Chatterbox. A dead port reports an empty engine, so matching has to
# require a non-empty answer - otherwise "find me anything" happily
# settles on a port with nothing behind it.
matches() {
    local got="$1" want="$2"
    [ -n "$got" ] || return 1
    if [ "$want" = source ]; then [ "$got" != chatterbox ]; else [ "$got" = "$want" ]; fi
}

find_engine() {
    local want="$1" explicit="$2" url

    # An explicit URL is a claim about where the thing is. If it's wrong,
    # say so rather than wandering off and using a different server than
    # the one that was named.
    if [ -n "$explicit" ]; then
        matches "$(engine_at "$explicit")" "$want" && { printf '%s' "$explicit"; return 0; }
        return 1
    fi

    for url in http://127.0.0.1:8899 http://127.0.0.1:8900 \
               http://127.0.0.1:8898 http://127.0.0.1:8901; do
        matches "$(engine_at "$url")" "$want" && { printf '%s' "$url"; return 0; }
    done
    return 1
}

player() {
    local candidate
    for candidate in ffplay paplay aplay mpv play; do
        command -v "$candidate" >/dev/null 2>&1 && { printf '%s' "$candidate"; return 0; }
    done
    return 1
}

play_file() {
    local tool; tool="$(player)" || {
        echo "  (no audio player found - install ffmpeg or alsa-utils)"
        echo "  saved: $1"; return 0; }
    case "$tool" in
        ffplay) ffplay -nodisp -autoexit -loglevel quiet "$1" ;;
        mpv)    mpv --really-quiet "$1" ;;
        play)   play -q "$1" ;;
        *)      "$tool" "$1" >/dev/null 2>&1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_list() { python3 make_voice.py --list; }

cmd_prepare() { python3 make_voice.py --prepare; }

cmd_grab() {
    [ $# -gt 0 ] || die "usage: $0 grab <voice> [voice...]"
    # Anything that isn't Chatterbox will do as a source; kokoro-reader
    # is just the likely one.
    local url
    url="$(find_engine source "$SOURCE")" || die \
"Couldn't find a source TTS server.

Start kokoro-reader, then try again. If Chatterbox has port 8899,
run it somewhere else:   CHATTERBOX_PORT=8900 ./run.sh
Or point this at the right one:   SOURCE_URL=http://127.0.0.1:8899 $0 grab ..."
    echo "source: $url"
    python3 make_voice.py --url "$url" "$@"
}

cmd_from() {
    [ $# -ge 1 ] || die "usage: $0 from <file> [name]"
    if [ $# -ge 2 ]; then
        python3 make_voice.py --from-file "$1" --name "$2"
    else
        python3 make_voice.py --from-file "$1"
    fi
}

cmd_test() {
    [ $# -ge 1 ] || die "usage: $0 test <voice> [text]"
    local voice="$1"; shift
    local text="${*:-$LINE}"
    local url; url="$(find_engine chatterbox "$CHATTERBOX")" || die \
"Chatterbox isn't answering. Start it with ./run.sh
(or set CHATTERBOX_URL if it's on an unusual port)."

    local out; out="$(mktemp --suffix=.wav)"
    trap 'rm -f "$out"' RETURN

    echo "$voice: \"$text\""
    local started; started=$(date +%s.%N)
    python3 - "$url" "$voice" "$text" "$out" <<'PY'
import json, sys, urllib.request, urllib.error
url, voice, text, out = sys.argv[1:5]
request = urllib.request.Request(
    f"{url}/tts",
    data=json.dumps({"text": text, "voice": voice, "speed": 1.0}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=300) as response:
        open(out, "wb").write(response.read())
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", "replace")[:200]
    sys.exit(f"  server said {e.code}: {body}")
PY
    local spent; spent=$(echo "$(date +%s.%N) - $started" | bc)
    printf '  generated in %.1fs\n' "$spent"
    play_file "$out"
}

cmd_use() {
    [ $# -eq 1 ] || die "usage: $0 use <voice>"
    [ -f "$CONFIG" ] || die "can't find $CONFIG"
    # Edited in place with json.load/dump so comments-free formatting and
    # every other setting survive - sed on JSON is how config files die.
    python3 - "$CONFIG" "$1" <<'PY'
import json, sys
path, voice = sys.argv[1], sys.argv[2]
with open(path) as handle:
    config = json.load(handle)
was = config.get("voice")
config["voice"] = voice
with open(path, "w") as handle:
    json.dump(config, handle, indent=4)
    handle.write("\n")
print(f"  voice: {was!r} -> {voice!r}")
PY
    echo "  restart the assistant to pick it up."
}

cmd_status() {
    local c s
    c="$(find_engine chatterbox "$CHATTERBOX")" && \
        echo "chatterbox   $c" || echo "chatterbox   not running"
    s="$(find_engine source "$SOURCE")" && \
        echo "source       $s" || echo "source       not running"
    [ -f "$CONFIG" ] && python3 -c \
        "import json;print('config       voice =', json.load(open('$CONFIG')).get('voice'))" \
        || true
}

usage() {
    cat <<EOF
$0 <command>

  prepare                    convert whatever you dropped in voices/
  list                       the reference clips you have
  grab <voice>...            pull voices off a running TTS server
  from <file> [name]         clean up a clip you already have
  test <voice> [text]        hear it, through the running Chatterbox
  use <voice>                set it in the assistant's config.json
  status                     what's running, and what's selected

A normal session - drop a clip in voices/, then:

  ./voices.sh prepare
  ./voices.sh test mysample
  ./voices.sh use mysample

Ports: both servers default to 8899, so run one elsewhere -
  CHATTERBOX_PORT=8900 ./run.sh
Override the lookup with CHATTERBOX_URL / SOURCE_URL.
EOF
}

case "${1:-}" in
    prepare) shift; cmd_prepare "$@" ;;
    list)   shift; cmd_list "$@" ;;
    grab)   shift; cmd_grab "$@" ;;
    from)   shift; cmd_from "$@" ;;
    test)   shift; cmd_test "$@" ;;
    use)    shift; cmd_use "$@" ;;
    status) shift; cmd_status "$@" ;;
    ""|-h|--help|help) usage ;;
    *) usage; exit 1 ;;
esac
