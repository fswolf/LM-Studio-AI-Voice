#!/usr/bin/env bash
# Start the assistant from anywhere - the script knows where it lives.
cd "$(dirname "$(realpath "$0")")" || exit 1

# Activate the venv unless one is already active. Checked in the order
# they're likely to exist: the README's, then two common in-tree names.
if [ -z "$VIRTUAL_ENV" ]; then
    for venv in ~/ai-voice-venv ./ai-voice-venv ./venv; do
        if [ -f "$venv/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$venv/bin/activate"
            break
        fi
    done
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtualenv found - running against system python3." >&2
    echo "  python3.12 -m venv ~/ai-voice-venv" >&2
fi

# Both of these are warnings, never blockers: she degrades to a text
# chat without speech, and says so herself if the model is missing.
# Starting anyway beats a launcher that refuses to launch.
# Prints on stdout so the caller can capture it; the caller decides
# whether there is anything worth saying before it redirects to stderr.
check() {
    command -v curl >/dev/null || return 0
    curl -fsS --max-time 2 "$1" >/dev/null 2>&1 || echo "  $2"
}

warnings=$(
    check "http://localhost:1234/v1/models" \
          "LM Studio isn't answering on :1234 - enable its Local Server."
    check "http://127.0.0.1:8899/health" \
          "kokoro-reader isn't answering on :8899 - she'll start mute."
)

if [ -n "$warnings" ]; then
    echo "Starting anyway:" >&2
    echo "$warnings" >&2
    echo >&2
fi

exec python3 main.py "$@"
