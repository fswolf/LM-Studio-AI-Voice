#!/usr/bin/env bash
# Start the thought viewer.
# Opens http://127.0.0.1:8791 in your browser.
cd "$(dirname "$(realpath "$0")")/.." || exit 1

if [ -z "$VIRTUAL_ENV" ]; then
    for venv in ~/ai-voice-venv ./ai-voice-venv ./venv; do
        if [ -f "$venv/bin/activate" ]; then
            # shellcheck disable=SC1090
            source "$venv/bin/activate"
            break
        fi
    done
fi

exec python3 thought-viewer/viewer.py "$@"
