#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Which voices exist depends on which talker is loaded:
#   customvoice  named speakers, nothing to set up  (default)
#   base         clones whatever .wav is in voices/
#   voicedesign  the voice is a sentence; 1.7B only
export QWEN_MODE="${QWEN_MODE:-customvoice}"
export QWEN_SIZE="${QWEN_SIZE:-0.6b}"
export QWEN_QUANT="${QWEN_QUANT:-Q8_0}"

# The standard TTS port, so anything already pointed at kokoro-reader -
# the assistant, the Firefox extension - finds this instead with no
# change. Only one thing can hold it, so stop kokoro-reader first.
#
# To run both and compare:  QWEN_PORT=8901 ./run.sh
export QWEN_PORT="${QWEN_PORT:-8899}"

exec python3 qwen_server.py
