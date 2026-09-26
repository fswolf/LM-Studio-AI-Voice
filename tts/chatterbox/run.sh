#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

source venv/bin/activate

# gfx1030 (RX 6800/6900/6950 XT) isn't in every ROCm build's official
# support list, and this is what makes it work anyway. Harmless on
# cards that don't need it.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"

# Same port kokoro-reader uses, so the assistant needs no change -
# stop one before starting the other.
export CHATTERBOX_PORT="${CHATTERBOX_PORT:-8899}"

exec python chatterbox_server.py
