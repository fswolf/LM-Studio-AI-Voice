#!/usr/bin/env bash
# Build this server its own venv.
#
# Its own, deliberately. chatterbox-tts pins a torch that would fight
# with whatever the assistant's venv has, and the assistant only needs
# `requests` to talk to this thing - so there is no reason for them to
# share a python at all.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

VENV="${VENV:-venv}"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.2}"

# chatterbox is developed on 3.11 and several of its dependencies have
# no 3.12 wheels. Prefer 3.11 where it exists; warn rather than refuse
# if it doesn't, because it may well work and finding out is cheap.
PY=""
for candidate in python3.11 python3.10 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

case "$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')" in
    3.10|3.11) ;;
    *) echo "!! $PY is $($PY -V). chatterbox targets 3.11 and some of its"
       echo "   dependencies have no newer wheels. Trying anyway; if pip"
       echo "   starts building things from source, install python3.11:"
       echo "     sudo dnf install python3.11"
       echo ;;
esac

echo "== venv ($PY) =="
"$PY" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo
echo "== torch for ROCm =="
echo "   index: $ROCM_INDEX"
echo "   (override with ROCM_INDEX=..., or CPU=1 for a CPU-only build)"
if [ "${CPU:-0}" = "1" ]; then
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchaudio --index-url "$ROCM_INDEX"
fi

echo
echo "== chatterbox =="
# --no-deps on purpose: its requirements pin a CUDA torch, and pip will
# cheerfully replace the ROCm build you just installed with it. The
# rest of its dependencies are listed in requirements.txt instead.
pip install -r requirements.txt
pip install --no-deps chatterbox-tts

echo
echo "== checking the GPU =="
python - <<'PY'
import torch
print(f"  torch       {torch.__version__}")
print(f"  cuda avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device      {torch.cuda.get_device_name(0)}")
    print(f"  hip/rocm    {getattr(torch.version, 'hip', None)}")
else:
    print("  !! no GPU visible - it will run on CPU, which is slow but works.")
    print("     For ROCm on gfx1030 you may need:")
    print("       export HSA_OVERRIDE_GFX_VERSION=10.3.0")
PY

echo
echo "Done. Start it with ./run.sh"
