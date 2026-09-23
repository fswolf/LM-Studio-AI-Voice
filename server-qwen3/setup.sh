#!/usr/bin/env bash
# Build qwentts.cpp and fetch the weights.
#
# No venv, no pip, no torch. This is a C++ binary and two GGUF files -
# which is the main reason it's worth trying on this machine: none of
# the ROCm-vs-CUDA-wheel problems that make Python TTS setups fragile
# apply, because there is no Python in the inference path at all.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

REPO="${QWEN_REPO:-$PWD/qwentts.cpp}"
SIZE="${QWEN_SIZE:-0.6b}"
QUANT="${QWEN_QUANT:-Q8_0}"
MODE="${QWEN_MODE:-customvoice}"
BACKEND="${QWEN_BACKEND:-vulkan}"

echo "== checking what's here =="
missing=()
for tool in git cmake curl; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || missing+=("gcc-c++")

if [ "$BACKEND" = vulkan ]; then
    # The shader toolchain is four separate Fedora packages and the
    # build needs all of them. Checked here rather than discovered one
    # at a time eight minutes into a compile.
    command -v glslc >/dev/null 2>&1 || missing+=("glslc")
    command -v glslangValidator >/dev/null 2>&1 || missing+=("glslang")

    # SPIRV-Headers is a headers-and-cmake-config package with no
    # binary, so it has to be looked for on disk.
    spirv=""
    for dir in /usr/lib64/cmake /usr/lib/cmake /usr/share/cmake \
               /usr/lib64/cmake/SPIRV-Headers /usr/share/cmake/SPIRV-Headers; do
        [ -d "$dir" ] || continue
        if find "$dir" -maxdepth 2 -iname 'SPIRV-Headers*onfig.cmake' \
             -o -maxdepth 2 -iname 'spirv-headers-config.cmake' 2>/dev/null \
           | grep -q .; then spirv=yes; break; fi
    done
    [ -n "$spirv" ] || missing+=("spirv-headers")

    [ -e /usr/share/vulkan/icd.d ] || missing+=("vulkan-loader + mesa-vulkan-drivers")
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing: ${missing[*]}"
    echo
    echo "  sudo dnf install git cmake gcc-c++ curl \\"
    echo "                   vulkan-loader vulkan-headers vulkan-tools \\"
    echo "                   mesa-vulkan-drivers \\"
    echo "                   glslc glslang spirv-headers spirv-tools"
    echo
    echo "(Debian/Ubuntu: glslc is in glslc, the rest in glslang-tools,"
    echo " spirv-headers, spirv-tools, libvulkan-dev, mesa-vulkan-drivers)"
    exit 1
fi

if [ "$BACKEND" = vulkan ] && command -v vulkaninfo >/dev/null 2>&1; then
    echo "  GPU: $(vulkaninfo --summary 2>/dev/null | grep -m1 deviceName | cut -d= -f2- | xargs || echo '?')"
fi

echo
echo "== qwentts.cpp =="
if [ -d "$REPO/.git" ]; then
    echo "  already cloned; pulling"
    git -C "$REPO" pull --ff-only --recurse-submodules
else
    git clone --recurse-submodules \
        https://github.com/ServeurpersoCom/qwentts.cpp.git "$REPO"
fi

echo
echo "== build ($BACKEND) =="
cd "$REPO"
case "$BACKEND" in
    vulkan) ./buildvulkan.sh ;;
    cpu)    ./buildcpu.sh ;;
    cuda)   ./buildcuda.sh ;;
    *)      echo "unknown backend: $BACKEND"; exit 1 ;;
esac

[ -x build/tts-server ] || [ -x build/bin/tts-server ] || {
    echo "!! tts-server didn't build. The compiler output above says why."
    exit 1
}

echo
echo "== weights =="
mkdir -p models
# voicedesign only exists at 1.7B.
[ "$MODE" = voicedesign ] && SIZE=1.7b
BASE="https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main"

for file in "qwen-talker-${SIZE}-${MODE}-${QUANT}.gguf" \
            "qwen-tokenizer-12hz-${QUANT}.gguf"; do
    if [ -s "models/$file" ]; then
        echo "  have $file"
    else
        echo "  downloading $file"
        # -C - resumes, which matters on a 1GB file over a flaky line.
        curl -fL -C - -o "models/$file" "$BASE/$file"
    fi
done

echo
du -h models/*.gguf 2>/dev/null || true
echo
echo "Done. Start it with ./run.sh"
