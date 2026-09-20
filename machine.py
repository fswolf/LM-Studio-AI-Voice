"""What the machine underneath her is doing.

One 16GB card runs the model, and often ComfyUI or a game as well. "Can
I load a bigger model right now" is a real question with a real answer,
and until now the only way she could answer it was to screenshot a
terminal and read the numbers back - which works, costs a vision round
trip, and is wrong the moment something allocates.

Everything here reads sysfs and /proc directly rather than shelling out
to rocm-smi or nvidia-smi. amdgpu exports all of it already, so there is
nothing to install, nothing to parse out of a table whose format changes
between releases, and no subprocess to hang. nvidia-smi is used only as
the NVIDIA path, where sysfs doesn't expose the same numbers - which
matters later, if the AI server happens to be green.
"""
import glob
import os
import shutil
import subprocess


def _read(path, cast=int):
    try:
        with open(path) as handle:
            return cast(handle.read().strip())
    except (OSError, ValueError):
        return None


def _gb(value_bytes):
    return value_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def _amd_cards():
    """Every amdgpu card's sysfs device directory."""
    found = []

    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        device = os.path.join(card, "device")

        # mem_info_vram_total is the tell: it exists on amdgpu and not
        # on the render nodes or on other drivers.
        if os.path.exists(os.path.join(device, "mem_info_vram_total")):
            found.append(device)

    return found


def _hwmon_value(device, filename, scale=1.0):
    for hwmon in glob.glob(os.path.join(device, "hwmon", "hwmon*")):
        value = _read(os.path.join(hwmon, filename))

        if value is not None:
            return value * scale

    return None


def _amd_gpu(device):
    total = _read(os.path.join(device, "mem_info_vram_total"))
    used = _read(os.path.join(device, "mem_info_vram_used"))

    if total is None:
        return None

    return {
        "name": _card_name(device),
        "vram_total": _gb(total),
        "vram_used": _gb(used) if used is not None else None,
        "busy": _read(os.path.join(device, "gpu_busy_percent")),
        # Millidegrees and microwatts, which is what hwmon deals in.
        "temp": _hwmon_value(device, "temp1_input", 0.001),
        "watts": _hwmon_value(device, "power1_average", 0.000001),
    }


def _card_name(device):
    """A readable name, falling back through what sysfs offers."""
    for filename in ("product_name", "device"):
        value = _read(os.path.join(device, filename), str)

        if value and not value.startswith("0x"):
            return value

    return "GPU"


def _nvidia_gpus():
    """The NVIDIA path. sysfs doesn't carry VRAM there, so this is the
    one place a subprocess earns its keep."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        output = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
             "temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )

        if output.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []

    gpus = []

    for line in output.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < 6:
            continue

        def number(text):
            try:
                return float(text)
            except ValueError:
                return None

        gpus.append({
            "name": parts[0],
            # nvidia-smi reports MiB.
            "vram_total": (number(parts[1]) or 0) / 1024,
            "vram_used": (number(parts[2]) or 0) / 1024,
            "busy": number(parts[3]),
            "temp": number(parts[4]),
            "watts": number(parts[5]),
        })

    return gpus


def gpus():
    found = [g for g in (_amd_gpu(d) for d in _amd_cards()) if g]

    return found or _nvidia_gpus()


# ---------------------------------------------------------------------------
# Memory and disk
# ---------------------------------------------------------------------------
def memory():
    """(total_gb, available_gb). MemAvailable, not MemFree - free
    memory on Linux is mostly cache and reads alarmingly low."""
    values = {}

    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()

                if parts:
                    values[key] = int(parts[0]) / (1024 ** 2)  # kB -> GiB
    except (OSError, ValueError):
        return None, None

    return values.get("MemTotal"), values.get("MemAvailable")


def disk(path=None):
    try:
        usage = shutil.disk_usage(path or os.path.expanduser("~"))
    except OSError:
        return None, None

    return _gb(usage.total), _gb(usage.free)


def uptime():
    seconds = _read("/proc/uptime", lambda text: float(text.split()[0]))

    if seconds is None:
        return ""

    hours, minutes = divmod(int(seconds) // 60, 60)
    days, hours = divmod(hours, 24)

    if days:
        return f"{days}d {hours}h"

    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def describe():
    """Everything, as lines the model can read back conversationally."""
    lines = []

    for gpu in gpus():
        parts = [gpu["name"]]

        if gpu["vram_used"] is not None:
            free = gpu["vram_total"] - gpu["vram_used"]
            parts.append(
                f"VRAM {gpu['vram_used']:.1f} of {gpu['vram_total']:.1f} GB used "
                f"({free:.1f} GB free)"
            )
        else:
            parts.append(f"VRAM {gpu['vram_total']:.1f} GB")

        if gpu["busy"] is not None:
            parts.append(f"{gpu['busy']:.0f}% busy")

        if gpu["temp"] is not None:
            parts.append(f"{gpu['temp']:.0f}C")

        if gpu["watts"] is not None:
            parts.append(f"{gpu['watts']:.0f}W")

        lines.append("GPU: " + ", ".join(parts))

    if not lines:
        lines.append("GPU: no readable GPU found.")

    total, available = memory()

    if total and available:
        lines.append(
            f"RAM: {total - available:.1f} of {total:.1f} GB used "
            f"({available:.1f} GB available)"
        )

    total, free = disk()

    if total:
        lines.append(f"Disk (home): {free:.0f} GB free of {total:.0f} GB")

    running = uptime()

    if running:
        lines.append(f"Up {running}")

    try:
        import lmstudio

        lines.append(f"LM Studio: {lmstudio.label()}")
    except Exception:
        pass

    return "\n".join(lines)
