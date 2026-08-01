import platform

if platform.system() == "Linux":
    from .linux_keyboard import start_keyboard
elif platform.system() == "Windows":
    from .windows_keyboard import start_keyboard
elif platform.system() == "Darwin":
    from .mac_keyboard import start_keyboard
else:
    raise RuntimeError(f"Unsupported operating system: {platform.system()}")