<img width="1210" height="1090" alt="Screenshot_20260730_153535" src="https://github.com/user-attachments/assets/d080fc10-626d-4796-a2e0-9a1589b2b36e" />
# AI Voice Assistant

A local AI voice assistant powered by:

- 🎤 Faster-Whisper (Speech-to-Text)
- 🧠 LM Studio (Local LLM)
- 🗣️ Kokoro TTS (Text-to-Speech)
- 🎹 Global keyboard hotkeys
- 💬 Rich terminal interface

Everything runs locally. No cloud APIs required.

---

# Requirements

- Python 3.12+
- LM Studio
- A downloaded language model
- LM Studio Local Server enabled

---

# Create a Virtual Environment

```bash
python3.12 -m venv ai-voice-venv

source ai-voice-venv/bin/activate
```

# Python Dependencies

```text
requests>=2.32.0
numpy>=2.2.0
scipy>=1.15.0
sounddevice>=0.5.2
silero-vad
faster-whisper>=1.1.0
ctranslate2>=4.6.0
kokoro>=0.9.4
torch>=2.8.0
torchaudio>=2.8.0
huggingface_hub>=0.34.0
pynput>=1.8.1
evdev>=1.9.2
rich>=14.1.0
wcwidth
```

Or simply install everything with:

```bash
pip install \
requests \
numpy \
scipy \
sounddevice \
silero-vad \
faster-whisper \
ctranslate2 \
kokoro \
torch \
torchaudio \
huggingface_hub \
pynput \
evdev \
rich \
wcwidth
```

---

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# Linux Dependencies

## Debian / Ubuntu

```bash
sudo apt install \
python3-dev \
build-essential \
portaudio19-dev \
ffmpeg
```

## Fedora

```bash
sudo dnf install \
ffmpeg \
portaudio-devel \
python3-devel \
gcc
```

---

# macOS Dependencies

Install Homebrew if needed, then:

```bash
brew install portaudio ffmpeg
```

---

# Running LM Studio

1. Install LM Studio
2. Download a model
3. Start the Local Server
4. Verify it is running on:

```
http://localhost:1234
```

---

# Run the Assistant

```bash
python main.py
```

---

# Controls

| Key | Action |
|------|--------|
| Home | Start / Stop Voice Recording / Stop TTS |
| Enter | Send Typed Message |
| Esc | Quit |

---

# Reminders (Experemental)

```
The assistant supports natural language reminders.

You can create reminders by simply talking normally:

Example:

> "Hey Luna, remind me in 10 minutes to clean the desk."

The assistant will:

1. Understand the reminder request.
2. Extract the time duration.
3. Extract the task.
4. Schedule the reminder.
5. Notify you when the reminder time is reached.
```

---

# Project Structure

```text
ai-voice/
│
├── main.py
├── assistant.py
├── speech.py
├── llm.py
├── config.py
├── state.py
├── ui.py
│
├── input/
│   ├── __init__.py
│   ├── linux_keyboard.py
│   └── windows_keyboard.py
│
├── agent/
│   ├── agent.json
│   └── memory.json
├── history/
├── reminders/
│
└── README.md
```

---

# Features

- Local speech recognition
- Local language model
- Local text-to-speech
- Configurable AI personality
- Configurable memory
- Cross-platform architecture
- Rich terminal interface
- Basic Reminder System

---

# Planned Features

- [ ] Web search
- [ ] Tool calling

---

# License

MIT License
