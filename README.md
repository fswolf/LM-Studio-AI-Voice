<img width="1210" height="1090" alt="Screenshot_20260730_153535" src="https://github.com/user-attachments/assets/d080fc10-626d-4796-a2e0-9a1589b2b36e" />


/////////////////////////////////////////////////////////////////////
// REQUIRED
/////////////////////////////////////////////////////////////////////

Python 3.12+
requests>=2.32.0
numpy>=2.2.0
scipy>=1.15.0
sounddevice>=0.5.2
faster-whisper>=1.1.0
ctranslate2>=4.6.0
kokoro>=0.9.4
torch>=2.8.0
torchaudio>=2.8.0
huggingface_hub>=0.34.0
pynput>=1.8.1
evdev>=1.9.2
rich>=14.1.0

/////////////////////////////////////////////////////////////////////
// Enviroment
/////////////////////////////////////////////////////////////////////

python3.12 -m venv ai-voice-venv
source ai-voice-venv/bin/activate


/////////////////////////////////////////////////////////////////////
// PIP INSTALLS
/////////////////////////////////////////////////////////////////////

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


/////////////////////////////////////////////////////////////////////
// Linux Dependancys
/////////////////////////////////////////////////////////////////////

Debian:
sudo apt install \
python3-dev \
build-essential \
portaudio19-dev \
ffmpeg

Fedora:
sudo dnf install \
ffmpeg \
portaudio-devel \
python3-devel \
gcc

OSX:
brew install portaudio 


