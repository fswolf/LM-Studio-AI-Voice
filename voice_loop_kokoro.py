#!/usr/bin/env python3
"""Standalone listen -> transcribe -> speak loop.

TTS is handled by the kokoro-reader server (github.com/fswolf/kokoro-reader),
so this script no longer loads Kokoro itself:

    KOKORO_VOICE=am_adam python3 server/kokoro_server.py
"""
import io
import os
import time
import wave

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# Settings
MODEL_SIZE = "base"
SAMPLE_RATE = 16000
KOKORO_URL = os.environ.get("KOKORO_URL", "http://127.0.0.1:8899").rstrip("/")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "am_adam")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
VOICE_VOLUME = 1.0          # 0.0 - 1.0+ (1.0 = normal, 1.5 = louder, etc.)

# VAD settings - these are more sensitive for quiet mics
START_THRESHOLD = 0.004     # bar to START recording - lower = picks up sooner (try 0.003 - 0.008)
CONTINUE_THRESHOLD = 0.0015 # bar to KEEP recording once started - lower = tolerates quieter pauses without cutting off
SMOOTHING = 2               # avg volume over this many blocks, reduces jitter/false cutoffs (try 2-4)
SILENCE_DURATION = 1.2      # seconds of true silence (below CONTINUE_THRESHOLD) before stopping
BLOCK_DURATION = 0.05       # smaller = faster reaction to speech starting
MAX_RECORD_SECONDS = 30
PRE_ROLL = 0.8              # keep a little audio before speech was detected

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # change device="cuda" if you have GPU set up for it

print(f"Using Kokoro server at {KOKORO_URL} (voice {KOKORO_VOICE})")
try:
    requests.get(f"{KOKORO_URL}/health", timeout=5).raise_for_status()
except requests.exceptions.RequestException as e:
    raise SystemExit(
        f"Kokoro server not reachable at {KOKORO_URL}: {e}\n"
        "Start it with:  python3 server/kokoro_server.py"
    )

def record_audio():
    print("\nWaiting for you to speak...")
    blocksize = int(SAMPLE_RATE * BLOCK_DURATION)
    buffer = []
    pre_buffer = []
    recent_vols = []
    triggered = False
    silence_start = None
    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=blocksize) as stream:
        while True:
            block, _ = stream.read(blocksize)
            raw_volume = np.sqrt(np.mean(block**2))  # RMS

            # smooth over the last few blocks so one quiet frame doesn't
            # falsely start the silence timer or miss a soft start
            recent_vols.append(raw_volume)
            if len(recent_vols) > SMOOTHING:
                recent_vols.pop(0)
            volume = np.mean(recent_vols)

            # Keep a short pre-roll buffer
            pre_buffer.append(block.copy())
            if len(pre_buffer) > int(PRE_ROLL / BLOCK_DURATION):
                pre_buffer.pop(0)

            if not triggered:
                if raw_volume > START_THRESHOLD:
                    print(f"Speech detected (vol: {raw_volume:.4f}), recording...")
                    buffer.extend(pre_buffer)  # add the pre-roll
                    triggered = True
                    silence_start = None
                    buffer.append(block.copy())
            else:
                buffer.append(block.copy())
                if volume > CONTINUE_THRESHOLD:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        print("Silence detected, stopping.")
                        break

            if triggered and (time.time() - start_time > MAX_RECORD_SECONDS):
                print("Hit max recording length.")
                break

    if not buffer:
        return np.array([], dtype='float32')
    return np.concatenate(buffer).flatten()

def transcribe(audio):
    print("Transcribing...")
    segments, _ = model.transcribe(audio, language="en")
    return "".join(seg.text for seg in segments).strip()

def speak(text):
    if not text:
        print("No text detected.")
        return

    print(f"You said: {text}")
    print("Generating speech...")

    try:
        response = requests.post(
            f"{KOKORO_URL}/tts",
            json={"text": text[:1200], "voice": KOKORO_VOICE, "speed": KOKORO_SPEED},
            timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"TTS failed: {e}")
        return

    with wave.open(io.BytesIO(response.content), "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    sd.play(audio * VOICE_VOLUME, samplerate=rate)
    sd.wait()

def main():
    print("Voice loop started. Press Ctrl+C to stop.\n")
    try:
        while True:
            audio = record_audio()
            if len(audio) < SAMPLE_RATE * 0.3:  # ignore very short clips
                continue
            text = transcribe(audio)
            speak(text)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
