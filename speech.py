import sounddevice as sd
import numpy as np
import state

from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from kokoro import KPipeline
from config import SAMPLE_RATE, VOICE

whisper = None
tts = None

def load_models():
    global whisper
    global tts

    whisper = WhisperModel("small",device="cpu",compute_type="int8")
    tts = KPipeline(lang_code="a",repo_id="hexgrad/Kokoro-82M")

def record_audio():
    chunks = []

    silence_seconds = 0
    started = False

    block_size = 1024
    silence_limit = 1.5
    max_record_time = 60

    stream = sd.InputStream(samplerate=SAMPLE_RATE,channels=1,blocksize=block_size,dtype="float32")

    stream.start()
    elapsed = 0

    while True:
        audio, overflow = stream.read(block_size)
        audio = audio[:,0]
        volume = np.sqrt(np.mean(audio ** 2))
        chunks.append(audio)
        elapsed += block_size / SAMPLE_RATE

        if volume > 0.035:
            started = True
            silence_seconds = 0
        else:
            if started:
                silence_seconds += block_size / SAMPLE_RATE

        if state.stop_listening:
            break
        if started and silence_seconds >= silence_limit:
            break
        if elapsed >= max_record_time:
            break

    stream.stop()
    stream.close()
    audio = np.concatenate(chunks)
    filename = "/tmp/input.wav"
    write(filename,SAMPLE_RATE,audio)

    return filename

def transcribe(filename):
    segments, info = whisper.transcribe(filename)

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()

def speak(text):
    state.stop_speaking = False

    audio = tts(text,voice=VOICE)

    for _, _, samples in audio:

        if state.stop_speaking:
            sd.stop()
            return

        sd.play(samples,24000)

        while sd.get_stream().active:
            if state.stop_speaking:
                sd.stop()
                return

            sd.sleep(50)