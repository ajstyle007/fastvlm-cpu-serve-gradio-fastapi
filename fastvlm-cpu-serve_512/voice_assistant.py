import sounddevice as sd
print(sd.query_devices())
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

import edge_tts
import os



# Load whisper once
model = WhisperModel("tiny")

# Record audio
def record_audio():
    fs = 16000

    print("Speak...")

    audio = sd.rec(
        int(5 * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    write("input.wav", fs, audio)

# Speech -> Text
def transcribe():
    segments, _ = model.transcribe("input.wav")

    text = " ".join(
        segment.text
        for segment in segments
    )

    return text

# Text -> Speech
async def speak(text):

    tts = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )

    await tts.save("output.mp3")

    os.system("start output.mp3")

# Main
record_audio()

text = transcribe()

print("You said:", text)

asyncio.run(
    speak(text)
)