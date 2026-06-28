import gradio as gr
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import tempfile
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# Load once
model = WhisperModel("tiny", device="cpu")

# Speech -> Text
def transcribe(audio_path):

    segments, _ = model.transcribe(audio_path)

    text = " ".join(
        segment.text
        for segment in segments
    )

    return text

# Text -> Speech
async def generate_audio(text):

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ).name

    tts = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )

    await tts.save(output_file)

    return output_file

# Main pipeline
def voice_pipeline(audio_path):

    # Audio -> Text
    text = transcribe(audio_path)

    print("User:", text)

    # Later replace this with FastVLM / LLM response
    response = f"You said: {text}"

    # Text -> Speech
    audio_response = asyncio.run(
        generate_audio(response)
    )

    return text, response, audio_response


demo = gr.Interface(
    fn=voice_pipeline,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"
    ),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Response"),
        gr.Audio(label="Voice Response")
    ],
    title="Voice Assistant"
)

demo.launch()