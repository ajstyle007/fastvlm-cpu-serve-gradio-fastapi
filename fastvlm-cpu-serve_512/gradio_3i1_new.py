import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import base64
import os
import time
import io
import threading
from faster_whisper import WhisperModel
import edge_tts
import tempfile
import asyncio
import httpx

API_URL = "http://127.0.0.1:8000"


async def run_inference(image: Image.Image, prompt: str):

    print("IMAGE TYPE:", type(image))
    print("PROMPT TYPE:", type(prompt))
    print("PROMPT:", repr(prompt))

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if image is None:
        yield "Please upload an image."
        return

    if not prompt.strip():
        prompt = "Describe this image in detail."

    # Convert PIL image to bytes
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # 1. Stream the request so we don't block on the full payload
            async with client.stream(
                "POST",
                f"{API_URL}/predict",
                files={"image": ("image.jpg", buf, "image/jpeg")},
                data={"prompt": prompt},
            ) as response:

                print("STATUS:", response.status_code)
                print("HEADERS:", response.headers)

                if response.status_code == 200:
                    partial_text = ""
                    # 2. Iterate over small token chunks as they are flushed by the FastAPI server
                    async for chunk in response.aiter_text(chunk_size=16):
                        # print("CHUNK:", repr(chunk))
                        if chunk:
                            partial_text += chunk
                            yield partial_text  # CRITICAL: yield triggers token streaming in UI
                else:
                    error_body = await response.aread()
                    yield f"API error {response.status_code}: {error_body.decode(errors='ignore')}"

    except httpx.ConnectError:
        yield "Cannot connect to API. Make sure api.py is running on port 8000."
    except Exception as e:
        yield f"Error: {str(e)}"

def process_static_image():

    #Example prompts 
    EXAMPLE_PROMPTS = [
        "Describe this image in detail.",
        "What is happening in this image?",
        "What objects can you see in this image?",
        "Describe the mood and atmosphere of this image.",
        "What is the main subject of this image?",
        "Is there any text visible in this image?",
        ]

    gr.Markdown("**Upload an image and ask a question about it.**", elem_classes="header")

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=400,
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe this image in detail.",
                value="Describe this image in detail.",
                lines=2,
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 Run Inference",
                    variant="primary",
                    scale=2
                )
                clear_btn = gr.Button(
                    "🗑 Clear",
                    variant="secondary",
                    scale=1
                )

            # Example prompts setup
            gr.Markdown("**Quick prompts:**")
            with gr.Row():
                for p in EXAMPLE_PROMPTS[:3]:
                    gr.Button(p, size="sm").click(
                        fn=lambda x=p: x,
                        outputs=prompt_input
                    )
            with gr.Row():
                for p in EXAMPLE_PROMPTS[3:]:
                    gr.Button(p, size="sm").click(
                        fn=lambda x=p: x,
                        outputs=prompt_input
                    )

        # Right column — output
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Model Response",
                lines=20,
                max_lines=30,
            )
            gr.Markdown("""
            <div class="model-info">
            Vision encoder: MobileCLIP-L (FastViT, 125M params) → ONNX<br>
            Language model: Qwen2-0.5B (Q4_K_M, 463MB) → GGUF<br>
            Image tokens: 256 × 896-dim embeddings
            </div>
            """)

    # Examples section
    gr.Markdown("### 📸 Try with example image")
    if os.path.exists("GOT.jpg"):
        gr.Examples(
                examples=[
                    ["GOT.jpg", "Describe what you see in this image in detail."],
                    ["GOT.jpg", "What is the mood and atmosphere of this scene?"],
                    ["GOT.jpg", "Who appears to be the main character and what are they doing?"],
                ],
                inputs=[image_input, prompt_input]
            )

    # Event handlers
    submit_btn.click(
        fn=run_inference, inputs=[image_input, prompt_input],
        outputs=output_text, show_progress=True,
    )

    prompt_input.submit(
        fn=run_inference, inputs=[image_input, prompt_input],
        outputs=output_text, show_progress=True,
    )

    clear_btn.click(
        fn=lambda: (None, "Describe this image in detail.", ""),
        outputs=[image_input, prompt_input, output_text],
    )


# ==========================================
# 3. TAB 2: LIVE CAMERA SNAPSHOT 
# ==========================================

def live_camera_inference():

    PROMPT_PRESETS = [
        "Describe briefly.",
        "What's in my hand?",
        "What am I doing?",
        "Any text visible?",
    ]

    def _status_html(state: str, ttft: float = 0.0, total: float = 0.0) -> str:
        # kbd = "background:#2d2d44;padding:1px 6px;border-radius:3px;color:#cdd6f4;font-size:11px;"
        base = "font:12px/1.5 'JetBrains Mono',monospace;padding:7px 14px;border-radius:4px;text-align:center;letter-spacing:.04em;"
        if state == "idle":
            return (f'<div class="cam-status-bar idle" style="{base}background:#1a1a2e;color:#6c7086;">')
        if state == "processing":
            return (f'<div class="cam-status-bar busy" style="{base}background:#1a1a2e;color:#f9e2af;">'
                    f'⏳ PROCESSING — please wait…</div>')
        if state == "done":
            return (f'<div class="cam-status-bar idle" style="{base}background:#1a1a2e;color:#a6e3a1;">'
                    f'✔ &nbsp;TTFT <strong>{ttft:.0f} ms</strong> &nbsp;·&nbsp; total <strong>{total:.0f} ms</strong>')
        return ""

    gr.Markdown("## 🎥 Live Camera Analytics", elem_classes="header")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            # Clean snapshot webcam — no streaming, no record button loop
            webcam = gr.Image(
                sources=["webcam"],
                streaming=False,
                type="pil",
                label="Click the 📷 icon to capture & analyse",
                height=360,
            )
            status_bar = gr.HTML(value=_status_html("idle"))

        with gr.Column(scale=1):
            response_box = gr.Textbox(
                label="📝 Model Output",
                lines=7,
                max_lines=12,
                interactive=False,
                placeholder="Click the camera icon in the feed to capture and analyse…",
            )
            prompt_selector = gr.Radio(
                choices=PROMPT_PRESETS,
                value=PROMPT_PRESETS[0],
                label="🎯 Select Prompt",
            )

    is_busy = gr.State(False)

    async def on_capture(frame, busy, selected_prompt):
        if busy or frame is None:
            yield gr.skip(), gr.skip(), gr.skip()
            return

        yield _status_html("processing"), True, "⏳ Analysing frame…"

        t_start = time.time()
        ttft_ms = 0.0
        final_text = ""

        try:
            async for partial in run_inference(frame, selected_prompt):
                if partial:
                    if not final_text:
                        ttft_ms = (time.time() - t_start) * 1000
                    final_text = partial
            total_ms = (time.time() - t_start) * 1000
            caption = final_text.strip() or "Model returned an empty response."
        except Exception as exc:
            total_ms = (time.time() - t_start) * 1000
            caption = f"⚠ Error: {exc}"

        yield _status_html("done", ttft=ttft_ms, total=total_ms), False, caption

    webcam.change(
        fn=on_capture,
        inputs=[webcam, is_busy, prompt_selector],
        outputs=[status_bar, is_busy, response_box],
        queue=True,
    )



# --- Improved Inference Processing ---
def run_inference_voice(image: Image.Image, prompt: str):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if image is None:
        yield "Please upload an image."
        return
        
    if not prompt.strip():
        prompt = "Describe this image in detail."

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85) # Reduced quality slightly to 85% to save bandwidth & memory
    buf.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"image": ("image.jpg", buf, "image/jpeg")},
            data={"prompt": prompt},
            stream=True,  
            timeout=300
        )
        
        if response.status_code == 200:
            partial_text = ""
            # FIX: Increased chunk_size to 128 bytes to significantly reduce yield frequency
            for chunk in response.iter_content(chunk_size=128, decode_unicode=True):
                if chunk:
                    partial_text += chunk
                    yield partial_text  
        else:
            yield f"API error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        yield "Cannot connect to API. Make sure api.py is running on port 8000."
    except Exception as e:
        yield f"Error: {str(e)}"



whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)

def transcribe_audio(audio_path):

    if audio_path is None:
        return None

    segments, _ = whisper_model.transcribe(audio_path)

    return " ".join(
        segment.text
        for segment in segments
    )

def handle_audio_transcription(audio_path, fallback_prompt):

    voice_prompt = transcribe_audio(audio_path)
    
    if voice_prompt and voice_prompt.strip():
        return voice_prompt.strip()
        
    return fallback_prompt


async def text_to_speech_edge(text):
    try:
        if not text or not text.strip():
            return None
        
        clean_text = text.replace("Streaming error:", "").strip()
        
        # Premium Natural Voice: 'en-US-ChristopherNeural' (Male) ya 'en-US-EmmaNeural' (Female)
        voice = "en-US-ChristopherNeural"
        output_filename = "response_voice.mp3"
        
        # Edge TTS Communicate object pipeline
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_filename)
        
        return output_filename
    except Exception as e:
        print(f"Edge TTS Conversion Error: {str(e)}")
        return None


def live_camera_voice_infer():


    gr.Markdown("### 🎙️ Voice Prompt & Image Analysis", elem_classes="header")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Clean snapshot webcam — no streaming, no record button loop
            webcam = gr.Image(
                sources=["webcam"],
                streaming=False,
                type="pil",
                label="Click the 📷 icon to capture & analyse",
                height=300,
            )

            input_audio = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Record or Upload Audio Prompt"
            )

            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):

            with gr.Column():
                # Outputs
                output_text = gr.Textbox(label="Model Response", interactive=False)
                output_audio = gr.Audio(label="Response Audio (TTS)", interactive=False, autoplay=True)


    async def process_voice_and_predict(image, audio_path):
        final_prompt = handle_audio_transcription(audio_path, fallback_prompt="Describe this image in detail.")
        
        last_text = ""
        async for text_out in run_inference(image, final_prompt):
            if text_out:
                last_text = text_out
                yield last_text, gr.skip()

        if last_text.strip():
            print(f"Generating Premium Edge TTS Audio...")
            audio_file = await text_to_speech_edge(last_text)
            yield last_text, audio_file

    # Click event trigger
    submit_btn.click(
        fn=process_voice_and_predict,
        inputs=[webcam, input_audio],
        outputs=[output_text, output_audio],
        queue=True
    )


def live_camera_continous_inference():
    gr.Markdown("### 🎥 Live Camera Frame Analytics", elem_classes="header")

    PROMPT_PRESETS = [
        "Describe this image in one brief sentence.",
        "What is in my hand?",
        "Identify the main objects visible here.",
        "What is the person doing in this frame?",
        "Is there any text or book visible?"
    ]

    with gr.Row():
        # LEFT COLUMN: Live Camera Feed with continuous streaming & System Kill Button
        with gr.Column(scale=1, min_width=400):
            webcam = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Live Camera Feed", height=380)
            stream_state = gr.State(False)
            
            with gr.Row():
                toggle_btn = gr.Button("▶ Start Live Analytics", variant="primary", scale=2)
                kill_cam_btn = gr.Button("🛑 Kill Camera UI", variant="stop", scale=1)

        # RIGHT COLUMN: Preset Selector, JSON Metric Monitor & Text Panel
        with gr.Column(scale=1):
            prompt_selector = gr.Radio(
                choices=PROMPT_PRESETS,
                value=PROMPT_PRESETS[0],
                label="🎯 Choose Active VLM Directive / Prompt"
            )
            
            response_box = gr.Textbox(
                label="VLM Caption Output (Updates every 4 seconds)", 
                lines=5, 
                max_lines=7, 
                interactive=False, 
                placeholder="System Paused. Click 'Start Live Analytics' to begin..."
            )
    
            ttft_display = gr.JSON(
                label="⏱️ Hardware Latency Monitor", 
                value={"TTFT (Time to First Token)": "0.00 ms", "Total Pipeline Execution": "0.00 ms"}
            )
            

    # State variables time tracking ke liye
    last_run = gr.State(0.0)

    # Core Snapshot Execution Function
    async def on_frame(frame, last_run_time, is_streaming, selected_prompt):
        now = time.time()

        if not is_streaming:
            return gr.skip(), last_run_time, gr.skip()
        
        if frame is None or (now - last_run_time) < 4.0:
            return gr.skip(), last_run_time, gr.skip()
        
        print(f"--- [!] 4 SECONDS PASSED: FETCHING FOR PROMPT: '{selected_prompt}' ---")
        
        request_start_time = time.time()
        ttft_recorded = 0.0
        final_caption = ""
        
        try:
            inference_stream = run_inference(frame, selected_prompt)
            
            async for partial in inference_stream:
                if partial:
                    if not final_caption:
                        ttft_recorded = (time.time() - request_start_time) * 1000 
                    final_caption = partial 
            
            # Formulating clean dictionary output for gr.JSON component
            latency_metrics = {
                "TTFT (Time to First Token)": f"{ttft_recorded:.2f} ms" if ttft_recorded > 0 else "N/A",
                "Total Pipeline Execution": f"{(time.time() - request_start_time)*1000:.2f} ms"
            }

            if final_caption.strip():
                return final_caption, now, latency_metrics
            else:
                return "Model generated an empty response.", now, latency_metrics

        except Exception as e:
            return f"Streaming error: {str(e)}", now, {"Error Status": f"Pipeline Failure: {str(e)}"}

    # Analytics Toggle Controller Function
    def toggle_stream(current_state):
        new_state = not current_state
        if new_state:
            return new_state, gr.update(value="⏹ Stop Live Analytics", variant="stop"), "Starting API pipeline..."
        else:
            return new_state, gr.update(value="▶ Start Live Analytics", variant="primary"), "System Paused."

    # Completely kills and purges the webcam element state container
    def absolute_kill_switch():             
        return (
            False,                                                             
            gr.update(value="▶ Start Live Analytics", variant="primary"),     
            "System Disconnected & Purged.",                                  
            gr.update(value=None),                                             
            {"System Health Monitor": "Offline / Purged / UI Killed"}                     
        )

    # Connect UI Interactions
    toggle_btn.click(
        fn=toggle_stream,
        inputs=[stream_state],
        outputs=[stream_state, toggle_btn, response_box]
    )

    # Connect the explicit hard kill switch button interface
    kill_cam_btn.click(
        fn=absolute_kill_switch,
        inputs=[],
        outputs=[stream_state, toggle_btn, response_box, webcam, ttft_display]
    )

    # Connect continuous streaming data injection channel pipeline loops
    webcam.stream(
        fn=on_frame,
        inputs=[webcam, last_run, stream_state, prompt_selector],
        outputs=[response_box, last_run, ttft_display], 
        show_progress="hidden",
        queue=True,
        concurrency_limit=1,
        concurrency_id="cam_stream"
    )



#Gradio UI 
def build_ui():


    with gr.Blocks(
        title="FastVLM — Inference on CPU",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .model-info { font-size: 0.85em; color: #666; }
        """
    ) as demo:

        # Header
        gr.Markdown("""
        # FastVLM — Inference on CPU 
        **MobileCLIP-L (FastViT) + Qwen2-0.5B** — Fast multimodal(Vision Language Model) inference pipeline**
        """)

        with gr.Tabs():
        
            # TAB 1: Normal Image Upload
            with gr.Tab("Image Upload"):
                process_static_image()

            # TAB 2: Live Camera Stream
            with gr.Tab("FastVLM Live Camera"):
                live_camera_inference()

            # TAB 3: Audio Input / Voice Prompt 
            with gr.Tab("live_camera_voice_infer"):
                live_camera_voice_infer()
            
            with gr.Tab("Continous text generation"):
                live_camera_continous_inference()

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=40,
        show_api=False, 
    )
    