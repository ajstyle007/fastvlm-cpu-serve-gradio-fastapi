import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import base64
import os

API_URL = "http://127.0.0.1:8000"


#Example prompts 
EXAMPLE_PROMPTS = [
    "Describe this image in detail.",
    "What is happening in this image?",
    "What objects can you see in this image?",
    "Describe the mood and atmosphere of this image.",
    "What is the main subject of this image?",
    "Is there any text visible in this image?",
]

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
        **MobileCLIP-L (FastViT) + Qwen2-0.5B** — Fast multimodal(Vision Language Model) inference pipeline

        Upload an image and ask a question about it.
        """, elem_classes="header")

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

                # Example prompts
                gr.Markdown("**Quick prompts:**")
                with gr.Row():
                    for i, p in enumerate(EXAMPLE_PROMPTS[:3]):
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

    return demo

def run_inference(image: Image.Image, prompt: str):

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
        # 1. Use requests.post with stream=True to prevent full payload buffering
        response = requests.post(
            f"{API_URL}/predict",
            files={"image": ("image.jpg", buf, "image/jpeg")},
            data={"prompt": prompt},
            stream=True,  # CRITICAL: Stream the incoming API bytes
            timeout=300
        )

        print("STATUS:", response.status_code)
        print("HEADERS:", response.headers)
        
        if response.status_code == 200:
            partial_text = ""
            # 2. Iterate over small token chunks as they are flushed by the FastAPI server
            for chunk in response.iter_content(chunk_size=16, decode_unicode=True):
                # print("CHUNK:", repr(chunk))
                if chunk:
                    partial_text += chunk
                    yield partial_text  # CRITICAL: yield triggers token streaming in UI
        else:
            yield f"API error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        yield "Cannot connect to API. Make sure api.py is running on port 8000."
    except Exception as e:
        yield f"Error: {str(e)}"


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )