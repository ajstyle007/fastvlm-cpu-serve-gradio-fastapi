# FastVLM CPU Serve — Gradio + FastAPI

<img width="1885" height="950" alt="Screenshot 2026-06-28 180204" src="https://github.com/user-attachments/assets/ecc9e5d7-be12-42ed-ba60-b374c0231c17" />


This repository contains the CPU deployment code for FastVLM with two variants:

- `fastvlm-cpu-serve_1024` — higher-quality image encoder at 1024×1024 resolution
- `fastvlm-cpu-serve_512` — faster image encoder at 512×512 resolution

- For 512 resolution: https://musk12-fastvlm-cpu-inference-demo.hf.space
- For 1024 resolution: https://musk12-fastvlm-fast-inference-on-cpu.hf.space

Both folders provide a complete Gradio frontend + FastAPI backend deployment, including voice, camera, and text-based multimodal interfaces.

## What this repo provides

- **FastAPI inference backend** (`stream_api.py`)
- **Persistent GGUF LLM server** (`fastvlm_server`, compiled from `fastvlm_infer_v3.c`)
- **ONNX vision encoder** for image preprocessing and embedding
- **Gradio UI** for image question answering, with support for voice prompt and spoken response
- **Live camera snapshot mode** for real-time image analysis
- **Deployment-ready layout** for both 1024 and 512 resolution models

## Why two variants?

### `fastvlm-cpu-serve_1024`
- Uses the 1024×1024 vision encoder
- Higher visual detail and quality
- Slightly slower than 512 resolution
- Best for tasks where fine image detail matters

### `fastvlm-cpu-serve_512`
- Uses the optimized 512×512 vision projector model
- Faster vision encoding
- Smaller runtime memory usage
- Best for low-latency CPU inference

## Multimodal capabilities

<img width="1857" height="1036" alt="Screenshot 2026-06-28 175821" src="https://github.com/user-attachments/assets/0b8a76a7-2c0e-4d07-9a9d-583c530facb6" />


These deployments support:

- **Image + text input**
- **Voice prompt input** via Whisper STT
- **Spoken response output** via Edge TTS
- **Camera snapshot analysis** in Gradio
- **Streaming generation** from the FastAPI backend

## Voice models used

### Speech-to-Text (STT)
- `faster-whisper`
- Model: `tiny`
- Device: CPU
- Quantization: `int8`

### Text-to-Speech (TTS)
- `edge-tts`
- Voice: `en-US-AriaNeural`

## Folder structure

Each variant folder contains:

- `stream_api.py` — FastAPI server that loads ONNX + starts the persistent LLM server
- `fastvlm_server` — compiled llama.cpp-based server binary
- `fastvlm_qwen2_q4km.gguf` — quantized Qwen2 model file
- `vision_encoder_fp32.onnx` or `vision_projector_v1_standalone.onnx` — ONNX vision encoder
- `gradio_app.py` — text + image Gradio demo
- `gradio_app_voice.py` / `gradio_3i1_new.py` — voice and camera-enabled Gradio demos
- required shared libraries: `libllama.so`, `libggml.so`, `libggml-cpu.so`, `libggml-base.so`

## Setup

1. Clone the repo:
```bash
git clone https://github.com/ajstyle007/fastvlm-cpu-serve-gradio-fastapi.git
cd fastvlm-cpu-serve-gradio-fastapi
```

2. Create a Python environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install gradio fastapi uvicorn onnxruntime pillow requests numpy faster-whisper edge-tts httpx
```

4. Confirm the required model files exist in each folder:
- `fastvlm_qwen2_q4km.gguf`
- `vision_encoder_fp32.onnx` or `vision_projector_v1_standalone.onnx`
- `fastvlm_server`
- required `.so` libraries

## Running the server

### 1024 variant
```bash
cd fastvlm-cpu-serve_1024
python stream_api.py
```

### 512 variant
```bash
cd fastvlm-cpu-serve_512
python stream_api.py
```

Once the server is running, start a Gradio interface in a second terminal:

```bash
python gradio_app.py
```

or for voice-enabled UI:

```bash
python gradio_app_voice.py
```

## API endpoints

The FastAPI server exposes:

- `GET /` — health/info endpoint
- `GET /health` — service health and model status
- `POST /predict` — image + prompt inference endpoint

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@./GOT.jpg" \
  -F "prompt=Describe this image in detail."
```

## Notes on deployment

- `fastvlm_server` is launched once by `stream_api.py` and stays live.
- The backend writes image embeddings to a temporary `.bin` file and sends it to `fastvlm_server`.
- `fastvlm_server` returns streamed output tokens until the `---END---` sentinel.
- This design avoids reloading the LLM for every request.

<img width="1860" height="1062" alt="Screenshot 2026-06-28 175909" src="https://github.com/user-attachments/assets/d2c88fcf-b3aa-4596-ad61-71580e8bf198" />

## Hugging Face deployment

This repository is designed for easy deployment on Hugging Face Spaces:
- Each variant can be deployed separately
- The Gradio app serves the frontend
- The FastAPI backend can run alongside or inside the same environment

## Recommended usage

- Use `fastvlm-cpu-serve_1024` when you need the best visual quality and detailed reasoning.
- Use `fastvlm-cpu-serve_512` when you need faster CPU response and lower latency.
- Use the voice-enabled apps for hands-free interaction.
- Use the live camera app for quick snapshot-based analysis.

## License

This repo follows the same license as the imported FastVLM and llama.cpp components. Check the included `LICENSE` files in the repository.
