import os
import sys
import struct
import tempfile
import subprocess
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import asyncio
import time


# Config
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH  = os.path.join(BASE_DIR, "vision_encoder_fp32.onnx")
# ONNX_PATH  = os.path.join(BASE_DIR, "vision_encoder_safe_fp16.onnx")
GGUF_PATH  = os.path.join(BASE_DIR, "fastvlm_qwen2_q4km.gguf")
SERVER_BIN  = os.path.join(BASE_DIR, "fastvlm_server")

app = FastAPI(title="Custom FastVLM onnx gguf API",  
              description="Vision Language Model inference using MobileCLIP + Qwen2",
              version="1.0.0")


# Load ONNX session once at startup
ort_session = None

@app.on_event("startup")
async def load_models():
    global ort_session
    print("Loading ONNX vision encoder...")

    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print("Providers:", ort_session.get_providers())

    print("ONNX session ready ✅")

    await start_llm_server()

@app.on_event("shutdown")
async def shutdown():
    if llm_process:
        llm_process.stdin.close()
        await llm_process.wait()
        print("[api] LLM server stopped")

#Preprocessing
def expand_2_square(image: Image.Image):
    w, h = image.size
    if w == h:
        return image
    
    size = max(w, h)
    result = Image.new("RGB", (size, size), (0,0,0))

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    result.paste(image, (x_offset, y_offset))

    return result

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = expand_2_square(image)
    w, h  = image.size
    scale = 1024 / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h  = image.size
    left  = (w - 1024) // 2
    top   = (h - 1024) // 2
    image = image.crop((left, top, left + 1024, top + 1024))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 1024, 1024)

def encode_image(image : Image.Image):

    t0 = time.perf_counter()
    
    pixel_values = preprocess_image(image)

    t1 = time.perf_counter()

    embeddings = ort_session.run(["image_embeddings"], {"pixel_values": pixel_values})[0]

    t2 = time.perf_counter()

    print(
        f"[VISION] preprocess: {(t1-t0)*1000:.1f} ms"
    )

    print(
        f"[VISION] onnx inference: {(t2-t1)*1000:.1f} ms"
    )

    return embeddings[0]   # (256, 896)

def save_embeddings(embeddings: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
        n_tokens, n_embd = embeddings.shape

        # Write header: two int32 values
        f.write(struct.pack("ii", int(n_tokens), int(n_embd)))

        # Write float32 data
        f.write(embeddings.astype(np.float32).tobytes())
    return path
    

llm_process = None
llm_lock    = asyncio.Lock()   # one request at a time (server is single-threaded)

async def start_llm_server():
    global llm_process
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": BASE_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    }
    llm_process = await asyncio.create_subprocess_exec(
        SERVER_BIN, GGUF_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=BASE_DIR
    )
    print("[api] Waiting for LLM server to load model...")
    while True:
        line = await llm_process.stderr.readline()
        line = line.decode("utf-8", errors="ignore").strip()
        print(f"[llm] {line}")
        if "READY" in line:
            break
        if llm_process.returncode is not None:
            raise RuntimeError("LLM server died during startup")
    print("[api] LLM server ready ✅")
    
async def run_llm_stream(embed_path: str, prompt: str, request_start: float):
    """
    Send request to persistent LLM server via stdin pipe.
    Stream response tokens from stdout until ---END--- sentinel.
    Model stays loaded between requests — no per-request startup cost.
    """
    async with llm_lock:   # serialize: server handles one request at a time
        llm_start    = time.perf_counter()
        first_token  = True

        try:
            # Send request: embd_path\nprompt\n
            llm_process.stdin.write(
                (embed_path + "\n").encode()
            )
            llm_process.stdin.write(
                (prompt + "\n").encode()
            )
            await llm_process.stdin.drain()

            print(
                f"[TIMING] request sent to server: "
                f"{(time.perf_counter()-llm_start)*1000:.1f} ms"
            )

            # Stream response until ---END--- sentinel
            buffer = ""
            while True:
                chunk = await llm_process.stdout.read(16)
                if not chunk:
                    # Server died
                    print("[api] LLM server stdout closed unexpectedly")
                    break

                text = chunk.decode("utf-8", errors="ignore")
                buffer += text

                # Check if sentinel is in buffer
                if "---END---" in buffer:
                    # Yield everything before the sentinel
                    before, _ = buffer.split("---END---", 1)
                    if before:
                        if first_token:
                            now = time.perf_counter()
                            print(
                                f"[TIMING] TTFT from request start: "
                                f"{(now-request_start)*1000:.1f} ms"
                            )
                            print(
                                f"[TIMING] LLM first token delay: "
                                f"{(now-llm_start)*1000:.1f} ms"
                            )
                            first_token = False
                        yield before
                    break

                # Yield buffered text that definitely isn't the sentinel
                # Keep last 12 chars buffered in case sentinel is split
                # across chunks ("---END" + "---\n")
                safe = buffer[:-12]
                if safe:
                    if first_token and safe.strip():
                        now = time.perf_counter()
                        print(
                            f"[TIMING] TTFT from request start: "
                            f"{(now-request_start)*1000:.1f} ms"
                        )
                        print(
                            f"[TIMING] LLM first token delay: "
                            f"{(now-llm_start)*1000:.1f} ms"
                        )
                        first_token = False
                    yield safe
                    buffer = buffer[-12:]

        except Exception as e:
            print(f"[api] Streaming error: {e}")
            yield f"\n[Error: {e}]"

        finally:
            if os.path.exists(embed_path):
                os.unlink(embed_path)

#Routes

@app.get("/")
def root():
    return {
        "name":    "FastVLM API",
        "status":  "running",
        "model":   "MobileCLIP-L + Qwen2-0.5B",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "onnx_loaded":  ort_session is not None,
        "gguf_exists":  os.path.exists(GGUF_PATH),
        "binary_exists": os.path.exists(SERVER_BIN),
    }

@app.post("/predict")
async def predict(image:  UploadFile = File(...), prompt: str = Form(default="Describe this image in detail.")):
    
    try:
        t0 = time.perf_counter()

        # Load image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        t1 = time.perf_counter()
        print(f"[TIMING] image load: {(t1-t0)*1000:.1f} ms")

        # Encode with ONNX
        embeddings = encode_image(img)

        t2 = time.perf_counter()
        print(f"[TIMING] vision encoder: {(t2-t1)*1000:.1f} ms")    

        # Save embeddings to temp file
        embd_path  = save_embeddings(embeddings)

        t3 = time.perf_counter()
        print(f"[TIMING] save embeddings: {(t3-t2)*1000:.1f} ms")

        headers = {
            "X-Status": "ok",
            "X-Prompt": prompt.encode('utf-8').decode('latin-1'), 
            "X-Model": "custom-onnx-fastvlm-0.5b"
        }

        # try:
        #     # Run LLM
        #     response = run_llm_stream(embd_path, prompt)
        # finally:
        #     os.unlink(embd_path)
        
        
        
        return StreamingResponse(
            run_llm_stream(embd_path, prompt, t0),  
            media_type="text/plain",
            headers=headers
        )
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)