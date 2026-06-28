"""
FastVLM-style continuous camera demo using OpenCV.
 
Replicates the look of Apple's FastVLM demo app:
  - Live camera feed
  - TTFT badge (top of frame)
  - Status pill: "Processing Prompt" (yellow) / "Generating Response" (green)
  - Prompt / Response panel below the video
 
Controls:
  c  - toggle Continuous mode on/off
  s  - Single shot (run inference once on current frame)
  p  - cycle through preset prompts
  q  - quit
"""

import cv2
import numpy as np
import threading
import time
import io
import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://127.0.0.1:8000"
SENTINEL = "---END---"

HOLD_SECONDS = 3.0

def predict_stream(pil_image, prompt):
    """
    Sends the frame + prompt to your FastAPI /predict endpoint and
    yields response chunks as they stream in, stripping the
    ---END--- sentinel your server appends.
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=70)
    buf.seek(0)

    files = {"image": ("frame.jpg", buf, "image/jpeg")}
    data = {"prompt": prompt}

    with requests.post(f"{API_URL}/predict", files=files, data=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        leftover = ""
        for raw_chunk in r.iter_content(chunk_size=16):
            if not raw_chunk:
                continue
            leftover += raw_chunk.decode("utf-8", errors="ignore")

            if SENTINEL in leftover:
                before, _ = leftover.split(SENTINEL, 1)
                if before:
                    yield before
                return

            safe_len = len(leftover) - (len(SENTINEL) - 1)
            if safe_len > 0:
                yield leftover[:safe_len]
                leftover = leftover[safe_len:]


PRESET_PROMPTS = [
    "Describe this image in detail.",
    "How many fingers am I holding up? Respond using a single number. If no hand is present, respond with 0.",
    "What is written in this image? Output only the text in the image.",
]

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"          # idle | processing | generating
        self.ttft_ms = None
        self.response_text = ""
        self.prompt = PRESET_PROMPTS[0]
        self.continuous = False
        self.busy = False
        self.latest_frame = None
        self.running = True
        self.last_done_time = 0.0
 
state = State()

def inference_worker():
    while state.running:
        with state.lock:
            cooldown_elapsed = (time.perf_counter() - state.last_done_time) >= HOLD_SECONDS
            should_run = (
                state.continuous
                and not state.busy
                and state.latest_frame is not None
                and cooldown_elapsed
            )
            frame_copy = state.latest_frame.copy() if should_run else None
            prompt = state.prompt

        if not should_run:
            time.sleep(0.05)
            continue

        run_inference(frame_copy, prompt)

def run_inference(frame_bgr, prompt):
    """Run one full inference pass: encode -> stream tokens -> update state."""
    with state.lock:
        state.busy = True
        state.status = "processing"
        state.response_text = ""
        state.ttft_ms = None
 
    start = time.perf_counter()
 
    # Convert BGR (OpenCV) -> RGB PIL image for the model
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    first_token = True
    try:
        for chunk in predict_stream(pil_img, prompt):
            if first_token:
                ttft = (time.perf_counter() - start) * 1000
                with state.lock:
                    state.ttft_ms = ttft
                    state.status = "generating"
                first_token = False

            with state.lock:
                state.response_text += chunk
    except requests.exceptions.RequestException as e:
        with state.lock:
            state.response_text = f"[API error: {e}]"

    with state.lock:
        state.status = "idle"
        state.busy = False
        state.last_done_time = time.perf_counter()

def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()
 
 
FONT_BADGE = get_font(20)
FONT_LABEL = get_font(18)
FONT_TEXT = get_font(24)

def rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

def draw_overlay(frame_bgr, status, ttft_ms):
    """Draw TTFT badge (top) and status pill (bottom) onto the camera frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
 
    # --- TTFT badge (top center) ---
    if ttft_ms is not None:
        text = f"TTFT  {ttft_ms:.0f} ms"
    else:
        text = "TTFT  -- ms"
    tw, th = draw.textbbox((0, 0), text, font=FONT_BADGE)[2:]
    pad_x, pad_y = 14, 8
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    bx = (w - box_w) // 2
    by = 16
    rounded_rect(draw, (bx, by, bx + box_w, by + box_h), radius=12, fill=(0, 0, 0, 170))
    draw.text((bx + pad_x, by + pad_y - 2), text, font=FONT_BADGE, fill=(255, 255, 255, 255))

    # --- Status pill (bottom center) ---
    if status == "processing":
        label = "Processing Prompt"
        color = (255, 204, 0, 230)     # yellow
        dot_color = (255, 255, 255, 255)
    elif status == "generating":
        label = "Generating Response"
        color = (52, 199, 89, 230)     # green
        dot_color = (255, 255, 255, 255)
    else:
        label = None
        color = None
    
    if label:
        tw, th = draw.textbbox((0, 0), label, font=FONT_BADGE)[2:]
        dot_r = 6
        pad_x, pad_y, gap = 18, 10, 10
        box_w = dot_r * 2 + gap + tw + pad_x * 2
        box_h = th + pad_y * 2
        bx = (w - box_w) // 2
        by = h - box_h - 24
        rounded_rect(draw, (bx, by, bx + box_w, by + box_h), radius=box_h // 2, fill=color)
        cx, cy = bx + pad_x + dot_r, by + box_h // 2
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=dot_color)
        draw.text((bx + pad_x + dot_r * 2 + gap, by + pad_y - 2), label, font=FONT_BADGE, fill=(0, 0, 0, 255))
    
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def wrap_text(draw, text, font, max_width):
    """Simple word-wrap for the panel text."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def draw_panel(width, prompt, response, mode_label):
    """Draw the bottom panel with mode, prompt, and response (like the gif's cards)."""
    panel_h = 340
    panel = Image.new("RGB", (width, panel_h), (242, 242, 247))  # iOS-ish light gray
    draw = ImageDraw.Draw(panel)
    margin = 16
    max_w = width - margin * 2

    y = 10
    draw.text((margin, y), mode_label, font=FONT_LABEL, fill=(120, 120, 128))
    y += 28

    # PROMPT card
    draw.text((margin, y), "PROMPT", font=FONT_LABEL, fill=(120, 120, 128))
    y += 22
    card_h = 50
    rounded_rect(draw, (margin, y, width - margin, y + card_h), radius=10, fill=(255, 255, 255))
    lines = wrap_text(draw, prompt, FONT_TEXT, max_w - 16)[:2]
    ty = y + 8
    for line in lines:
        draw.text((margin + 10, ty), line, font=FONT_TEXT, fill=(20, 20, 20))
        ty += 26
    y += card_h + 10

    # RESPONSE card
    draw.text((margin, y), "RESPONSE", font=FONT_LABEL, fill=(120, 120, 128))
    y += 22
    card_h = panel_h - y - 6
    rounded_rect(draw, (margin, y, width - margin, y + card_h), radius=10, fill=(255, 255, 255))

    line_h = 26
    max_lines = (card_h - 16) // line_h
    all_lines = wrap_text(draw, response, FONT_TEXT, max_w - 16)
    lines = all_lines[:max_lines]
    if len(all_lines) > max_lines and lines:
        lines[-1] = lines[-1].rstrip() + " ..."

    ty = y + 8
    for line in lines:
        draw.text((margin + 10, ty), line, font=FONT_TEXT, fill=(20, 20, 20))
        ty += line_h

    return cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
 

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera (index 0). Try a different index or check permissions.")
        return
 
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()
 
    print("Controls: [c] toggle continuous  [s] single shot  [p] cycle prompt  [q] quit")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        with state.lock:
            state.latest_frame = frame
            status = state.status
            ttft = state.ttft_ms
            response = state.response_text
            prompt = state.prompt
            continuous = state.continuous
 
        display = draw_overlay(frame, status, ttft)
        mode_label = f"Mode: {'Continuous' if continuous else 'Single'}"
        panel = draw_panel(display.shape[1], prompt, response, mode_label)
 
        combined = np.vstack([display, panel])
        cv2.imshow("FastVLM Demo (OpenCV)", combined)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            with state.lock:
                state.continuous = not state.continuous
        elif key == ord('s'):
            if not state.busy:
                with state.lock:
                    frame_copy = state.latest_frame.copy()
                    p = state.prompt
                threading.Thread(target=run_inference, args=(frame_copy, p), daemon=True).start()
        elif key == ord('p'):
            with state.lock:
                idx = PRESET_PROMPTS.index(state.prompt)
                state.prompt = PRESET_PROMPTS[(idx + 1) % len(PRESET_PROMPTS)]
                state.response_text = ""
                state.ttft_ms = None
                state.last_done_time = 0.0
 
    state.running = False
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()