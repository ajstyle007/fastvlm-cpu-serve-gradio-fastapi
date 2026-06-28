import gradio as gr

with gr.Blocks() as demo:
    cam = gr.Image(sources="webcam", type="pil")
    btn = gr.Button("Test")
    out = gr.Textbox()

    btn.click(lambda x: str(type(x)), cam, out)

demo.launch()