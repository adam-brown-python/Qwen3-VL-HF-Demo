import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, Iterable

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
import requests
import fitz

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.thistle = colors.Color(
    name="thistle",
    c50="#F9F5F9", c100="#F0E8F1", c200="#E7DBE8", c300="#DECEE0",
    c400="#D2BFD8", c500="#D8BFD8", c600="#B59CB7", c700="#927996",
    c800="#6F5675", c900="#4C3454", c950="#291233",
)

colors.red_gray = colors.Color(
    name="red_gray",
    c50="#f7eded", c100="#f5dcdc", c200="#efb4b4", c300="#e78f8f",
    c400="#d96a6a", c500="#c65353", c600="#b24444", c700="#8f3434",
    c800="#732d2d", c900="#5f2626", c950="#4d2020",
)

class ThistleTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.thistle,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="black",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_400, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_600)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            button_cancel_background_fill=f"linear-gradient(90deg, {colors.red_gray.c400}, {colors.red_gray.c500})",
            button_cancel_background_fill_dark=f"linear-gradient(90deg, {colors.red_gray.c700}, {colors.red_gray.c800})",
            button_cancel_background_fill_hover=f"linear-gradient(90deg, {colors.red_gray.c500}, {colors.red_gray.c600})",
            button_cancel_background_fill_hover_dark=f"linear-gradient(90deg, {colors.red_gray.c800}, {colors.red_gray.c900})",
            button_cancel_text_color="white",
            button_cancel_text_color_dark="white",
            button_cancel_text_color_hover="white",
            button_cancel_text_color_hover_dark="white",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

thistle_theme = ThistleTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
:root {
    --color-grey-50: #f9fafb;
    --banner-background: var(--secondary-400);
    --banner-text-color: var(--primary-100);
    --banner-background-dark: var(--secondary-800);
    --banner-text-color-dark: var(--primary-100);
    --banner-chrome-height: calc(16px + 43px);
    --chat-chrome-height-wide-no-banner: 320px;
    --chat-chrome-height-narrow-no-banner: 450px;
    --chat-chrome-height-wide: calc(var(--chat-chrome-height-wide-no-banner) + var(--banner-chrome-height));
    --chat-chrome-height-narrow: calc(var(--chat-chrome-height-narrow-no-banner) + var(--banner-chrome-height));
}
.banner-message { background-color: var(--banner-background); padding: 5px; margin: 0; border-radius: 5px; border: none; }
.banner-message-text { font-size: 13px; font-weight: bolder; color: var(--banner-text-color) !important; }
body.dark .banner-message { background-color: var(--banner-background-dark) !important; }
body.dark .gradio-container .contain .banner-message .banner-message-text { color: var(--banner-text-color-dark) !important; }
.toast-body { background-color: var(--color-grey-50); }
.html-container:has(.css-styles) { padding: 0; margin: 0; }
.css-styles { height: 0; }
.model-message { text-align: end; }
.model-dropdown-container { display: flex; align-items: center; gap: 10px; padding: 0; }
.user-input-container .multimodal-textbox{ border: none !important; }
.control-button { height: 51px; }
button.cancel { border: var(--button-border-width) solid var(--button-cancel-border-color); background: var(--button-cancel-background-fill); color: var(--button-cancel-text-color); box-shadow: var(--button-cancel-shadow); }
button.cancel:hover, .cancel[disabled] { background: var(--button-cancel-background-fill-hover); color: var(--button-cancel-text-color-hover); }
.opt-out-message { top: 8px; }
.opt-out-message .html-container, .opt-out-checkbox label { font-size: 14px !important; padding: 0 !important; margin: 0 !important; color: var(--neutral-400) !important; }
div.block.chatbot { height: calc(100svh - var(--chat-chrome-height-wide)) !important; max-height: 900px !important; }
div.no-padding { padding: 0 !important; }
@media (max-width: 1280px) { div.block.chatbot { height: calc(100svh - var(--chat-chrome-height-wide)) !important; } }
@media (max-width: 1024px) {
    .responsive-row { flex-direction: column; }
    .model-message { text-align: start; font-size: 10px !important; }
    .model-dropdown-container { flex-direction: column; align-items: flex-start; }
    div.block.chatbot { height: calc(100svh - var(--chat-chrome-height-narrow)) !important; }
}
@media (max-width: 400px) {
    .responsive-row { flex-direction: column; }
    .model-message { text-align: start; font-size: 10px !important; }
    .model-dropdown-container { flex-direction: column; align-items: flex-start; }
    div.block.chatbot { max-height: 360px !important; }
}
@media (max-height: 932px) { .chatbot { max-height: 500px !important; } }
@media (max-height: 1280px) { div.block.chatbot { max-height: 800px !important; } }
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", device)

MODEL_ID_Q3VL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
processor_q3vl = AutoProcessor.from_pretrained(MODEL_ID_Q3VL, trust_remote_code=True, use_fast=False)
model_q3vl = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_ID_Q3VL,
    trust_remote_code=True,
    dtype=torch.float16
).to(device).eval()

def extract_gif_frames(gif_path: str):
    if not gif_path:
        return []
    with Image.open(gif_path) as gif:
        total_frames = gif.n_frames
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
        frames = []
        for i in frame_indices:
            gif.seek(i)
            frames.append(gif.convert("RGB").copy())
    return frames

def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frames.append(pil_image)
    vidcap.release()
    return frames

def convert_pdf_to_images(file_path: str, dpi: int = 200):
    if not file_path:
        return []
    images = []
    pdf_document = fitz.open(file_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(BytesIO(img_data)))
    pdf_document.close()
    return images

def get_initial_pdf_state() -> Dict[str, Any]:
    return {"pages": [], "total_pages": 0, "current_page_index": 0}

def load_and_preview_pdf(file_path: Optional[str]) -> Tuple[Optional[Image.Image], Dict[str, Any], str]:
    state = get_initial_pdf_state()
    if not file_path:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    try:
        pages = convert_pdf_to_images(file_path)
        if not pages:
            return None, state, '<div style="text-align:center;">Could not load file</div>'
        state["pages"] = pages
        state["total_pages"] = len(pages)
        page_info_html = f'<div style="text-align:center;">Page 1 / {state["total_pages"]}</div>'
        return pages[0], state, page_info_html
    except Exception as e:
        return None, state, f'<div style="text-align:center;">Failed to load preview: {e}</div>'

def navigate_pdf_page(direction: str, state: Dict[str, Any]):
    if not state or not state["pages"]:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    current_index = state["current_page_index"]
    total_pages = state["total_pages"]
    if direction == "prev":
        new_index = max(0, current_index - 1)
    elif direction == "next":
        new_index = min(total_pages - 1, current_index + 1)
    else:
        new_index = current_index
    state["current_page_index"] = new_index
    image_preview = state["pages"][new_index]
    page_info_html = f'<div style="text-align:center;">Page {new_index + 1} / {total_pages}</div>'
    return image_preview, state, page_info_html

@spaces.GPU
def generate_image(text: str, image: Image.Image, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt_full = processor_q3vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_q3vl(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor_q3vl, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model_q3vl.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(text: str, video_path: str, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return
    frames = downsample_video(video_path)
    if not frames:
        yield "Could not process video.", "Could not process video."
        return
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    for frame in frames:
        messages[0]["content"].insert(0, {"type": "image"})
    prompt_full = processor_q3vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_q3vl(text=[prompt_full], images=frames, return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor_q3vl, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}
    thread = Thread(target=model_q3vl.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_pdf(text: str, state: Dict[str, Any], max_new_tokens: int = 2048, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if not state or not state["pages"]:
        yield "Please upload a PDF file first.", "Please upload a PDF file first."
        return
    page_images = state["pages"]
    full_response = ""
    for i, image in enumerate(page_images):
        page_header = f"--- Page {i+1}/{len(page_images)} ---\n"
        yield full_response + page_header, full_response + page_header
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt_full = processor_q3vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor_q3vl(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
        streamer = TextIteratorStreamer(processor_q3vl, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_q3vl.generate, kwargs=generation_kwargs)
        thread.start()
        page_buffer = ""
        for new_text in streamer:
            page_buffer += new_text
            yield full_response + page_header + page_buffer, full_response + page_header + page_buffer
            time.sleep(0.01)
        full_response += page_header + page_buffer + "\n\n"

@spaces.GPU
def generate_caption(image: Image.Image, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if image is None:
        yield "Please upload an image to caption.", "Please upload an image to caption."
        return
    system_prompt = (
        "You are an AI assistant that rigorously follows this response protocol: For every input image, your primary "
        "task is to write a precise caption that captures the essence of the image in clear, concise, and contextually "
        "accurate language. Along with the caption, provide a structured set of attributes describing the visual "
        "elements, including details such as objects, people, actions, colors, environment, mood, and other notable "
        "characteristics. Ensure captions are precise, neutral, and descriptive, avoiding unnecessary elaboration or "
        "subjective interpretation unless explicitly required. Do not reference the rules or instructions in the output; "
        "only return the formatted caption, attributes, and class_name."
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": system_prompt}]}]
    prompt_full = processor_q3vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_q3vl(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor_q3vl, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model_q3vl.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_gif(text: str, gif_path: str, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if gif_path is None:
        yield "Please upload a GIF.", "Please upload a GIF."
        return
    frames = extract_gif_frames(gif_path)
    if not frames:
        yield "Could not process GIF.", "Could not process GIF."
        return
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    for frame in frames:
        messages[0]["content"].insert(0, {"type": "image"})
    prompt_full = processor_q3vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_q3vl(text=[prompt_full], images=frames, return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor_q3vl, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}
    thread = Thread(target=model_q3vl.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

image_examples = [["Perform OCR on the image precisely...", "examples/images/1.jpg"],
                  ["Caption the image. Describe the safety measures shown in the image. Conclude whether the situation is (safe or unsafe)...", "examples/images/2.jpg"],
                  ["Solve the problem...", "examples/images/3.png"]]
video_examples = [["Explain the Ad video in detail.", "examples/videos/1.mp4"],
                  ["Explain the video in detail.", "examples/videos/2.mp4"]]
pdf_examples = [["Extract the content precisely.", "examples/pdfs/doc1.pdf"],
                ["Analyze and provide a short report.", "examples/pdfs/doc2.pdf"]]
gif_examples = [["Describe this GIF.", "examples/gifs/1.gif"],
                ["Describe this GIF.", "examples/gifs/2.gif"]]
#caption_examples = [["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/candy.JPG"],
#                    ["examples/captions/2.png"], ["examples/captions/3.png"]]
caption_examples = [["examples/captions/1.JPG"],
                    ["examples/captions/2.jpeg"], ["examples/captions/3.jpeg"]]

with gr.Blocks(theme=thistle_theme, css=css) as demo:
    pdf_state = gr.State(value=get_initial_pdf_state())
    gr.Markdown("# **Qwen-3VL:Multimodal**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image", height=290)
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])

                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])

                with gr.TabItem("PDF Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_query = gr.Textbox(label="Query Input", placeholder="e.g., 'Summarize this document'")
                            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                            pdf_submit = gr.Button("Submit", variant="primary")
                        with gr.Column(scale=1):
                            pdf_preview_img = gr.Image(label="PDF Preview", height=290)
                            with gr.Row():
                                prev_page_btn = gr.Button("◀ Previous")
                                page_info = gr.HTML('<div style="text-align:center;">No file loaded</div>')
                                next_page_btn = gr.Button("Next ▶")
                    gr.Examples(examples=pdf_examples, inputs=[pdf_query, pdf_upload])

                with gr.TabItem("Gif Inference"):
                    gif_query = gr.Textbox(label="Query Input", placeholder="e.g., 'What is happening in this gif?'")
                    gif_upload = gr.Image(type="filepath", label="Upload GIF", height=290)
                    gif_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=gif_examples, inputs=[gif_query, gif_upload])

                with gr.TabItem("Caption"):
                    caption_image_upload = gr.Image(type="pil", label="Image to Caption", height=290)
                    caption_submit = gr.Button("Generate Caption", variant="primary")
                    gr.Examples(examples=caption_examples, inputs=[caption_image_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=14, show_copy_button=True)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(label="(Result.Md)", latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ])

    image_submit.click(fn=generate_image,
                       inputs=[image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                       outputs=[output, markdown_output])
    video_submit.click(fn=generate_video,
                       inputs=[video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                       outputs=[output, markdown_output])
    pdf_submit.click(fn=generate_pdf,
                     inputs=[pdf_query, pdf_state, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                     outputs=[output, markdown_output])
    gif_submit.click(fn=generate_gif,
                     inputs=[gif_query, gif_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                     outputs=[output, markdown_output])
    caption_submit.click(fn=generate_caption,
                         inputs=[caption_image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                         outputs=[output, markdown_output])

    pdf_upload.change(fn=load_and_preview_pdf, inputs=[pdf_upload], outputs=[pdf_preview_img, pdf_state, page_info])
    prev_page_btn.click(fn=lambda s: navigate_pdf_page("prev", s), inputs=[pdf_state], outputs=[pdf_preview_img, pdf_state, page_info])
    next_page_btn.click(fn=lambda s: navigate_pdf_page("next", s), inputs=[pdf_state], outputs=[pdf_preview_img, pdf_state, page_info])

if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)