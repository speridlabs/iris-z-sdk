import json
import os
import random
from pathlib import Path

import gradio as gr
import model

NAME = "Iris-Z"
TAGLINE = (
    "<strong>One generative model for images and their depth.</strong> Estimate the depth of any photo, generate a "
    "photo, or a photo together with its depth, from a prompt, or paint a new photo onto a depth map. Choose the "
    "model at the top; pick an example below, or <em>bring your own.</em>"
)
LINKS = {
    "Code": "https://github.com/speridlabs/iris-z-sdk",
    "Weights": "https://huggingface.co/speridlabs/iris-z",
}

ROOT = Path(__file__).parent
PRESETS_PATH = ROOT / "examples" / "presets.json"
PRESETS = json.loads(PRESETS_PATH.read_text()) if PRESETS_PATH.exists() else []
DEPTH_PRESETS = [p for p in PRESETS if p["task"] == "depth"]
PROMPT_PRESETS = [p for p in PRESETS if p["task"] == "rgbd"]
DEPTH_MAP_PRESETS = [p for p in PRESETS if p["task"] == "depth_to_image"]


def meta(kind, text):
    return f'<div class="sl-meta"><span>{kind}</span><span>{text}</span></div>'


def fresh_seed(seed, randomize):
    return random.randint(0, 2**31 - 1) if randomize else int(seed)


def run_depth(model_name, image, scale_label, seed, steps):
    if image is None:
        raise gr.Error("Upload a photo first.")
    coloured, seen, seconds = model.depth(model_name, image, model.SCALES[scale_label], int(seed), int(steps))
    return coloured, meta("Depth", f"{seen.width}×{seen.height} · {int(steps)} steps · {seconds:.1f} s")


def run_rgb(model_name, prompt, size, steps, guidance, seed, randomize):
    seed = fresh_seed(seed, randomize)
    height, width = model.SIZES[size]
    image, seconds = model.rgb(model_name, prompt, height, width, int(steps), guidance, seed)
    return image, seed, meta("Generated", f"{size} · seed {seed} · {int(steps)} steps · {seconds:.0f} s")


def run_rgbd(model_name, prompt, size, steps, guidance, seed, randomize):
    seed = fresh_seed(seed, randomize)
    height, width = model.SIZES[size]
    image, coloured, seconds = model.rgbd(model_name, prompt, height, width, int(steps), guidance, seed)
    return image, coloured, seed, meta("Generated", f"{size} · seed {seed} · {int(steps)} steps · {seconds:.0f} s")


def run_depth_to_image(model_name, path, depth_format, prompt, scale_label, steps, guidance, seed, randomize):
    if path is None:
        raise gr.Error("Upload a depth map first.")
    seed = fresh_seed(seed, randomize)
    image, seconds = model.depth_to_image(
        model_name, path, depth_format, prompt, model.SCALES[scale_label], int(steps), guidance, seed
    )
    return image, seed, meta("Painted", f"seed {seed} · {int(steps)} steps · {seconds:.0f} s")


def run_reimagine(model_name, image, prompt, scale_label, steps, guidance, seed, randomize):
    if image is None:
        raise gr.Error("Upload a photo first.")
    seed = fresh_seed(seed, randomize)
    new_image, _depth, seconds = model.reimagine(
        model_name, image, prompt, model.SCALES[scale_label], int(steps), guidance, seed
    )
    return new_image, seed, meta("Painted", f"from the photo's depth · seed {seed} · {seconds:.0f} s")


def switch_model(model_name):
    model.use(model_name)
    guidance = model.MODELS[model_name]["guidance"]
    return [gr.update(value=guidance)] * 3


FONT = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
MONO = [gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]
theme = gr.themes.Base(font=FONT, font_mono=MONO, radius_size=gr.themes.sizes.radius_none).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#000000",
    body_text_color="#000000",
    body_text_color_dark="#ffffff",
    block_background_fill="#ffffff",
    block_background_fill_dark="#000000",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#000000",
    background_fill_secondary="#f7f7f7",
    background_fill_secondary_dark="#111111",
    border_color_primary="#e3e3e3",
    border_color_primary_dark="rgba(255,255,255,0.15)",
    block_border_color="#e3e3e3",
    block_border_color_dark="rgba(255,255,255,0.15)",
    block_shadow="none",
    input_background_fill="#ffffff",
    input_background_fill_dark="#000000",
    input_border_color="#d4d4d4",
    input_border_color_dark="rgba(255,255,255,0.25)",
    input_border_color_focus="#000000",
    input_border_color_focus_dark="#ffffff",
    button_primary_background_fill="#000000",
    button_primary_background_fill_dark="#ffffff",
    button_primary_background_fill_hover="#d94a1f",
    button_primary_background_fill_hover_dark="#d94a1f",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#000000",
    color_accent="#d94a1f",
    color_accent_soft="#fbe7e0",
    slider_color="#d94a1f",
    slider_color_dark="#d94a1f",
    checkbox_background_color_selected="#000000",
    checkbox_background_color_selected_dark="#ffffff",
)
CSS = """
:root { --sl-fg: #000; --sl-fg-2: #444; --sl-fg-3: #6b6b6b; --sl-rule: #e3e3e3; }
.dark { --sl-fg: #fff; --sl-fg-2: rgba(255,255,255,.75); --sl-fg-3: rgba(255,255,255,.55);
  --sl-rule: rgba(255,255,255,.15); }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 32px !important; }
footer { display: none !important; }
.sl-head { padding: 40px 0 28px; border-bottom: 1px solid var(--sl-rule); margin-bottom: 8px; }
.sl-label, .sl-meta, .sl-links a, .sl-links button, .sl-section, .block-label, .label-wrap span, label > span {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important; text-transform: uppercase;
  letter-spacing: .12em; font-size: 10.5px !important; color: var(--sl-fg-3) !important; }
.sl-head h1 { margin: 14px 0 0; font-size: clamp(56px, 6.4vw, 104px); font-weight: 700; line-height: .86;
  letter-spacing: -0.055em; color: var(--sl-fg); }
.sl-row { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px 40px; }
.sl-head p { margin: 18px 0 0; max-width: 60ch; font-size: 17px; line-height: 1.5; color: var(--sl-fg-2); }
.sl-head p strong { color: var(--sl-fg); font-weight: 600; }
.sl-head p em { color: #d94a1f; font-style: normal; font-weight: 600; }
.sl-links { display: flex; gap: 10px; }
.sl-links a, .sl-links button { display: inline-flex; align-items: center; height: 40px; padding: 0 20px;
  border: 1px solid var(--sl-fg); background: transparent; color: var(--sl-fg) !important; text-decoration: none;
  cursor: pointer; transition: background .15s, color .15s; }
.sl-links a:hover, .sl-links button:hover { background: var(--sl-fg); color: var(--body-background-fill) !important; }
#sl-theme { width: 40px; padding: 0; justify-content: center; font-size: 15px !important; letter-spacing: 0; }
#sl-theme::before { content: "☾"; }
.dark #sl-theme::before { content: "☀"; }
.sl-meta { display: flex; justify-content: space-between; gap: 16px; padding-top: 6px; }
.sl-meta span:first-child { color: #d94a1f; }
.sl-section { border-top: 1px solid var(--sl-rule); padding-top: 28px !important; margin-top: 28px; }
#generate { height: 48px; font-size: 14px; font-weight: 500; letter-spacing: .01em; }
.sl-out { background: #0c0c0c; }
#presets .thumbnail-item { border: 0 !important; box-shadow: none !important; overflow: hidden; }
#presets .caption-label { display: none; }
#presets .grid-wrap { max-height: none !important; height: auto !important; overflow: visible !important; }
@media (max-width: 760px) {
  .gradio-container { padding: 0 16px !important; }
  .sl-head { padding-top: 24px; }
  #presets .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
"""
THEME_JS = """() => {
  const set = (dark) => { document.body.classList.toggle('dark', dark); localStorage.setItem('sl-theme', dark ? 'dark' : 'light'); };
  set(localStorage.getItem('sl-theme') !== 'light');
  document.addEventListener('click', (e) => { if (e.target.closest('#sl-theme')) set(!document.body.classList.contains('dark')); });
}"""
LINK_HTML = "".join(f'<a href="{url}" target="_blank">{text}</a>' for text, url in LINKS.items())
HEAD = f"""
<div class="sl-head">
  <span class="sl-label">Speridlabs · Research</span>
  <div class="sl-row">
    <h1>{NAME}</h1>
    <div class="sl-links">{LINK_HTML}<button id="sl-theme" type="button" aria-label="Toggle light/dark"></button></div>
  </div>
  <p>{TAGLINE}</p>
</div>
"""


def gallery(presets, key="image"):
    return gr.Gallery(
        [str(ROOT / p[key]) for p in presets],
        columns=6,
        rows=max(1, (len(presets) + 5) // 6),
        height="auto",
        object_fit="cover",
        allow_preview=False,
        show_label=False,
        show_share_button=False,
        show_download_button=False,
        elem_id="presets",
    )


def generation_settings(guidance):
    with gr.Accordion("Settings", open=False):
        size = gr.Dropdown(list(model.SIZES), value=model.DEFAULT_SIZE, label="Size (width × height)")
        steps = gr.Slider(10, 100, value=model.DEFAULT_STEPS, step=1, label="Steps")
        guidance_scale = gr.Slider(1.0, 10.0, value=guidance, step=0.1, label="Guidance scale")
        seed = gr.Number(value=0, precision=0, label="Seed")
        randomize = gr.Checkbox(value=False, label="Random seed")
    return size, steps, guidance_scale, seed, randomize


def output_image():
    return gr.Image(type="pil", format="png", show_label=False, elem_classes="sl-out", height="min(560px, 92vw)")


START_GUIDANCE = model.MODELS[model.DEFAULT_MODEL]["guidance"]
with gr.Blocks(title=f"{NAME} · Speridlabs", theme=theme, css=CSS, js=THEME_JS) as demo:
    gr.HTML(HEAD)
    model_choice = gr.Radio(list(model.MODELS), value=model.DEFAULT_MODEL, label="Model")
    with gr.Tabs():
        with gr.Tab("Image → depth"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    depth_input = gr.Image(type="pil", label="Photo", height="min(480px, 92vw)")
                    depth_button = gr.Button("Estimate depth", variant="primary", elem_id="generate")
                    with gr.Accordion("Settings", open=False):
                        depth_scale = gr.Dropdown(list(model.SCALES), value=model.DEFAULT_SCALE, label="Resolution")
                        depth_steps = gr.Slider(1, 50, value=model.DEPTH_STEPS, step=1, label="Steps")
                        depth_seed = gr.Number(value=0, precision=0, label="Seed")
                with gr.Column(scale=7):
                    depth_output = output_image()
                    depth_info = gr.HTML(meta("Ready", "near = red, far = blue"))
            if DEPTH_PRESETS:
                gr.HTML('<div class="sl-section">Example photos</div>')
                depth_gallery = gallery(DEPTH_PRESETS)
        with gr.Tab("Prompt → image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    rgb_prompt = gr.Textbox(
                        label="Prompt", lines=4, value=PROMPT_PRESETS[0]["prompt"] if PROMPT_PRESETS else ""
                    )
                    rgb_button = gr.Button("Generate", variant="primary", elem_id="generate")
                    rgb_size, rgb_steps, rgb_guidance, rgb_seed, rgb_random = generation_settings(START_GUIDANCE)
                with gr.Column(scale=7):
                    rgb_output = output_image()
                    rgb_info = gr.HTML(meta("Ready", "the image alone"))
            if PROMPT_PRESETS:
                gr.HTML('<div class="sl-section">Example prompts</div>')
                rgb_gallery = gallery(PROMPT_PRESETS)
        with gr.Tab("Prompt → image + depth"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    rgbd_prompt = gr.Textbox(
                        label="Prompt", lines=4, value=PROMPT_PRESETS[0]["prompt"] if PROMPT_PRESETS else ""
                    )
                    rgbd_button = gr.Button("Generate", variant="primary", elem_id="generate")
                    rgbd_size, rgbd_steps, rgbd_guidance, rgbd_seed, rgbd_random = generation_settings(START_GUIDANCE)
                with gr.Column(scale=7):
                    with gr.Row():
                        rgbd_image = gr.Image(type="pil", format="png", show_label=False, elem_classes="sl-out")
                        rgbd_depth = gr.Image(type="pil", format="png", show_label=False, elem_classes="sl-out")
                    rgbd_info = gr.HTML(meta("Ready", "near = red, far = blue"))
            if PROMPT_PRESETS:
                gr.HTML('<div class="sl-section">Example prompts</div>')
                rgbd_gallery = gallery(PROMPT_PRESETS)
        with gr.Tab("Depth → image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Tabs():
                        with gr.Tab("From a depth map"):
                            map_input = gr.Image(
                                type="filepath",
                                format="png",
                                image_mode=None,
                                label="Depth map",
                                height="min(380px, 92vw)",
                            )
                            map_format = gr.Dropdown(
                                list(model.DEPTH_FORMATS), value=list(model.DEPTH_FORMATS)[0], label="Format"
                            )
                            map_button = gr.Button("Paint", variant="primary", elem_id="generate")
                        with gr.Tab("From a photo's depth"):
                            photo_input = gr.Image(type="pil", label="Photo", height="min(380px, 92vw)")
                            photo_button = gr.Button("Estimate depth and paint", variant="primary", elem_id="generate")
                    paint_prompt = gr.Textbox(label="Prompt for the new image (optional)", lines=3)
                    with gr.Accordion("Settings", open=False):
                        paint_scale = gr.Dropdown(list(model.SCALES), value=model.DEFAULT_SCALE, label="Resolution")
                        paint_steps = gr.Slider(10, 100, value=model.DEFAULT_STEPS, step=1, label="Steps")
                        paint_guidance = gr.Slider(1.0, 10.0, value=START_GUIDANCE, step=0.1, label="Guidance scale")
                        paint_seed = gr.Number(value=0, precision=0, label="Seed")
                        paint_random = gr.Checkbox(value=False, label="Random seed")
                with gr.Column(scale=7):
                    paint_output = output_image()
                    paint_info = gr.HTML(meta("Ready", "a new image that follows the depth"))
            if DEPTH_MAP_PRESETS:
                gr.HTML('<div class="sl-section">Example depth maps</div>')
                map_gallery = gallery(DEPTH_MAP_PRESETS)

    model_choice.change(switch_model, model_choice, [rgb_guidance, rgbd_guidance, paint_guidance], concurrency_id="gpu")
    depth_button.click(
        run_depth,
        [model_choice, depth_input, depth_scale, depth_seed, depth_steps],
        [depth_output, depth_info],
        concurrency_id="gpu",
    )
    rgb_inputs = [model_choice, rgb_prompt, rgb_size, rgb_steps, rgb_guidance, rgb_seed, rgb_random]
    rgb_button.click(run_rgb, rgb_inputs, [rgb_output, rgb_seed, rgb_info], concurrency_id="gpu")
    rgbd_inputs = [model_choice, rgbd_prompt, rgbd_size, rgbd_steps, rgbd_guidance, rgbd_seed, rgbd_random]
    rgbd_button.click(run_rgbd, rgbd_inputs, [rgbd_image, rgbd_depth, rgbd_seed, rgbd_info], concurrency_id="gpu")
    paint_settings = [paint_prompt, paint_scale, paint_steps, paint_guidance, paint_seed, paint_random]
    map_button.click(
        run_depth_to_image,
        [model_choice, map_input, map_format] + paint_settings,
        [paint_output, paint_seed, paint_info],
        concurrency_id="gpu",
    )
    photo_button.click(
        run_reimagine,
        [model_choice, photo_input] + paint_settings,
        [paint_output, paint_seed, paint_info],
        concurrency_id="gpu",
    )

    if DEPTH_PRESETS:

        def pick_photo(evt: gr.SelectData):
            return str(ROOT / DEPTH_PRESETS[evt.index]["image"])

        depth_gallery.select(pick_photo, None, depth_input, show_progress="hidden")
    if PROMPT_PRESETS:

        def pick_prompt(evt: gr.SelectData):
            preset = PROMPT_PRESETS[evt.index]
            return preset["prompt"], preset["seed"]

        rgb_gallery.select(pick_prompt, None, [rgb_prompt, rgb_seed], show_progress="hidden")
        rgbd_gallery.select(pick_prompt, None, [rgbd_prompt, rgbd_seed], show_progress="hidden")
    if DEPTH_MAP_PRESETS:

        def pick_depth_map(evt: gr.SelectData):
            preset = DEPTH_MAP_PRESETS[evt.index]
            return str(ROOT / preset["image"]), preset["format"], preset["prompt"]

        map_gallery.select(pick_depth_map, None, [map_input, map_format, paint_prompt], show_progress="hidden")

if __name__ == "__main__":
    model.load_all()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860"))
    )
