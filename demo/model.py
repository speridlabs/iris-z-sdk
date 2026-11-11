import gc
import os
import threading
import time

import torch
from PIL import Image

from iris_z import IrisZPipeline

WEIGHTS = os.environ.get("IRIS_Z_WEIGHTS", "speridlabs/iris-z")
MODELS = {
    "Iris-Z-big · 32B": {"variant": "big", "guidance": 3.5},
    "Iris-Z-small · 9B": {"variant": "small", "guidance": 4.0},
}
DEFAULT_MODEL = list(MODELS)[0]
SCALES = {"512 (fast)": 512, "1024 (sharper, slower)": 1024}
DEFAULT_SCALE = "512 (fast)"
SIZES = {
    f"{width}×{height}": (height, width)
    for width, height in [
        (512, 512),
        (576, 448),
        (448, 576),
        (640, 384),
        (384, 640),
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1344, 768),
        (768, 1344),
    ]
}
DEFAULT_SIZE = "512×512"
DEFAULT_STEPS = 50
DEPTH_STEPS = 4
DEPTH_FORMATS = {
    "Our colours (near = red)": "colours",
    "Depth (bright = far)": "depth",
    "Disparity (bright = near)": "disparity",
}

TWO_GPUS = torch.cuda.device_count() >= 2
PLACEMENT = {
    "big": {"device": "cuda:0", "text": "cuda:1"} if TWO_GPUS else {"device": "cuda:0", "text": "cuda:0"},
    "small": {"device": "cuda:1", "text": "cuda:1"} if TWO_GPUS else {"device": "cuda:0", "text": "cuda:0"},
}
PIPELINES = {}
ACTIVE = {"name": None}
GPU_LOCK = threading.RLock()


def load_all():
    """With one GPU, the model not in use waits in CPU memory."""
    for name, spec in MODELS.items():
        placement = PLACEMENT[spec["variant"]]
        PIPELINES[name] = IrisZPipeline.from_pretrained(
            WEIGHTS, variant=spec["variant"], device=placement["device"], text_encoder_device=placement["text"]
        )
        if not TWO_GPUS:
            move(PIPELINES[name], "cpu")
    use(DEFAULT_MODEL)


def move(pipeline, device):
    pipeline.to(device)
    if pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device)
        pipeline.text_device = torch.device(device)
    gc.collect()
    torch.cuda.empty_cache()


def use(name):
    with GPU_LOCK:
        if not TWO_GPUS and ACTIVE["name"] != name:
            if ACTIVE["name"] is not None:
                move(PIPELINES[ACTIVE["name"]], "cpu")
            move(PIPELINES[name], "cuda:0")
        ACTIVE["name"] = name
        return PIPELINES[name]


def depth(name, image, scale, seed, steps):
    with GPU_LOCK:
        pipeline = use(name)
        started = time.time()
        output = pipeline.image_to_depth(image, resolution=scale, steps=steps, seeds=(seed,))
        return output.colourized(), output.image, time.time() - started


def rgb(name, prompt, height, width, steps, guidance, seed):
    with GPU_LOCK:
        pipeline = use(name)
        started = time.time()
        image = pipeline.text_to_image(prompt, width=width, height=height, steps=steps, guidance=guidance, seed=seed)
        return image, time.time() - started


def rgbd(name, prompt, height, width, steps, guidance, seed):
    with GPU_LOCK:
        pipeline = use(name)
        started = time.time()
        output = pipeline.text_to_rgbd(prompt, width=width, height=height, steps=steps, guidance=guidance, seed=seed)
        return output.image, output.depth.colourized(), time.time() - started


def depth_to_image(name, path, depth_format, prompt, scale, steps, guidance, seed):
    with GPU_LOCK:
        pipeline = use(name)
        started = time.time()
        depth_map = Image.open(path)
        image = pipeline.depth_to_image(
            depth_map,
            prompt,
            resolution=scale,
            steps=steps,
            guidance=guidance,
            seed=seed,
            depth_format=DEPTH_FORMATS[depth_format],
        )
        return image, time.time() - started


def reimagine(name, image, prompt, scale, steps, guidance, seed):
    """Photo -> its depth -> a new photo with that geometry."""
    with GPU_LOCK:
        pipeline = use(name)
        started = time.time()
        estimated = pipeline.image_to_depth(image, resolution=scale, seeds=(seed,))
        new_image = pipeline.depth_to_image(
            estimated, prompt, resolution=scale, steps=steps, guidance=guidance, seed=seed
        )
        return new_image, estimated.colourized(), time.time() - started
