import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors.torch import load_file

from . import depth as depth_tools
from . import rgbd
from .model.autoencoder import AutoEncoder, AutoEncoderParams
from .model.model import Flux2, Flux2Params, Klein9BParams
from .model.sampling import batched_prc_txt
from .model.text_encoder import load_mistral_small_embedder, load_qwen3_embedder

DEFAULT_REPO = "speridlabs/iris-z"
VARIANT_FOLDERS = {"big": "Iris-Z-big", "small": "Iris-Z-small"}
VARIANTS = tuple(VARIANT_FOLDERS)
AUTOENCODER_REPO = "black-forest-labs/FLUX.2-dev"
BASE_MODELS = {
    "flux.2-dev": {"params": Flux2Params, "text_encoder": load_mistral_small_embedder},
    "flux.2-klein-base-9b": {
        "params": Klein9BParams,
        "text_encoder": lambda device: load_qwen3_embedder(variant="8B", device=device),
    },
}
SMALL_IMAGE_PIXELS = 500_000


@dataclass
class DepthOutput:
    log_depth: np.ndarray  # (H, W) relative log depth, at the input photo's size
    sky_mask: np.ndarray  # (H, W) bool, at the input photo's size
    log_depth_internal: np.ndarray  # log depth at the model's working resolution
    image: Image.Image  # the image the model saw

    def colourized(self):
        return Image.fromarray(depth_tools.colourize(self.log_depth, self.sky_mask))

    def sky_mask_internal(self):
        height, width = self.log_depth_internal.shape
        return resize_mask(self.sky_mask, width, height)


def resize_log_depth(log_depth, width, height):
    return np.asarray(Image.fromarray(log_depth.astype(np.float32)).resize((width, height), Image.BILINEAR))


def resize_mask(mask, width, height):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 127


@dataclass
class EncodedPrompt:
    embedding: torch.Tensor | None  # [empty prompt, prompt]; None for the empty prompt


@dataclass
class RGBDOutput:
    image: Image.Image
    depth: DepthOutput


class IrisZPipeline:
    def __init__(self, model, autoencoder, text_encoder, config, empty_context, device, text_device):
        self.model = model
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.config = config
        self.empty_context = empty_context
        self.device = torch.device(device)
        self.text_device = torch.device(text_device)
        self.sky_ceiling = float(config.get("sky_ceiling", 1.0))

    @classmethod
    def from_pretrained(
        cls, repo=DEFAULT_REPO, variant="big", device="cuda", text_encoder_device=None, load_text_encoder=True
    ):
        """repo: a Hugging Face repo or a local folder. load_text_encoder=False loads only what image -> depth needs."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        folder_name = VARIANT_FOLDERS[variant]
        folder = Path(repo) / folder_name
        if not folder.is_dir():
            folder = Path(snapshot_download(repo, allow_patterns=[f"{folder_name}/*"])) / folder_name
        config = json.loads((folder / "config.json").read_text())
        base = BASE_MODELS[config["base_model"]]
        text_device = text_encoder_device or device

        params = base["params"]()
        if config.get("guidance_mode") == "remove":
            params.use_guidance_embed = False
        with torch.device("meta"):
            model = Flux2(params, num_modalities=config["num_modalities"]).to(torch.bfloat16)
        state = {}
        index = json.loads((folder / "transformer.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(folder / shard), device=str(device)))
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device).eval().requires_grad_(False)

        autoencoder_path = hf_hub_download(AUTOENCODER_REPO, "ae.safetensors")
        with torch.device("meta"):
            autoencoder = AutoEncoder(AutoEncoderParams())
        autoencoder.load_state_dict(load_file(autoencoder_path, device=str(device)), strict=True, assign=True)
        autoencoder = autoencoder.eval().requires_grad_(False)

        empty = load_file(str(folder / "empty_prompt.safetensors"), device=str(device))["empty_prompt"]
        empty_context = batched_prc_txt(empty)
        text_encoder = None
        if load_text_encoder:
            text_encoder = base["text_encoder"](device=text_device).eval().requires_grad_(False)
        return cls(model, autoencoder, text_encoder, config, empty_context, device, text_device)

    def to(self, device):
        self.model.to(device)
        self.autoencoder.to(device)
        self.empty_context = tuple(part.to(device) for part in self.empty_context)
        self.device = torch.device(device)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt):
        """Every task accepts the result in place of the prompt string."""
        if isinstance(prompt, EncodedPrompt):
            return prompt
        if not prompt.strip():
            return EncodedPrompt(None)
        if self.text_encoder is None:
            raise RuntimeError("this pipeline was loaded without its text encoder (load_text_encoder=False)")
        return EncodedPrompt(self.text_encoder(["", prompt]).to(torch.bfloat16))

    def prompt_context(self, prompt):
        encoded = self.encode_prompt(prompt)
        if encoded.embedding is None:
            return self.empty_context
        return batched_prc_txt(encoded.embedding.to(self.device))

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    @torch.no_grad()
    def encode_image(self, rgb_uint8):
        rgb = torch.from_numpy(rgb_uint8.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        dtype = next(self.autoencoder.parameters()).dtype
        return self.autoencoder.encode(rgb.to(self.device, dtype))

    @torch.no_grad()
    def decode_image(self, latent):
        dtype = next(self.autoencoder.parameters()).dtype
        rgb = self.autoencoder.decode(latent.to(dtype)).float().clamp(-1, 1)
        return Image.fromarray((127.5 * (rgb[0] + 1.0)).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())

    @torch.no_grad()
    def decode_log_depth(self, latent):
        dtype = next(self.autoencoder.parameters()).dtype
        encoded = self.autoencoder.decode(latent.to(dtype)).float().mean(dim=1).clamp(-1, 1)[0].cpu().numpy()
        return depth_tools.unsqueeze_from_ceiling(encoded, self.sky_ceiling)

    @torch.no_grad()
    def encode_depth(self, encoded):
        depth_image = torch.from_numpy(np.asarray(encoded, dtype=np.float32))[None, None].expand(-1, 3, -1, -1)
        dtype = next(self.autoencoder.parameters()).dtype
        return self.autoencoder.encode(depth_image.to(self.device, dtype))

    def depth_output(self, log_depth, rgb_uint8, output_size=None):
        log_depth = log_depth.astype(np.float32)
        if self.sky_ceiling < 1.0:
            sky_mask = log_depth >= rgbd.SKY_VALUE
        else:
            sky_mask = np.zeros(log_depth.shape, dtype=bool)
        log_depth_out, sky_mask_out = log_depth, sky_mask
        if output_size is not None and tuple(output_size) != (log_depth.shape[1], log_depth.shape[0]):
            width, height = output_size
            log_depth_out = resize_log_depth(log_depth, width, height)
            sky_mask_out = resize_mask(sky_mask, width, height)
        return DepthOutput(log_depth_out, sky_mask_out, log_depth, Image.fromarray(rgb_uint8))

    def prepare_image(self, image, resolution):
        image = image.convert("RGB")
        if resolution is None:
            resolution = 512 if image.width * image.height < SMALL_IMAGE_PIXELS else 1024
        width, height = depth_tools.choose_resolution(image.width, image.height, resolution)
        return np.asarray(image.resize((width, height), Image.LANCZOS))

    @torch.no_grad()
    def image_to_depth(self, image, resolution=None, steps=4, average_last=2, seeds=(0,)):
        """resolution: 512 or 1024, None = 512 under 0.5 MP. Several seeds need load_text_encoder=False."""
        seeds = list(seeds)
        if len(seeds) > 1 and self.text_encoder is not None:
            raise ValueError(
                "Several seeds are for the image -> depth pipeline only: load it with "
                "IrisZPipeline.from_pretrained(..., load_text_encoder=False)"
            )
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return self.depth_from_rgb(self.prepare_image(image, resolution), steps, average_last, seeds, image.size)

    @torch.no_grad()
    def depth_from_rgb(self, rgb_uint8, steps=4, average_last=2, seeds=(0,), output_size=None):
        with self.autocast():
            rgb_latent = self.encode_image(rgb_uint8)
            members = rgbd.sample_depth(
                self.model,
                self.empty_context,
                rgb_latent,
                list(seeds),
                steps,
                average_last=average_last,
                decode=self.decode_log_depth,
            )
            if not average_last:
                members = [self.decode_log_depth(latent) for latent in members]
        log_depth = members[0] if len(members) == 1 else rgbd.ensemble(members)
        return self.depth_output(log_depth, rgb_uint8, output_size)

    @torch.no_grad()
    def text_to_image(self, prompt, width=1024, height=1024, steps=50, guidance=None, seed=0):
        guidance = self.config["default_guidance"] if guidance is None else guidance
        context = self.paired_context(prompt)
        with self.autocast():
            rgb_latent = rgbd.sample_rgb(self.model, context, height, width, seed, guidance, steps)
            return self.decode_image(rgb_latent)

    @torch.no_grad()
    def text_to_rgbd(self, prompt, width=1024, height=1024, steps=50, guidance=None, seed=0, joint_depth=False):
        """The depth is estimated again from the finished image (sharper); joint_depth=True returns the joint one."""
        guidance = self.config["default_guidance"] if guidance is None else guidance
        context = self.paired_context(prompt)
        with self.autocast():
            rgb_latent, depth_latent = rgbd.sample_joint(self.model, context, height, width, seed, guidance, steps)
            image = self.decode_image(rgb_latent)
            joint_log_depth = self.decode_log_depth(depth_latent) if joint_depth else None
        if joint_depth:
            return RGBDOutput(image, self.depth_output(joint_log_depth, np.asarray(image)))
        return RGBDOutput(image, self.depth_from_rgb(np.asarray(image), seeds=(seed,)))

    @torch.no_grad()
    def depth_to_image(self, depth, prompt="", resolution=1024, steps=50, guidance=None, seed=0, depth_format="depth"):
        """depth: a DepthOutput, a log depth array, or a depth map image read with depth_format
        ("depth": bright = far, "disparity": bright = near, "colours": our colouring)."""
        guidance = self.config["default_guidance"] if guidance is None else guidance
        if isinstance(depth, (str, Path)):
            depth = Image.open(depth)
        if isinstance(depth, Image.Image):
            encoded = depth_tools.read_depth_map(depth, depth_format, resolution, self.sky_ceiling)
        else:
            if isinstance(depth, DepthOutput):
                log_depth, sky_mask = depth.log_depth_internal, depth.sky_mask_internal()
            else:
                log_depth, sky_mask = np.asarray(depth), None
            encoded = depth_tools.encode_log_depth(log_depth, sky_mask, self.sky_ceiling)
            height, width = encoded.shape
            if (width * height) ** 0.5 > resolution * 1.1 or width % 16 or height % 16:
                new_width, new_height = depth_tools.enlarged_size(width, height, resolution)
                encoded = np.asarray(
                    Image.fromarray(encoded).resize((new_width, new_height), Image.BILINEAR), dtype=np.float32
                )
        context = self.prompt_context(prompt)
        with self.autocast():
            depth_latent = self.encode_depth(encoded)
            rgb_latent = rgbd.sample_rgb_from_depth(self.model, context, depth_latent, seed, guidance, steps)
            return self.decode_image(rgb_latent)

    def paired_context(self, prompt):
        context = self.prompt_context(prompt)
        if context[0].shape[0] == 1:
            context = (torch.cat([context[0]] * 2), torch.cat([context[1]] * 2))
        return context
