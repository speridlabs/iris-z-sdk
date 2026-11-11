import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import DEFAULT_REPO, VARIANTS, IrisZPipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_paths(path):
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [path]


def save_depth(output, out_dir, name):
    np.save(out_dir / f"{name}_depth.npy", output.log_depth)
    output.colourized().save(out_dir / f"{name}_depth.png")
    output.image.save(out_dir / f"{name}_input.png")


def main():
    parser = argparse.ArgumentParser(prog="python -m iris_z.sample")
    parser.add_argument("task", choices=["depth", "image", "rgbd", "paint"])
    parser.add_argument("--model", choices=VARIANTS, default="big")
    parser.add_argument("--weights", default=DEFAULT_REPO, help="a Hugging Face repo or a local folder")
    parser.add_argument("--image", help="depth: a photo or a folder of photos")
    parser.add_argument("--depth-map", help="paint: a depth map image")
    parser.add_argument("--depth-format", choices=["depth", "disparity", "colours"], default="depth")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=None, help="depth / paint: 512 or 1024")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    needs_text = args.task != "depth" and bool(args.prompt.strip())
    pipeline = IrisZPipeline.from_pretrained(args.weights, variant=args.model, load_text_encoder=needs_text)

    if args.task == "depth":
        for path in image_paths(args.image):
            output = pipeline.image_to_depth(
                Image.open(path), resolution=args.resolution, steps=args.steps or 4, seeds=(args.seed,)
            )
            save_depth(output, out_dir, path.stem)
            print(f"{path} -> {out_dir / (path.stem + '_depth.png')}")
        return

    settings = {"steps": args.steps or 50, "guidance": args.guidance, "seed": args.seed}
    name = f"{args.task}_seed{args.seed}"
    if args.task == "image":
        image = pipeline.text_to_image(args.prompt, width=args.width, height=args.height, **settings)
        image.save(out_dir / f"{name}.png")
    elif args.task == "rgbd":
        output = pipeline.text_to_rgbd(args.prompt, width=args.width, height=args.height, **settings)
        output.image.save(out_dir / f"{name}.png")
        save_depth(output.depth, out_dir, name)
    else:
        image = pipeline.depth_to_image(
            Image.open(args.depth_map),
            args.prompt,
            resolution=args.resolution or 1024,
            depth_format=args.depth_format,
            **settings,
        )
        image.save(out_dir / f"{name}.png")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
