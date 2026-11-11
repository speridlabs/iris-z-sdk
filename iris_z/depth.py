import math

import numpy as np
from PIL import Image


def sky_threshold(sky_ceiling):
    return (sky_ceiling + 1.0) / 2.0


def unsqueeze_from_ceiling(encoded, sky_ceiling):
    """Real depth back from [-1, sky_ceiling] to [-1, 1]; sky to +1."""
    encoded = np.clip(np.asarray(encoded, dtype=np.float64), -1.0, 1.0)
    if sky_ceiling >= 1.0:
        return encoded
    real = np.clip(-1.0 + (encoded + 1.0) * 2.0 / (sky_ceiling + 1.0), -1.0, 1.0)
    return np.where(encoded > sky_threshold(sky_ceiling), 1.0, real)


def encode_log_depth(log_depth, sky_mask=None, sky_ceiling=1.0):
    """The 2nd to 98th percentile of the non-sky pixels spans [-1, sky_ceiling]; sky is +1."""
    log_depth = np.asarray(log_depth, dtype=np.float64)
    if sky_mask is None:
        sky_mask = np.zeros(log_depth.shape, dtype=bool)
    real = log_depth[~sky_mask] if (~sky_mask).any() else log_depth.ravel()
    low, high = np.percentile(real, [2, 98])
    encoded = np.clip(2.0 * (log_depth - low) / max(high - low, 1e-6) - 1.0, -1.0, 1.0)
    encoded = -1.0 + (encoded + 1.0) * (sky_ceiling + 1.0) / 2.0
    encoded[sky_mask] = 1.0
    return encoded.astype(np.float32)


# the colour scale stops at 85 % of turbo_r: its last 15 % is near-black indigo that hides far depth next to black sky
FAR_COLOUR_END = 0.85


def colourize(log_depth, sky_mask=None):
    """Near = red, far = blue, sky black; the 1st to 99.5th percentile of the non-sky depth spans the scale."""
    import matplotlib

    log_depth = np.asarray(log_depth, dtype=np.float64)
    if sky_mask is None:
        sky_mask = np.zeros(log_depth.shape, dtype=bool)
    real = log_depth[~sky_mask] if (~sky_mask).any() else log_depth.ravel()
    low, high = np.percentile(real, [1, 99.5])
    normalized = np.clip((log_depth - low) / max(high - low, 1e-6), 0, 1) * FAR_COLOUR_END
    colours = (matplotlib.colormaps["turbo_r"](normalized)[..., :3] * 255).astype(np.uint8)
    colours[sky_mask] = 0
    return colours


def choose_resolution(source_width, source_height, target_side, multiple=16, area_tolerance=0.2):
    """About target_side^2 pixels, same aspect ratio, multiples of 16, never larger than the source."""
    source_aspect = source_width / source_height
    target_area = min(target_side * target_side, source_width * source_height)
    best = None
    for height in range(multiple, source_height + 1, multiple):
        for width in range(multiple, source_width + 1, multiple):
            area = width * height
            if abs(area - target_area) > area_tolerance * target_area:
                continue
            aspect_error = abs(math.log((width / height) / source_aspect))
            key = (round(aspect_error, 6), abs(area - target_area))
            if best is None or key < best[0]:
                best = (key, width, height)
    if best is None:
        raise ValueError(f"no size near {target_side}^2 fits {source_width}x{source_height}")
    return best[1], best[2]


def enlarged_size(source_width, source_height, target_side):
    """Like choose_resolution, but smaller sources are enlarged."""
    area = source_width * source_height
    if area < target_side * target_side:
        factor = (target_side * target_side / area) ** 0.5
        source_width, source_height = round(source_width * factor), round(source_height * factor)
    return choose_resolution(source_width, source_height, target_side)


DEPTH_MAP_FORMATS = ("depth", "disparity", "colours")


def read_depth_map(image, depth_format, target_side, sky_ceiling=1.0):
    """depth_format: "depth" (bright = far), "disparity" (bright = near) or "colours" (our colouring)."""
    if depth_format == "colours":
        return read_coloured_depth(image, target_side, sky_ceiling)
    values = np.asarray(image, dtype=np.float64)
    if values.ndim == 3:
        values = np.asarray(image.convert("L"), dtype=np.float64)
    gray = values / (65535.0 if values.max() > 255 else 255.0)
    width, height = enlarged_size(gray.shape[1], gray.shape[0], target_side)
    gray = np.asarray(Image.fromarray(gray.astype(np.float32)).resize((width, height), Image.BILINEAR), np.float64)
    if depth_format == "disparity":
        log_depth = -np.log(np.maximum(gray, 1e-3))
    else:
        log_depth = np.log(np.maximum(gray, 1e-3))
    return encode_log_depth(log_depth, sky_ceiling=sky_ceiling)


def read_coloured_depth(image, target_side, sky_ceiling=1.0):
    import matplotlib
    from scipy.spatial import cKDTree

    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    positions = np.linspace(0.0, 1.0, 1024)
    scale_colours = matplotlib.colormaps["turbo_r"](positions * FAR_COLOUR_END)[:, :3] * 255.0
    _, nearest = cKDTree(scale_colours).query(pixels.reshape(-1, 3))
    position = positions[nearest].reshape(pixels.shape[0], pixels.shape[1])
    black = pixels.max(axis=2) < 24
    position[black] = 1.0
    width, height = enlarged_size(image.width, image.height, target_side)
    position = np.asarray(
        Image.fromarray(position.astype(np.float32)).resize((width, height), Image.BILINEAR), dtype=np.float64
    )
    sky = np.asarray(Image.fromarray(black.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 127
    return encode_log_depth(position, sky_mask=sky, sky_ceiling=sky_ceiling)
