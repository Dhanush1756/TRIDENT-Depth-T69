"""Explicit-unit loaders for exported NYU/KITTI pairs; never infer depth units."""
from pathlib import Path
import math
import numpy as np
from PIL import Image
from .sample import DepthSample, clean_depth


def load_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValueError("depth/mask must be .npy, .png, .tif, or .tiff")
    with Image.open(path) as image:
        return np.array(image)


def load_sample(record, root, *, min_depth=0.0, max_depth=100.0):
    factor = record["depth_scale_to_m"]
    if isinstance(factor, bool) or not isinstance(factor, (float, int)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("depth_scale_to_m must be explicitly specified, finite and positive")
    root = Path(root)
    raw = load_array(root / record["depth"])
    if raw.ndim != 2 or raw.dtype.kind not in "uif":
        raise ValueError("depth must be numeric HxW, not a colour visualization")
    mask = load_array(root / record["mask"]) if record.get("mask") else None
    depth, valid = clean_depth(raw.astype(np.float64) * factor, mask,
                               min_depth=min_depth, max_depth=max_depth)
    rgb = None
    if record.get("rgb"):
        with Image.open(root / record["rgb"]) as image:
            if image.mode != "RGB":
                raise ValueError("guidance image must be RGB; convert explicitly before loading")
            rgb = np.array(image)
    return DepthSample(record["id"], record["scene_id"], depth, valid, rgb)
