"""Aligned resizing with mask-normalized depth and nearest-neighbour validity."""
import numpy as np
from PIL import Image
from .sample import DepthSample, clean_depth


def resize_sample(sample: DepthSample, size: tuple[int, int]) -> DepthSample:
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("size must be positive integer (height, width)")
    if sample.depth.shape == tuple(size):
        return sample
    pil_size = (size[1], size[0])
    # Numerator/support resized identically: missing zero values never bias averages.
    weights = np.asarray(Image.fromarray(sample.valid.astype(np.float32)).resize(pil_size, Image.Resampling.BILINEAR))
    numerator = np.asarray(Image.fromarray(sample.depth).resize(pil_size, Image.Resampling.BILINEAR))
    nearest = np.asarray(Image.fromarray(sample.valid).resize(pil_size, Image.Resampling.NEAREST))
    depth = np.divide(numerator, weights, out=np.zeros(size, np.float32), where=weights > 1e-8)
    valid = nearest & (weights > 1e-8)
    depth, valid = clean_depth(depth, valid, max_depth=float(np.finfo(np.float32).max))
    rgb = None if sample.rgb is None else np.asarray(Image.fromarray(sample.rgb).resize(pil_size, Image.Resampling.BILINEAR))
    return DepthSample(sample.sample_id, sample.scene_id, depth, valid, rgb)
