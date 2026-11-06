"""Validity-normalized nearest, bilinear and bicubic baselines."""
import numpy as np
from PIL import Image
from ..data.sample import clean_depth


def interpolate(depth, mask, size, method="bilinear"):
    methods = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
               "bicubic": Image.Resampling.BICUBIC}
    if method not in methods or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("invalid interpolation method or (height, width)")
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    shape = (size[1], size[0])
    numerator = np.asarray(Image.fromarray(depth).resize(shape, methods[method]))
    support = np.asarray(Image.fromarray(mask.astype(np.float32)).resize(shape, methods[method]))
    valid = support > 1e-6
    out = np.divide(numerator, support, out=np.zeros(size, np.float32), where=valid)
    # Cubic kernels can overshoot: retain the observed positive metric range.
    if mask.any():
        out[valid] = np.clip(out[valid], depth[mask].min(), depth[mask].max())
    return clean_depth(out, valid, max_depth=float(np.finfo(np.float32).max))
