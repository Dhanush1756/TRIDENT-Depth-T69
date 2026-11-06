"""Mask-weighted local linear guided filter using RGB luminance guidance."""
import numpy as np
from scipy.ndimage import uniform_filter
from ..data.sample import clean_depth


def guided_filter(depth, mask, guidance, radius=2, epsilon=1e-3):
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    guide = np.asarray(guidance, np.float64)
    if guide.ndim == 3 and guide.shape[2] == 3:
        guide = guide @ np.array([0.299, 0.587, 0.114])
    if guide.shape != depth.shape or not np.isfinite(guide).all() or guide.min() < 0 or guide.max() > 1:
        raise ValueError("guidance must be aligned grayscale or RGB in [0,1]")
    if type(radius) is not int or not 1 <= radius <= 32 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("radius must be 1..32, epsilon positive finite")
    box = lambda x: uniform_filter(x.astype(np.float64), size=2 * radius + 1, mode="constant")
    weight = box(mask)
    mean = lambda x: np.divide(box(x * mask), weight, out=np.zeros_like(weight), where=weight > 0)
    mean_g, mean_d = mean(guide), mean(depth)
    variance = np.maximum(0, mean(guide * guide) - mean_g * mean_g)
    covariance = mean(guide * depth) - mean_g * mean_d
    a = covariance / (variance + epsilon)
    b = mean_d - a * mean_g
    support = weight > 0
    count = box(support)
    avg = lambda x: np.divide(box(x * support), count, out=np.zeros_like(count), where=count > 0)
    out = avg(a) * guide + avg(b)
    valid = count > 0
    if mask.any():
        out[valid] = np.clip(out[valid], depth[mask].min(), depth[mask].max())
    return clean_depth(out, valid, max_depth=float(np.finfo(np.float32).max))
