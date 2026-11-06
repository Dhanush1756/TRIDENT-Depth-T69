"""Local joint bilateral filter with explicit validity-normalized weights."""
import numpy as np
from ..data.sample import clean_depth


def joint_bilateral(depth, mask, guidance, radius=2, sigma_spatial=2.0, sigma_range=0.1):
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    guide = np.asarray(guidance, np.float32)
    if guide.ndim == 2:
        guide = guide[..., None]
    if guide.ndim != 3 or guide.shape[:2] != depth.shape or not np.isfinite(guide).all():
        raise ValueError("guidance must be finite, aligned HxW or HxWxC, normalized to [0,1]")
    if guide.min() < 0 or guide.max() > 1:
        raise ValueError("guidance must lie in [0,1]")
    if type(radius) is not int or not 1 <= radius <= 16 or not np.isfinite([sigma_spatial, sigma_range]).all() or min(sigma_spatial, sigma_range) <= 0:
        raise ValueError("radius must be 1..16 and sigmas finite positive")
    h, w = depth.shape
    pd = np.pad(depth, radius)
    pm = np.pad(mask, radius)
    pg = np.pad(guide, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    numerator = np.zeros_like(depth, dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sl = (slice(radius + dy, radius + dy + h), slice(radius + dx, radius + dx + w))
            dist = np.mean((guide - pg[sl]) ** 2, axis=2)
            weights = np.exp(-(dy * dy + dx * dx) / (2 * sigma_spatial ** 2) - dist / (2 * sigma_range ** 2)) * pm[sl]
            numerator += weights * pd[sl]
            denominator += weights
    valid = denominator > 1e-12
    out = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
    return clean_depth(out, valid, max_depth=float(np.finfo(np.float32).max))
