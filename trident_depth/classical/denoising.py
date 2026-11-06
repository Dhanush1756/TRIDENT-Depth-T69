"""Local median denoising excluding all invalid measurements."""
import numpy as np
from ..data.sample import clean_depth


def denoise(depth, mask, radius=1):
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    if type(radius) is not int or not 1 <= radius <= 8:
        raise ValueError("radius must be 1..8")
    h, w = depth.shape
    padded = np.pad(np.where(mask, depth, np.inf), radius, constant_values=np.inf)
    windows = np.stack([padded[y:y + h, x:x + w] for y in range(2 * radius + 1) for x in range(2 * radius + 1)])
    counts = np.isfinite(windows).sum(axis=0)
    windows.sort(axis=0)
    indices = np.maximum(counts - 1, 0) // 2
    out = np.take_along_axis(windows, indices[None], axis=0)[0]
    # Denoising keeps the original support; filling is an explicit separate method.
    return clean_depth(out, mask, max_depth=float(np.finfo(np.float32).max))
