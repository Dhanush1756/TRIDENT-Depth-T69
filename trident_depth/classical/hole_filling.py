"""Nearest valid observation filling, with an optional spatial distance cap."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from ..data.sample import clean_depth


def fill_holes(depth, mask, max_distance=None):
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance < 0):
        raise ValueError("max_distance must be finite and nonnegative")
    if not mask.any():
        return depth, mask
    distances, indices = distance_transform_edt(~mask, return_indices=True)
    valid = np.ones_like(mask) if max_distance is None else distances <= max_distance
    out = depth[tuple(indices)]
    return np.where(valid, out, 0).astype(np.float32), valid
