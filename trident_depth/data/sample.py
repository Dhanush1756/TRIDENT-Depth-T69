"""Canonical array contracts: metric float32 depth, bool masks, uint8 RGB."""
from dataclasses import dataclass
import numpy as np


def clean_depth(depth, mask=None, *, min_depth=0.0, max_depth=100.0):
    array = np.asarray(depth)
    if array.ndim != 2 or 0 in array.shape or array.dtype.kind not in "uif":
        raise ValueError("depth must be a nonempty numeric HxW array")
    if not np.isfinite([min_depth, max_depth]).all() or not 0 <= min_depth < max_depth:
        raise ValueError("invalid depth range")
    valid = np.isfinite(array) & (array > min_depth) & (array <= max_depth)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != array.shape or mask.dtype.kind not in "buif":
            raise ValueError("mask must be a numeric/bool array matching depth")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("mask must contain only 0, 1, or 255")
        valid &= mask != 0
    return np.where(valid, array, 0).astype(np.float32), valid


@dataclass(frozen=True)
class DepthSample:
    sample_id: str
    scene_id: str
    depth: np.ndarray
    valid: np.ndarray
    rgb: np.ndarray | None = None

    def __post_init__(self):
        if not self.sample_id or not self.scene_id:
            raise ValueError("sample_id and scene_id are required")
        if (self.depth.ndim != 2 or 0 in self.depth.shape or self.depth.dtype != np.float32
                or self.valid.shape != self.depth.shape or self.valid.dtype != np.bool_):
            raise ValueError("expected float32 HxW depth and matching bool mask")
        if not np.isfinite(self.depth).all() or np.any(self.depth[self.valid] <= 0):
            raise ValueError("valid depth must be finite and positive")
        if np.any(self.depth[~self.valid] != 0):
            raise ValueError("invalid pixels must be zero")
        if self.rgb is not None and (self.rgb.shape != (*self.depth.shape, 3) or self.rgb.dtype != np.uint8):
            raise ValueError("RGB must be aligned HxWx3 uint8; registration cannot be inferred")
