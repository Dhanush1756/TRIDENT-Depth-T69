"""Valid-neighbour metric depth jumps; missing pixels never create boundaries."""
import numpy as np
from ..data.sample import clean_depth


def depth_edges(depth, mask, threshold_m=0.05):
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("threshold_m must be positive finite metres")
    depth, mask = clean_depth(depth, mask, max_depth=float(np.finfo(np.float32).max))
    edges = np.zeros_like(mask)
    horizontal = (np.abs(np.diff(depth, axis=1)) >= threshold_m) & mask[:, 1:] & mask[:, :-1]
    vertical = (np.abs(np.diff(depth, axis=0)) >= threshold_m) & mask[1:] & mask[:-1]
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:] |= vertical
    edges[:-1] |= vertical
    return edges
