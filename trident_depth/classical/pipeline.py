"""Classical pipeline with inspectable intermediate depth candidates."""
import numpy as np
from .interpolation import interpolate
from .denoising import denoise
from .hole_filling import fill_holes
from .joint_bilateral import joint_bilateral
from .guided_filter import guided_filter
from .edge_detection import boundary_refine


def reconstruct(depth, mask, size, rgb=None, method="classical", preserve_measurements=False):
    if method not in {"nearest", "bilinear", "bicubic", "classical", "guided", "joint_bilateral"}:
        raise ValueError("unknown classical method")
    initial, valid = interpolate(depth, mask, size, method if method in {"nearest", "bilinear", "bicubic"} else "bilinear")
    if method in {"nearest", "bilinear", "bicubic"}:
        return initial, valid, {"interpolated": initial}
    if rgb is not None and (rgb.shape != (*size, 3) or rgb.dtype != np.uint8):
        raise ValueError("RGB must match target dimensions as uint8")
    filtered, fm = denoise(initial, valid)
    filled, fm = fill_holes(filtered, fm)
    guide = (rgb.astype(np.float32) / 255 if rgb is not None else
             filled / max(float(filled.max()), 1e-6))
    bilateral, bm = joint_bilateral(filled, fm, guide)
    guided, gm = guided_filter(filled, fm, guide)
    if method == "guided":
        out, output_mask = guided, gm
    elif method == "joint_bilateral":
        out, output_mask = bilateral, bm
    else:
        out, output_mask, _ = boundary_refine(bilateral, bm, rgb)
    if preserve_measurements:
        if depth.shape != tuple(size):
            raise ValueError("hard consistency is defined only for aligned same-resolution observations")
        out = np.where(mask, depth, out).astype(np.float32)
        output_mask = output_mask | mask
    return out, output_mask, {"interpolated": initial, "denoised": filtered,
                             "filled": filled, "joint_bilateral": bilateral, "guided": guided}
