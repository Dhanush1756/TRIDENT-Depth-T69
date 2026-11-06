"""RGB gradient visualization and geometry-validated boundary refinement."""
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from .discontinuity import depth_edges
from .guided_filter import guided_filter


def rgb_edges(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("RGB must be HxWx3 uint8")
    gray = array.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8,
                         cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8)
    return np.clip(magnitude, 0, 1)


def boundary_refine(depth, mask, rgb, threshold_m=0.05):
    geometric = depth_edges(depth, mask, threshold_m)
    if rgb is None:
        return np.array(depth, copy=True), np.array(mask, copy=True), geometric.astype(np.float32)
    evidence = rgb_edges(rgb)
    if evidence.shape != depth.shape:
        raise ValueError("RGB is not aligned")
    confidence = evidence * maximum_filter(geometric.astype(np.float32), size=3)
    guided, valid = guided_filter(depth, mask, rgb.astype(np.float32) / 255)
    # RGB-only texture cannot alter the candidate: geometric support is required.
    out = (1 - confidence) * depth + confidence * guided
    return np.where(mask, out, 0).astype(np.float32), mask & valid, confidence
