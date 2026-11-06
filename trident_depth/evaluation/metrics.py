"""Metric definitions are documented in docs/evaluation_protocol.md."""
import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt
from ..classical.discontinuity import depth_edges


def evaluate(prediction, target, mask, *, data_range_m=10.0, edge_threshold_m=0.05, boundary_tolerance_px=1, ssim_window=7):
    pred, gt, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(mask)
    if pred.shape != gt.shape or mask.shape != gt.shape or gt.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("metrics require aligned 2D prediction/target and bool ground-truth mask")
    if not np.isfinite([data_range_m, edge_threshold_m, boundary_tolerance_px]).all() or data_range_m <= 0 or edge_threshold_m <= 0 or boundary_tolerance_px < 0:
        raise ValueError("invalid metric ranges or boundary parameters")
    if type(ssim_window) is not int or ssim_window < 3 or ssim_window % 2 != 1:
        raise ValueError("SSIM window must be an odd integer >=3")
    if not mask.any():
        raise ValueError("no valid ground-truth pixels for evaluation")
    if not np.isfinite(pred[mask]).all() or not np.isfinite(gt[mask]).all() or np.any(gt[mask] <= 0):
        raise ValueError("valid target must be positive finite; predictions must be finite")
    error = pred[mask] - gt[mask]
    mse = float(np.mean(error ** 2))
    rmse, mae = float(np.sqrt(mse)), float(np.mean(np.abs(error)))
    # JSON-safe exact-match PSNR: null plus an explicit flag means +infinity.
    psnr = None if mse == 0 else float(10 * np.log10(data_range_m ** 2 / mse))
    ssim = None
    if min(gt.shape) >= ssim_window:
        p, g = np.where(mask, pred, 0), np.where(mask, gt, 0)
        box = lambda x: uniform_filter(x, size=ssim_window, mode="constant")
        eligible = box(mask.astype(np.float64)) >= 1 - 1e-12
        border = ssim_window // 2
        eligible[:border] = False
        eligible[-border:] = False
        eligible[:, :border] = False
        eligible[:, -border:] = False
        if eligible.any():
            mp, mg = box(p), box(g)
            vp, vg = np.maximum(box(p * p) - mp * mp, 0), np.maximum(box(g * g) - mg * mg, 0)
            covariance = box(p * g) - mp * mg
            c1, c2 = (0.01 * data_range_m) ** 2, (0.03 * data_range_m) ** 2
            scores = ((2 * mp * mg + c1) * (2 * covariance + c2) /
                      ((mp * mp + mg * mg + c1) * (vp + vg + c2)))
            ssim = float(scores[eligible].mean())
    # Predicted missing/zero values are errors and can create boundaries; never mask them out.
    def metric_edges(d):
        edges = np.zeros_like(mask)
        for axis in (0, 1):
            left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            hit = (np.abs(d[right] - d[left]) >= edge_threshold_m) & mask[left] & mask[right]
            edges[left] |= hit
            edges[right] |= hit
        return edges
    pe, ge = metric_edges(np.where(mask, pred, 0)), metric_edges(np.where(mask, gt, 0))
    if not pe.any() and not ge.any():
        precision = recall = 1.0
    else:
        matched_p = int((pe & (distance_transform_edt(~ge) <= boundary_tolerance_px)).sum()) if ge.any() else 0
        matched_g = int((ge & (distance_transform_edt(~pe) <= boundary_tolerance_px)).sum()) if pe.any() else 0
        precision = matched_p / int(pe.sum()) if pe.any() else 0.0
        recall = matched_g / int(ge.sum()) if ge.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gradients = []
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        valid_pair = mask[left] & mask[right]
        pg, gg = np.diff(pred, axis=axis)[valid_pair], np.diff(gt, axis=axis)[valid_pair]
        gradients.extend((1 - np.abs(pg - gg) / (np.abs(pg) + np.abs(gg) + 1e-12)).tolist())
    return dict(rmse_m=rmse, mae_m=mae, psnr_db=psnr, psnr_infinite=(mse == 0), ssim=ssim,
                boundary_precision=precision, boundary_recall=recall, boundary_f1=f1,
                edge_preservation=float(np.mean(gradients)) if gradients else None,
                valid_pixels=int(mask.sum()), data_range_m=data_range_m,
                edge_threshold_m=edge_threshold_m, boundary_tolerance_px=boundary_tolerance_px,
                ssim_window=ssim_window)
