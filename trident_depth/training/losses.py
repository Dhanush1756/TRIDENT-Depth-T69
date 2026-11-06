"""Masked losses with valid-neighbour gradients and Laplace uncertainty."""
import math
import torch


def masked_mean(value, mask):
    if value.shape != mask.shape:
        raise ValueError("loss and mask must be aligned")
    active = mask.bool()
    # where removes invalid target NaNs; empty support remains connected to autograd.
    safe = torch.where(active, value, torch.zeros_like(value))
    return safe.sum() / active.sum().clamp_min(1)


def masked_l1(prediction, target, mask):
    safe_target = torch.where(mask.bool(), target, prediction.detach())
    return masked_mean((prediction - safe_target).abs(), mask)


def gradient_loss(prediction, target, mask):
    safe_target = torch.where(mask.bool(), target, torch.zeros_like(target))
    terms = []
    for axis in (-1, -2):
        pair_mask = mask.narrow(axis, 1, mask.shape[axis] - 1) & mask.narrow(axis, 0, mask.shape[axis] - 1)
        error = (torch.diff(prediction, dim=axis) - torch.diff(safe_target, dim=axis)).abs()
        terms.append(masked_mean(error, pair_mask))
    return sum(terms)


def depth_edge_weights(target, mask, threshold_m=0.05):
    safe = torch.where(mask.bool(), target, torch.zeros_like(target))
    weights = torch.ones_like(target)
    for axis in (-1, -2):
        pair = mask.narrow(axis, 1, mask.shape[axis] - 1) & mask.narrow(axis, 0, mask.shape[axis] - 1)
        hit = (torch.diff(safe, dim=axis).abs() >= threshold_m) & pair
        weights.narrow(axis, 1, weights.shape[axis] - 1).add_(hit.to(weights.dtype))
        weights.narrow(axis, 0, weights.shape[axis] - 1).add_(hit.to(weights.dtype))
    return weights


def combined_loss(output, target, gt_mask, *, observations=None, input_mask=None, noisy=False,
                  weights=None, use_uncertainty=True):
    weights = weights or dict(depth=1.0, gradient=0.2, edge=0.1, uncertainty=0.05, measurement=0.2)
    pred = output["depth"]
    if pred.ndim != 4 or pred.shape[1] != 1 or target.shape != pred.shape or gt_mask.shape != pred.shape or gt_mask.dtype != torch.bool:
        raise ValueError("loss requires Bx1xHxW tensors and bool masks")
    if set(weights) != {"depth", "gradient", "edge", "uncertainty", "measurement"} or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("loss weights must be finite nonnegative values for all five terms")
    if not torch.isfinite(pred).all() or not torch.isfinite(target[gt_mask]).all() or torch.any(target[gt_mask] <= 0):
        raise ValueError("predictions must be finite; valid reference depths must be finite positive metres")
    safe_target = torch.where(gt_mask, target, pred.detach())
    error = (pred - safe_target).abs()
    terms = dict(depth=masked_mean(error, gt_mask), gradient=gradient_loss(pred, target, gt_mask),
                 edge=masked_mean(error * depth_edge_weights(target, gt_mask), gt_mask))
    if use_uncertainty and "log_scale" in output:
        if output["log_scale"].shape != pred.shape or not torch.isfinite(output["log_scale"]).all():
            raise ValueError("log_scale must match depth and contain finite values")
        s = output["log_scale"].clamp(-6, 6)
        # Laplace NLL: b=exp(s), error/b + log(b), constant log(2) omitted.
        terms["uncertainty"] = masked_mean(torch.exp(-s) * error + s, gt_mask)
    else:
        terms["uncertainty"] = pred.sum() * 0
    if observations is not None:
        if input_mask is None or input_mask.shape != pred.shape or observations.shape != pred.shape:
            raise ValueError("measurement loss requires aligned observations and input mask")
        terms["measurement"] = masked_l1(pred, observations, input_mask) * (0.25 if noisy else 1.0)
    else:
        terms["measurement"] = pred.sum() * 0
    return sum(weights[k] * value for k, value in terms.items()), terms
