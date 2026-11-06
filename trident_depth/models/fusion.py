"""Explicit normalized candidate fusion and hard/soft measurement consistency."""
import torch


def fuse(classical, edge_aware, neural, depth_confidence, boundary_confidence, log_scale, *,
         observations=None, observation_mask=None, observation_confidence=None):
    arrays = [classical, edge_aware, neural, depth_confidence, boundary_confidence, log_scale]
    if any(x.shape != classical.shape or not torch.isfinite(x).all() for x in arrays):
        raise ValueError("fusion inputs must have identical shape and finite values")
    c, b = depth_confidence.clamp(0, 1), boundary_confidence.clamp(0, 1)
    confidence_n = torch.exp(-log_scale.clamp(-6, 6)).clamp(max=100)
    raw = torch.stack([0.05 + c * (1 - b), 0.05 + b * c, 0.05 + confidence_n * (1 - c)], dim=0)
    weights = raw / raw.sum(dim=0, keepdim=True)
    fused = (weights * torch.stack([classical, edge_aware, neural], dim=0)).sum(dim=0)
    if observations is not None:
        if observation_mask is None or observation_confidence is None:
            raise ValueError("observations require validity and reliability confidence")
        if any(x.shape != fused.shape or not torch.isfinite(x).all()
               for x in (observations, observation_mask, observation_confidence)):
            raise ValueError("measurement inputs must be aligned and finite")
        alpha = observation_mask.to(fused.dtype) * observation_confidence.clamp(0, 1)
        fused = alpha * observations + (1 - alpha) * fused
    return fused, weights
