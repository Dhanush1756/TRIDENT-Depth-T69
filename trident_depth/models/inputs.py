"""Scale/degradation-conditioned feature packing in metric depth units."""
import torch


def pack_features(rgb, initial, validity, scale, protocol):
    if initial.ndim != 4 or initial.shape[1] != 1 or validity.shape != initial.shape:
        raise ValueError("initial depth and validity must be aligned Bx1xHxW")
    if scale not in (2, 4, 8) or protocol not in {"dense_sr", "sparse_completion"}:
        raise ValueError("invalid scale or protocol")
    if not torch.isfinite(initial).all() or not torch.isfinite(validity).all() or torch.any(initial < 0):
        raise ValueError("initial depth/validity must be finite; depth nonnegative")
    if torch.any((validity != 0) & (validity != 1)):
        raise ValueError("validity must be binary")
    shape = (initial.shape[0], 3, *initial.shape[2:])
    if rgb is None:
        rgb = torch.zeros(shape, dtype=initial.dtype, device=initial.device)
        present = torch.zeros_like(initial)
    else:
        if rgb.shape != shape or not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1)):
            raise ValueError("RGB must be aligned Bx3xHxW in [0,1]")
        present = torch.ones_like(initial)
    # Extra RGB-presence channel distinguishes missing guidance from a black image.
    return torch.cat([rgb, initial, validity.to(initial.dtype),
                      torch.full_like(initial, scale / 8),
                      torch.full_like(initial, float(protocol == "sparse_completion")), present], dim=1)
