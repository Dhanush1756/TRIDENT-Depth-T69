"""CPU/GPU inference with checkpoint provenance checks and inspectable candidates."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .classical.pipeline import reconstruct
from .classical.edge_detection import boundary_refine


class InferenceEngine:
    def __init__(self, checkpoint=None, device="cpu"):
        self.model = None
        self.metadata = None
        self.device = device
        if checkpoint is not None:
            from .models.checkpoint import load_checkpoint
            self.model, self.metadata = load_checkpoint(checkpoint, require_trained=True, device=device)

    def run(self, depth, mask, size, rgb=None, *, method="classical", scale=4,
            protocol="dense_sr", noisy=False, ablations=()):
        if method not in {"deep", "hybrid"}:
            out, valid, maps = reconstruct(depth, mask, size, rgb, method)
            return dict(depth=out, valid=valid, maps=maps, checkpoint=None)
        if self.model is None:
            raise ValueError("deep/hybrid requires a trained checkpoint; none is configured")
        if method == "hybrid" and getattr(self.model, "uncertainty_enabled", True) is False:
            raise ValueError("B0 uncertainty is untrained/unavailable; hybrid confidence is disabled")
        import torch
        from .models.fusion import fuse
        allowed = {"depth_confidence", "geometry", "residual", "uncertainty", "adaptive_fusion", "measurement"}
        if not set(ablations) <= allowed:
            raise ValueError("unknown ablation")
        classical, valid, maps = reconstruct(depth, mask, size, rgb, "classical")
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, dtype=torch.float32)[None, None]
        initializer = (self.metadata or {}).get("config", {}).get("experiment", {}).get("initializer", "classical")
        if initializer not in {"classical", "bicubic", "guided"}:
            raise ValueError("unsupported checkpoint initializer")
        neural_initial = classical if initializer == "classical" else reconstruct(depth, mask, size, rgb, initializer)[0]
        initial = tensor(neural_initial)
        # Observed support is distinct from filled candidate support.
        if depth.shape == tuple(size):
            observed = mask.astype(bool)
        else:
            from .classical.interpolation import interpolate
            _, observed = interpolate(depth, mask, size, "nearest")
        validity = tensor(observed)
        rgb_tensor = None if rgb is None else torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device, dtype=torch.float32)[None] / 255
        with torch.inference_mode():
            output = self.model(rgb_tensor, initial, validity, scale, protocol)
            neural = initial if "residual" in ablations else output["depth"]
            edge, _, boundary = boundary_refine(classical, valid, rgb)
            confidence = np.exp(-distance_transform_edt(~observed) / 8).astype(np.float32) if observed.any() else np.zeros(size, np.float32)
            if "depth_confidence" in ablations:
                confidence[:] = 0.5
            if "geometry" in ablations:
                boundary[:] = 0
                edge = classical
            log_scale = output.get("log_scale", torch.zeros_like(initial))
            if "uncertainty" in ablations:
                log_scale = torch.zeros_like(initial)
            if method == "deep":
                fused, weights = neural, None
            elif "adaptive_fusion" in ablations:
                fused = (tensor(classical) + tensor(edge) + neural) / 3
                weights = torch.full((3, *initial.shape), 1 / 3, device=self.device)
            else:
                fused, weights = fuse(tensor(classical), tensor(edge), neural, tensor(confidence), tensor(boundary), log_scale)
            if depth.shape == tuple(size) and "measurement" not in ablations:
                alpha = validity * (0.25 if noisy else 1.0)
                fused = alpha * tensor(depth) + (1 - alpha) * fused
            out = fused[0, 0].cpu().numpy().astype(np.float32)
            maps.update(neural=neural[0, 0].cpu().numpy(), edge_aware=edge,
                        depth_confidence=confidence, boundary_confidence=boundary,
                        log_scale=log_scale[0, 0].cpu().numpy())
            if getattr(self.model, "uncertainty_enabled", True) is False:
                maps.pop("log_scale", None)
            if weights is not None:
                for i, name in enumerate(("classical", "edge", "neural")):
                    maps[f"weight_{name}"] = weights[i, 0, 0].cpu().numpy()
        return dict(depth=out, valid=np.isfinite(out) & (out > 0), maps=maps, checkpoint=self.metadata)
