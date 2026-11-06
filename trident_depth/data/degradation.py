"""Distinct dense super-resolution and synthetic sparse completion protocols."""
from dataclasses import dataclass
import hashlib
import numpy as np
from ..config import PipelineConfig
from .sample import DepthSample, clean_depth
from .preprocessing import resize_sample


@dataclass(frozen=True)
class PreparedSample:
    target: DepthSample
    input_depth: np.ndarray
    input_valid: np.ndarray
    metadata: dict


def degrade(sample: DepthSample, config: PipelineConfig) -> PreparedSample:
    target = resize_sample(sample, config.target_size) if config.target_size else sample
    depth, mask = clean_depth(target.depth, target.valid, min_depth=config.min_depth, max_depth=config.max_depth)
    target = DepthSample(target.sample_id, target.scene_id, depth, mask, target.rgb)
    h, w = depth.shape
    seed = int.from_bytes(hashlib.sha256(f"{config.seed}:{target.sample_id}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    metadata = {"protocol": config.protocol, "scale": config.scale, "seed": config.seed,
                "sample_seed": seed, "units": "metres", "target_shape": [h, w]}
    if config.protocol == "dense_sr":
        s = config.scale
        if h % s or w % s:
            raise ValueError("dense_sr target dimensions must be divisible by scale; set target_size explicitly")
        # Dense protocol is strict: every contributing target pixel must be valid.
        if not mask.all():
            raise ValueError("dense_sr requires fully valid dense ground truth; crop/select a valid region explicitly")
        low = depth.reshape(h // s, s, w // s, s).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
        valid = np.ones(low.shape, dtype=bool)
        metadata["downsampling"] = "nonoverlapping area mean; fully valid blocks"
    else:
        valid = np.zeros((h, w), dtype=bool)
        if config.sampling == "random":
            valid = (rng.random((h, w)) < config.density) & mask
        elif config.density > 0:
            stride = max(1, int(round(config.density ** -0.5)))
            y, x = rng.integers(0, stride, size=2)
            valid[y::stride, x::stride] = True
            valid &= mask
        if config.hole_fraction > 0:
            # Central rectangle with approximately requested image area removed.
            hh = max(1, int(round(h * config.hole_fraction ** 0.5)))
            hw = max(1, int(round(w * config.hole_fraction ** 0.5)))
            y, x = (h - hh) // 2, (w - hw) // 2
            valid[y:y + hh, x:x + hw] = False
        low = np.zeros((h, w), np.float32)
        low[valid] = depth[valid] + rng.normal(0, config.noise_std_m, int(valid.sum()))
        low, valid = clean_depth(low, valid, min_depth=config.min_depth, max_depth=config.max_depth)
        metadata.update(sampling=config.sampling, requested_density=config.density,
                        noise_std_m=config.noise_std_m, hole_fraction=config.hole_fraction,
                        spatial_scale=1, synthetic=True)
    metadata.update(input_shape=list(low.shape), valid_fraction=float(valid.mean()))
    return PreparedSample(target, low, valid, metadata)
