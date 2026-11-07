"""Frozen B0 + heuristic classical/geometry fusion, dense 4x only.

No learned confidence, training, legacy dispatch changes, or test-data access.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
import torch
from .classical.pipeline import reconstruct
from .classical.interpolation import interpolate
from .classical.hole_filling import fill_holes
from .classical.edge_detection import boundary_refine
from .models.checkpoint import load_checkpoint
from .models.residual_v2 import ResidualCNNV2
from .training.corrective import approved_configuration
from .training.colab import NYU_MANIFEST_SHA256

B0_SHA256 = "a6416c47a16ec78e97c8b734ae7fb23cba4dbbeb5f5564a3e02d61a4a159c1b2"
B0_MEMBER = "corrective_v1_B0_dense4_seed69/epoch_002.pt"
METHODS = ("bicubic", "b0", "b0_consistency_only", "b0_plus_classical",
           "b0_plus_geometry", "hybrid_v1_dense4")


@dataclass(frozen=True)
class HybridConfig:
    version: str = "hybrid_v1_dense4"
    protocol: str = "dense_sr"
    scale: int = 4
    classical_method: str = "joint_bilateral"
    min_depth_m: float = 1e-6
    max_depth_m: float = 10.0
    prior_neural: float = 0.6
    prior_classical: float = 0.2
    prior_geometry: float = 0.2
    measurement_tau_m: float = 0.02
    local_tau_m: float = 0.05
    agreement_tau_m: float = 0.05
    edge_threshold_m: float = 0.05
    agreement_floor: float = 0.05
    epsilon: float = 1e-6
    local_window: int = 3
    support_window: int = 5
    projection_iterations: int = 48
    max_target_pixels: int = 1024 * 1024

    def __post_init__(self):
        if (self.version, self.protocol, self.scale) != ("hybrid_v1_dense4", "dense_sr", 4):
            raise ValueError("hybrid_v1_dense4 supports dense_sr at 4x only")
        if self.classical_method not in {"joint_bilateral", "guided"}:
            raise ValueError("classical_method must be joint_bilateral or guided")
        positive = (self.min_depth_m, self.max_depth_m, self.prior_neural,
                    self.prior_classical, self.prior_geometry, self.measurement_tau_m,
                    self.local_tau_m, self.agreement_tau_m, self.edge_threshold_m, self.epsilon)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("depth, priors, scales and epsilon must be finite positive numbers")
        if not self.min_depth_m < self.max_depth_m <= 10:
            raise ValueError("depth limits must be positive and within B0's 10 m reference range")
        if not np.isfinite(self.agreement_floor) or not 0 <= self.agreement_floor <= 1:
            raise ValueError("agreement_floor must lie in [0,1]")
        for value in (self.local_window, self.support_window):
            if type(value) is not int or value < 1 or value > 15 or value % 2 != 1:
                raise ValueError("windows must be odd integers between 1 and 15")
        if type(self.scale) is not int or type(self.projection_iterations) is not int or not 32 <= self.projection_iterations <= 64:
            raise ValueError("scale must be integer 4 and projection_iterations must be 32..64")
        if type(self.max_target_pixels) is not int or self.max_target_pixels < 16:
            raise ValueError("max_target_pixels must be an integer >=16")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def area_mean(depth):
    a = np.asarray(depth)
    if a.ndim != 2 or any(v % 4 for v in a.shape):
        raise ValueError("area_mean requires 2D dimensions divisible by four")
    h, w = a.shape
    return a.reshape(h//4, 4, w//4, 4).mean(axis=(1, 3), dtype=np.float64)


def _observations(depth, mask, config):
    d, m = np.asarray(depth), np.asarray(mask)
    if d.ndim != 2 or not d.size or d.dtype.kind not in "uif" or m.shape != d.shape or m.dtype != np.bool_:
        raise ValueError("observations require numeric HxW depth and matching bool validity")
    if not m.any():
        raise ValueError("dense 4x requires at least one valid LR measurement")
    if not np.isfinite(d[m]).all() or np.any(d[m] < config.min_depth_m) or np.any(d[m] > config.max_depth_m):
        raise ValueError("valid LR observations must be finite and within configured metre limits")
    if d.size * 16 > config.max_target_pixels:
        raise ValueError("target exceeds max_target_pixels")
    return np.where(m, d, 0).astype(np.float32), m.copy()


def _rgb(rgb, shape):
    if rgb is None or np.asarray(rgb).shape != (*shape, 3) or np.asarray(rgb).dtype != np.uint8:
        raise ValueError("B0 hybrid requires registered target-resolution HxWx3 uint8 RGB")
    return np.asarray(rgb)


def project_area_means(candidate, observations, mask, config):
    """Bounded least-squares projection onto valid 4x4 block means.

    x_i=clip(y_i+lambda_block,lower,upper); bisection solves mean(x)=LR.
    This preserves area measurements, not invented HR point observations.
    """
    low, valid = _observations(observations, mask, config)
    y = np.asarray(candidate)
    shape = tuple(v*4 for v in low.shape)
    if y.shape != shape or not np.isfinite(y).all():
        raise ValueError("projection requires finite target-resolution candidate")
    # Round inward to representable FP32 limits, so casting cannot cross a bound.
    lower, upper = np.float32(config.min_depth_m), np.float32(config.max_depth_m)
    if float(lower) < config.min_depth_m:
        lower = np.nextafter(lower, np.float32(np.inf))
    if float(upper) > config.max_depth_m:
        upper = np.nextafter(upper, np.float32(-np.inf))
    lower, upper = float(lower), float(upper)
    y = np.clip(y.astype(np.float64), lower, upper)
    h, w = low.shape
    blocks = y.reshape(h, 4, w, 4).transpose(0, 2, 1, 3).reshape(h, w, 16)
    target = np.where(valid, low, blocks.mean(axis=-1))
    left = lower - blocks.max(axis=-1)
    right = upper - blocks.min(axis=-1)
    for _ in range(config.projection_iterations):
        middle = (left + right) / 2
        average = np.clip(blocks + middle[..., None], lower, upper).mean(axis=-1)
        below = average < target
        left = np.where(below, middle, left)
        right = np.where(below, right, middle)
    projected = np.clip(blocks + ((left+right)/2)[..., None], lower, upper)
    projected = np.where(valid[..., None], projected, blocks)
    return projected.reshape(h, w, 4, 4).transpose(0, 2, 1, 3).reshape(shape).astype(np.float32)


def normalized_blend(candidates, scores, active):
    arrays = np.stack(candidates).astype(np.float64)
    reliability = np.stack(scores).astype(np.float64)
    if arrays.ndim != 3 or arrays.shape != reliability.shape or len(active) != len(arrays):
        raise ValueError("candidates and scores require identical KxHxW shapes")
    if not np.isfinite(arrays).all() or not np.isfinite(reliability).all() or np.any(reliability < 0):
        raise ValueError("candidate values and nonnegative scores must be finite")
    reliability *= np.asarray(active, dtype=bool)[:, None, None]
    total = reliability.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError("every pixel needs positive active reliability")
    weights = reliability / total
    return (arrays*weights).sum(axis=0).astype(np.float32), weights.astype(np.float32)


def comparison_outputs(candidates, scores, observations, mask, config, *, include_comparisons=True):
    """Apply the same projection to blends; comparisons are outside production timing."""
    n, c, g = (candidates[k] for k in ("neural_bounded", "classical", "geometry"))
    outputs, weights, deltas = {}, {}, {}
    if include_comparisons:
        outputs["b0_consistency_only"] = project_area_means(n, observations, mask, config)
    variants = [("hybrid_v1_dense4", (1,1,1))]
    if include_comparisons:
        variants += [("b0_plus_classical", (1,1,0)), ("b0_plus_geometry", (1,0,1))]
    for name, active in variants:
        mixed, weight = normalized_blend((n,c,g), scores, active)
        result = project_area_means(mixed, observations, mask, config)
        outputs[name], weights[name] = result, weight
        deltas[name] = float(np.abs(result.astype(np.float64)-mixed).mean())
    return outputs, weights, deltas


def refine_candidates(neural, observations, mask, rgb, config=HybridConfig(), *, include_comparisons=True):
    """Pure deterministic postprocessing; accepts ONLY depth from the model."""
    low, valid = _observations(observations, mask, config)
    size = tuple(v*4 for v in low.shape)
    rgb = _rgb(rgb, size)
    neural = np.asarray(neural)
    if neural.shape != size or neural.dtype != np.float32 or not np.isfinite(neural).all() or np.any(neural <= 0):
        raise ValueError("neural candidate must be finite positive target-resolution float32 metres")
    n = np.clip(neural, config.min_depth_m, config.max_depth_m)
    started = time.perf_counter()
    classical, cm, _ = reconstruct(low, valid, size, rgb, config.classical_method)
    classical, _ = fill_holes(classical, cm)
    c = np.clip(classical, config.min_depth_m, config.max_depth_m).astype(np.float32)
    classical_seconds = time.perf_counter()-started
    started = time.perf_counter()
    g, _, geometry_gate = boundary_refine(n, np.ones(size, bool), rgb, config.edge_threshold_m)
    g = np.clip(g, config.min_depth_m, config.max_depth_m).astype(np.float32)
    geometry_seconds = time.perf_counter()-started
    started = time.perf_counter()
    up = lambda a: np.repeat(np.repeat(a, 4, axis=0), 4, axis=1)
    support = uniform_filter(up(valid.astype(np.float64)), size=config.support_window, mode="nearest")
    measurement, local = [], []
    for candidate in (n, c, g):
        drift = np.abs(area_mean(candidate) - low)
        measurement.append(up(np.where(valid, np.exp(-drift/config.measurement_tau_m), 0)))
        local.append(np.exp(-np.abs(candidate-median_filter(candidate, size=config.local_window, mode="nearest"))/config.local_tau_m))
    agreement = np.exp(-np.abs(c-n)/config.agreement_tau_m)
    geometry_agreement = np.exp(-np.abs(g-n)/config.agreement_tau_m)
    floor = config.agreement_floor
    scores = [config.prior_neural*(config.epsilon+support*measurement[0]*local[0]),
              config.prior_classical*(config.epsilon+support*measurement[1]*local[1])*(floor+(1-floor)*agreement),
              config.prior_geometry*(config.epsilon+support*measurement[2]*local[2])*geometry_gate*(floor+(1-floor)*geometry_agreement)]
    reliability_seconds = time.perf_counter()-started
    started = time.perf_counter()
    candidates = dict(neural_bounded=n, classical=c, geometry=g)
    outputs, weights, projection_delta = comparison_outputs(candidates, scores, low, valid, config,
        include_comparisons=include_comparisons)
    fusion_projection_seconds = time.perf_counter()-started
    return dict(outputs=outputs, weights=weights,
        candidates=candidates,
        b0_range_clip_mean_abs_correction_m=float(np.abs(n.astype(np.float64)-neural).mean()),
        b0_range_clip_fraction=float(np.mean(n!=neural)),
        reliability=dict(validity_support=support.astype(np.float32), agreement=agreement.astype(np.float32),
            geometry_gate=geometry_gate.astype(np.float32), scores=np.stack(scores).astype(np.float64)),
        projection_mean_abs_correction_m=projection_delta,
        timing=dict(classical_refinement_seconds=classical_seconds, geometry_refinement_seconds=geometry_seconds,
            reliability_seconds=reliability_seconds, fusion_projection_seconds=fusion_projection_seconds))


def load_b0(checkpoint, device="cpu"):
    """Hash before loading; never extract/rewrite supplied checkpoint archives."""
    path = Path(checkpoint)
    if path.suffix.lower() not in {".pt", ".zip"}:
        raise ValueError("checkpoint must be a .pt file or the original B0 .zip archive")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.namelist().count(B0_MEMBER) != 1:
                raise ValueError("archive must contain exactly the approved B0 epoch-two member")
            raw = archive.read(B0_MEMBER)
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != B0_SHA256:
        raise ValueError("B0 checkpoint SHA256 mismatch")
    model, metadata = load_checkpoint(io.BytesIO(raw), require_trained=True, device=device)
    if not isinstance(model, ResidualCNNV2) or model.configuration() != dict(
            upsampling="nearest", alpha=1.0, uncertainty_enabled=False, initialization="exact_zero_v1"):
        raise ValueError("checkpoint must reconstruct the original B0 architecture")
    if (metadata.get("config", {}).get("experiment") != approved_configuration("B0")
            or metadata.get("split_manifest_sha256") != NYU_MANIFEST_SHA256
            or metadata.get("epochs") != 2 or metadata.get("uncertainty_available") is not False
            or metadata.get("boundary_head_supervised") is not False
            or sum(p.numel() for p in model.parameters()) != 473059):
        raise ValueError("B0 checkpoint provenance/configuration mismatch")
    if not all(p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("B0 parameters must remain FP32")
    model.eval().requires_grad_(False)
    return model, metadata


class HybridV1Dense4:
    """Separate opt-in engine. Does not replace legacy InferenceEngine."""
    def __init__(self, checkpoint, config=HybridConfig(), device="cpu"):
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; no silent device fallback")
        self.model, self.metadata = load_b0(checkpoint, self.device)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, depth, mask, rgb, *, scale=4, protocol="dense_sr", prepared=None):
        if scale != 4 or protocol != "dense_sr":
            raise ValueError("hybrid_v1_dense4 supports dense_sr at 4x only")
        total_started = time.perf_counter()
        low, valid = _observations(depth, mask, self.config)
        size = tuple(v*4 for v in low.shape)
        rgb = _rgb(rgb, size)
        preprocessing_started = time.perf_counter()
        if prepared is None:
            initial, initial_valid, _ = reconstruct(low, valid, size, rgb, "bicubic")
            # Missing LR blocks: explicit numerical fallback, not a trained completion claim.
            initial, _ = fill_holes(initial, initial_valid)
            _, support = interpolate(low, valid, size, "nearest")
            rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None]/255
            initial_tensor = torch.from_numpy(np.ascontiguousarray(initial))[None,None]
            support_tensor = torch.from_numpy(np.ascontiguousarray(support))[None,None]
        else:
            # Evaluator passes original TorchDepthDataset tensors, unchanged.
            rgb_tensor, initial_tensor, support_tensor = prepared
            if (rgb_tensor.shape != (1,3,*size) or initial_tensor.shape != (1,1,*size)
                    or support_tensor.shape != (1,1,*size) or rgb_tensor.dtype != torch.float32
                    or initial_tensor.dtype != torch.float32 or support_tensor.dtype != torch.bool):
                raise ValueError("prepared tensors must match original single-frame preprocessing")
            if not torch.isfinite(rgb_tensor).all() or not torch.isfinite(initial_tensor).all():
                raise ValueError("prepared tensors must be finite")
            initial = initial_tensor[0,0].detach().cpu().numpy().copy()
        rgb_tensor = rgb_tensor.to(self.device)
        initial_tensor = initial_tensor.to(self.device)
        support_tensor = support_tensor.to(self.device)
        self._sync()
        preprocessing_seconds = time.perf_counter()-preprocessing_started
        neural_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            # Deliberately read only depth: neither confidence head enters fusion.
            prediction_tensor = self.model(rgb_tensor, initial_tensor, support_tensor, 4, "dense_sr")["depth"]
        self._sync()
        neural_seconds = time.perf_counter()-neural_started
        transfer_started = time.perf_counter()
        neural = prediction_tensor[0,0].cpu().numpy().copy()
        transfer_seconds = time.perf_counter()-transfer_started
        refined = refine_candidates(neural, low, valid, rgb, self.config, include_comparisons=False)
        refined["outputs"] = dict(bicubic=initial.copy(), b0=neural, **refined["outputs"])
        refined["timing"].update(preprocessing_h2d_seconds=preprocessing_seconds,
            neural_forward_seconds=neural_seconds, prediction_d2h_seconds=transfer_seconds,
            total_pipeline_seconds=time.perf_counter()-total_started)
        refined.update(version=self.config.version, units="metres", config=asdict(self.config),
            depth=refined["outputs"]["hybrid_v1_dense4"],
            valid=np.ones(size, bool), checkpoint_sha256=B0_SHA256,
            lr_valid_fraction=float(valid.mean()), partial_lr_input_unvalidated=not bool(valid.all()))
        return refined
