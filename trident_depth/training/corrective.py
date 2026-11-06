"""Frozen A1/B0 contract and objective; legacy losses remain unchanged."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from ..config import PipelineConfig
from ..models.residual_v2 import ResidualCNNV2
from .losses import combined_loss, masked_l1


def approved_configuration(row):
    if row not in {"A1", "B0"}:
        raise ValueError("only A1 and B0 are implemented")
    return dict(manifest="../../data/nyu/converted/manifest_development.json",
        pipeline_config="../phase5/dense_sr_4x_pipeline.json",
        dataset="Official NYU labelled release; audited development partition",
        model="residual_cnn_v2", initializer="bicubic",
        output=f"../../artifacts/training/corrective_v1_{row}_dense4_seed69",
        epochs=2, batch_size=1, learning_rate=1e-4, weight_decay=0.0, device="cuda", seed=69,
        loss_weights=dict(depth=1.0, gradient=0.2, edge=0.1,
                          uncertainty=0.05 if row == "A1" else 0.0, measurement=0.2),
        decoder_upsampling="nearest", experiment_id=row, contract_version="A1_B0_v1",
        objective="L_J" if row == "A1" else "L_R", alpha=1.0,
        initialization="exact_zero_v1", uncertainty_enabled=row == "A1")


def verify_configuration(path):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg != approved_configuration(cfg.get("experiment_id")):
        raise ValueError("configuration differs from the frozen A1/B0 contract")
    pipeline = PipelineConfig.load(path.parent / cfg["pipeline_config"])
    # Freeze values independently: comparing a file to itself would not detect edits.
    # Semantic comparison also survives Git's Windows/Linux newline conversion.
    reference = PipelineConfig(protocol="dense_sr",scale=4,seed=69,min_depth=0.0,max_depth=10,
        target_size=None,density=.05,sampling="random",noise_std_m=.01,hole_fraction=.1)
    if pipeline != reference:
        raise ValueError("corrective pipeline differs from approved dense 4x pipeline")
    return cfg, pipeline


def make_model(cfg):
    return ResidualCNNV2(upsampling=cfg["decoder_upsampling"], alpha=cfg["alpha"],
                        uncertainty_enabled=cfg["uncertainty_enabled"], initialization=cfg["initialization"])


def corrective_loss(model, batch, config, device, *, weights=None, return_prediction=False):
    if not isinstance(model, ResidualCNNV2) or config.protocol != "dense_sr" or config.scale != 4:
        raise ValueError("corrective objective requires V2 dense 4x")
    expected = approved_configuration("A1" if model.uncertainty_enabled else "B0")["loss_weights"]
    if weights != expected:
        raise ValueError("objective weights differ from the approved control")
    b = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    if not b["rgb_present"].all():
        raise ValueError("paired controls require RGB")
    out = model(b["rgb"], b["initial"], b["observed_support"], config.scale, config.protocol)
    loss, terms = combined_loss(out, b["target"], b["target_valid"], weights=weights,
                                use_uncertainty=model.uncertainty_enabled)
    terms["measurement"] = masked_l1(F.avg_pool2d(out["depth"], 4, 4), b["observations"], b["input_valid"])
    loss = loss + weights["measurement"] * terms["measurement"]
    # Detached diagnostics never enter the objective or create head gradients.
    with torch.no_grad():
        mask = b["target_valid"]
        r = out["residual"][mask]
        terms.update({"weighted_"+k:weights[k]*v.detach() for k,v in list(terms.items())})
        if r.numel():
            e = (b["initial"] - torch.where(mask, b["target"], b["initial"]))[mask]
            terms.update(correction_mean_m=r.mean(), correction_abs_m=r.abs().mean(),
                         correction_rms_m=r.square().mean().sqrt(), correction_max_m=r.abs().max(),
                         correction_p50_m=r.abs().quantile(.5), correction_p90_m=r.abs().quantile(.9),
                         correction_p99_m=r.abs().quantile(.99), mse_cross_term_m2=(2*e*r).mean(),
                         mse_correction_term_m2=r.square().mean(),
                         corrected_pixel_improvement_fraction=((e+r).abs()<e.abs()).float().mean())
            if model.uncertainty_enabled:
                s = out["log_scale"][mask]
                error = (out["depth"]-torch.where(mask,b["target"],out["depth"]))[mask].abs()
                scale = s.exp()
                terms.update(uncertainty_scale_mean_m=scale.mean(),
                    uncertainty_clamp_fraction=((s<=-6)|(s>=6)).float().mean(),
                    uncertainty_coverage50=(error<=scale*0.69314718056).float().mean(),
                    uncertainty_coverage90=(error<=scale*2.30258509299).float().mean(),
                    uncertainty_coverage95=(error<=scale*2.99573227355).float().mean())
    return (loss, terms, out["depth"]) if return_prediction else (loss, terms)
