"""Read-only data/benchmark audit, untrained gradient probes, and proposed configs.

Never calls optimizer.step or the training runner. GPU budgets are estimates.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trident_depth.config import PipelineConfig
from trident_depth.data.dataset import DepthDataset
from trident_depth.data.io import load_sample
from trident_depth.data.sample import DepthSample
from trident_depth.data.degradation import degrade
from trident_depth.data.nyu_integrity import file_sha256
from trident_depth.models.checkpoint import MODELS
from trident_depth.classical.pipeline import reconstruct
from trident_depth.training.losses import combined_loss
from trident_depth.training.validation import audit_training_manifest
from trident_depth.training.state import seed_everything
from trident_depth.training.train import model_loss
from trident_depth.classical.interpolation import interpolate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh only the generated preparation audit/configs; no dataset or benchmark changes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = root / "docs/training_preparation"
    configs = root / "configs/phase5"
    if (output.exists() or configs.exists()) and not args.refresh:
        raise SystemExit("Preparation outputs already exist; refusing to overwrite")
    manifest = root / "data/nyu/converted/manifest_development.json"
    split = audit_training_manifest(manifest)
    source_report = json.loads((manifest.parent / "integrity_report.json").read_text(encoding="utf-8"))
    assert split["manifest_sha256"] == source_report["manifests_sha256"][manifest.name]
    assert split["sample_counts"] == {"test":654, "train":672, "validation":123}
    records = []
    for protocol in ("dense_sr", "sparse_completion"):
        path = root / f"docs/nyu/baseline_results_{protocol}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["results"]:
            with np.load(root / "artifacts/nyu/validation_baselines" / protocol / row["output"]) as arrays:
                error = arrays["prediction"].astype(np.float64) - arrays["ground_truth"].astype(np.float64)
                valid = arrays["ground_truth_valid"]
                assert np.isclose(np.sqrt(np.mean(error[valid]**2)), row["metrics"]["rmse_m"], atol=1e-12, rtol=0)
            records.append(row)
    assert len(records) == 288
    ranking = []
    for protocol in ("dense_sr", "sparse_completion"):
        for scale in (2, 4, 8):
            methods = {}
            for row in records:
                if row["protocol"] == protocol and row["scale"] == scale:
                    methods.setdefault(row["method"], []).append(row)
            averages = {method:{metric:float(np.mean([r["metrics"][metric] for r in rows]))
                               for metric in ("rmse_m", "mae_m", "ssim", "boundary_f1", "edge_preservation")}
                        for method,rows in methods.items()}
            ranking.append(dict(protocol=protocol, scale=scale, per_method=averages,
                best_rmse=min(averages, key=lambda m:averages[m]["rmse_m"]),
                best_boundary_f1=max(averages, key=lambda m:averages[m]["boundary_f1"])))
    seed_everything(69)
    torch.set_num_threads(2)
    profiles = []
    for name, model_type in MODELS.items():
        model = model_type(**(dict(upsampling="nearest") if name == "residual_cnn" else {}))
        before = {k:v.detach().clone() for k,v in model.state_dict().items()}
        parameters = sum(p.numel() for p in model.parameters())
        pointers = {p.untyped_storage().data_ptr() for p in model.parameters()}
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in pointers:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor
        depth = torch.full((1, 1, 128, 160), 2.0)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x:x):
            prediction = model(torch.zeros((1, 3, 128, 160)), depth, torch.ones_like(depth), 4)["depth"]
            prediction.mean().backward()
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        scaled = sum(saved.values()) * (480 * 640) / (128 * 160)
        profiles.append(dict(model=name, parameters=parameters, fp32_parameters_bytes=parameters*4,
            decoder_upsampling=getattr(model,"upsampling",None),
            fp32_adam_parameters_gradients_moments_bytes=parameters*16,
            diagnostic_shape=[128,160], retained_unique_nonparameter_storage_bytes=sum(saved.values()),
            estimated_retained_storage_full_resolution_bytes=int(scaled),
            not_gpu_peak_measurement=True, optimizer_steps=0))
    # Actual NYU values, deliberately cropped only for diagnostics, not benchmark/training data.
    val_rows = json.loads(manifest.read_text(encoding="utf-8"))
    row = next(r for r in val_rows if r["split"] == "validation")
    real = load_sample(row, manifest.parent, max_depth=10)
    real = DepthSample(real.sample_id, real.scene_id, real.depth[:128,:160], real.valid[:128,:160], real.rgb[:128,:160])
    probes = []
    for protocol in ("dense_sr", "sparse_completion"):
        for scale in (2,4,8):
            config = PipelineConfig(protocol=protocol, scale=scale, max_depth=10)
            sample = degrade(real, config)
            initializer = "bicubic" if protocol == "dense_sr" else "guided"
            initial = reconstruct(sample.input_depth, sample.input_valid, real.depth.shape, real.rgb, initializer)[0]
            d = torch.from_numpy(initial)[None,None]
            target = torch.from_numpy(real.depth.copy())[None,None]
            mask = torch.from_numpy(real.valid.copy())[None,None]
            for name, model_type in MODELS.items():
                model = model_type(**(dict(upsampling="nearest") if name == "residual_cnn" else {}))
                before = {k:v.detach().clone() for k,v in model.state_dict().items()}
                _, support = interpolate(sample.input_depth, sample.input_valid, real.depth.shape, "nearest")
                batch = dict(rgb=torch.from_numpy(real.rgb.copy().transpose(2,0,1))[None].float()/255,
                    rgb_present=torch.tensor([True]), initial=d, observed_support=torch.from_numpy(support)[None,None],
                    target=target, target_valid=mask, observations=torch.from_numpy(sample.input_depth)[None,None],
                    input_valid=torch.from_numpy(sample.input_valid)[None,None])
                loss, terms = model_loss(model, batch, config, "cpu")
                loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
                probes.append(dict(protocol=protocol, scale=scale, model=name, loss=float(loss.detach()),
                    terms={k:float(v.detach()) for k,v in terms.items()}, gradients_finite=True, optimizer_steps=0))
    experiments = []
    loss_weights = dict(depth=1., gradient=.2, edge=.1, uncertainty=.05, measurement=.2)
    for protocol in ("dense_sr", "sparse_completion"):
        for scale in (2,4,8):
            condition = f"{protocol}_{scale}x"
            pipeline = asdict(PipelineConfig(protocol=protocol, scale=scale, max_depth=10))
            write(configs / f"{condition}_pipeline.json", pipeline)
            for name in MODELS:
                experiment = dict(manifest="../../data/nyu/converted/manifest_development.json",
                    pipeline_config=f"{condition}_pipeline.json", dataset="Official NYU labelled release; audited development partition",
                    model=name, initializer="bicubic" if protocol=="dense_sr" else "guided",
                    output=f"../../artifacts/training/{condition}_{name}_seed69", epochs=20, batch_size=1,
                    learning_rate=.0001, weight_decay=0., device="cuda", seed=69, loss_weights=loss_weights)
                if name == "residual_cnn":
                    experiment["decoder_upsampling"] = "nearest"
                write(configs / f"{condition}_{name}.json", experiment)
                experiments.append(f"configs/phase5/{condition}_{name}.json")
    pilot = json.loads((configs / "dense_sr_4x_lightweight_cnn.json").read_text(encoding="utf-8"))
    pilot.update(epochs=2, output="../../artifacts/training/pilot_dense4_lightweight_seed69")
    write(configs / "pilot_dense4_lightweight.json", pilot)
    write(output / "audit.json", dict(split=split, classical_record_count=len(records), rankings=ranking,
        model_profiles=profiles, actual_nyu_gradient_probes=probes, proposed_experiments=experiments,
        official_test_used_for_selection=False, optimizer_steps=0, training_started=False,
        normalization=dict(rgb="uint8 / 255", depth="float32 metres, no per-image normalization", range_m=[0,10]),
        limitation="CPU diagnostic crops and analytical scaling do not measure GPU memory or GPU training time"))
    print(json.dumps(dict(split=split, profiles=profiles, gradient_probes=len(probes),
                         experiments=len(experiments), optimizer_steps=0), indent=2))


if __name__ == "__main__":
    main()
