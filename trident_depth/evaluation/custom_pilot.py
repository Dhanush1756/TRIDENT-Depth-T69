"""Paired validation-only comparison after the approved custom two-epoch pilot."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..config import PipelineConfig
from ..models.checkpoint import load_checkpoint
from ..training.colab import audit_pilot_dataset, best_checkpoint_path, sha256
from ..training.dataset import TorchDepthDataset
from ..training.portable import select_device
from ..training.state import seed_everything
from ..training.train import model_loss
from ..training.boundaries import boundary_f1_batch, BOUNDARY_EVALUATION
from ..training.validation import validation_errors
from .pilot_audit import require, tensor_hash, comparison_metrics

BASELINE_SHA256 = "a1fb5080360d9e693d43e67b3aa105da05be93af6baca6d1c84bee979ce3fb43"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-run", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-frames", type=Path, default=Path("docs/pilot_dense4/per_frame.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "comparison destination exists; refusing overwrite")
    require(sha256(args.baseline_checkpoint) == BASELINE_SHA256, "baseline checkpoint is not the audited epoch two")
    audit = audit_pilot_dataset(args.manifest)
    history = [json.loads(line) for line in (args.custom_run / "history.jsonl").read_text().splitlines() if line]
    require([r["epoch"] for r in history] == [1, 2], "comparison requires exactly two completed custom epochs")
    for row in history:
        require(row.get("validation_boundary_frames") == 123 and math.isfinite(row["validation_boundary_f1"]), "missing per-epoch boundary evaluation")
    for epoch in (1, 2):
        saved = torch.load(args.custom_run / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=True)
        require(saved["training_state"]["history"] == history[:epoch] and saved["metadata"]["epochs"] == epoch,
                "custom checkpoint/history mismatch")
        require(saved["scheduler_state"]["last_epoch"] == epoch and "optimizer_state" in saved,
                "custom scheduler/optimizer state missing")
    custom_path = best_checkpoint_path(args.custom_run)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    seed_everything(69)
    device = select_device(args.device)
    custom, custom_metadata = load_checkpoint(custom_path, device=device)
    baseline, baseline_metadata = load_checkpoint(args.baseline_checkpoint, device=device)
    require(custom_metadata["config"]["experiment"]["model"] == "residual_cnn" and sum(p.numel() for p in custom.parameters()) == 473059
            and custom.upsampling == "nearest", "custom architecture differs")
    require(custom_metadata["config"]["experiment"]["epochs"] == 2 and baseline_metadata["config"]["experiment"]["model"] == "lightweight_cnn", "unexpected model or epoch budget")
    require(custom_metadata.get("boundary_evaluation") == BOUNDARY_EVALUATION, "boundary protocol mismatch")
    for metadata in (custom_metadata, baseline_metadata):
        require(metadata["split_manifest_sha256"] == audit["manifest_sha256"] and metadata["config"]["experiment"]["initializer"] == "bicubic", "checkpoint dataset/initializer mismatch")
    pipeline_dict = custom_metadata["config"]["pipeline"]
    require(pipeline_dict == baseline_metadata["config"]["pipeline"] and pipeline_dict["protocol"] == "dense_sr" and pipeline_dict["scale"] == 4, "comparison protocols differ")
    pipeline = PipelineConfig(**pipeline_dict)
    dataset = TorchDepthDataset(args.manifest, pipeline, "validation", initializer="bicubic")
    references = json.loads(args.reference_frames.read_text())
    require(len(dataset) == len(references) == 123, "expected 123 reference validation frames")
    totals = {key:dict(squared=0., absolute=0., pixels=0, boundary=[], loss=[]) for key in ("bicubic", "baseline_cnn", "custom_residual")}
    frames = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
            target, mask = batch["target"], batch["target_valid"]
            reference = references[index]
            row = dataset.source.records[index]
            require(row["id"] == reference["id"] and row["official_split"] == "train" and row["scene_id"] == reference["scene_id"], "validation identity differs")
            require(tensor_hash(mask) == reference["mask_sha256"], "evaluation mask differs from baseline audit")
            require(all(sha256(args.manifest.parent / row[key]) == reference["source_files_sha256"][key] for key in ("rgb", "depth", "mask")), "evaluation files differ from baseline audit")
            predictions = dict(bicubic=batch["initial"])
            for key, model, metadata in (("baseline_cnn", baseline, baseline_metadata), ("custom_residual", custom, custom_metadata)):
                loss, _, prediction = model_loss(model, batch, pipeline, device,
                    weights=metadata["config"]["experiment"]["loss_weights"], return_prediction=True)
                predictions[key] = prediction.cpu()
                totals[key]["loss"].append(float(loss))
            frame = dict(id=row["id"], scene_id=row["scene_id"], mask_sha256=reference["mask_sha256"])
            for key, prediction in predictions.items():
                squared, absolute, count = validation_errors(prediction, target, mask)
                boundary = boundary_f1_batch(prediction, target, mask)[0]
                total = totals[key]
                total["squared"] += squared
                total["absolute"] += absolute
                total["pixels"] += count
                total["boundary"].append(boundary)
                frame[key] = dict(sum_squared_error_m2=squared, sum_absolute_error_m=absolute, valid_pixels=count,
                                  rmse_m=math.sqrt(squared/count), mae_m=absolute/count, boundary_f1=boundary)
            frames.append(frame)
            if (index + 1) % 20 == 0:
                print(f"Compared {index+1}/123 validation frames", flush=True)
    aggregate = {key:dict(rmse_m=math.sqrt(v["squared"]/v["pixels"]), mae_m=v["absolute"]/v["pixels"],
                         boundary_f1=float(np.mean(v["boundary"])), frames=len(frames), valid_pixels=v["pixels"]) for key,v in totals.items()}
    for key, reference_key in (("bicubic", "bicubic"), ("baseline_cnn", "cnn_epoch_2")):
        squared = sum(f[reference_key]["sum_squared_error_m2"] for f in references)
        absolute = sum(f[reference_key]["sum_absolute_error_m"] for f in references)
        pixels = sum(f[reference_key]["pixels"] for f in references)
        require(abs(aggregate[key]["rmse_m"] - math.sqrt(squared/pixels)) < 1e-6 and abs(aggregate[key]["mae_m"] - absolute/pixels) < 1e-6, "baseline replay differs from audited reference")
    best_epoch = custom_metadata["epochs"]
    best_row = history[best_epoch-1]
    require(abs(aggregate["custom_residual"]["rmse_m"] - best_row["validation_rmse_m"]) < 1e-6
            and abs(aggregate["custom_residual"]["mae_m"] - best_row["validation_mae_m"]) < 1e-6
            and abs(aggregate["custom_residual"]["boundary_f1"] - best_row["validation_boundary_f1"]) < 1e-6,
            "custom validation replay differs from history")
    report = dict(status="validation-only comparison; no training or inference export", best_custom_epoch=best_epoch,
        aggregate=aggregate, improvement_vs_bicubic=comparison_metrics(aggregate["bicubic"], aggregate["custom_residual"]),
        improvement_vs_baseline=comparison_metrics(aggregate["baseline_cnn"], aggregate["custom_residual"]),
        boundary_f1_delta={key:aggregate["custom_residual"]["boundary_f1"] - aggregate[key]["boundary_f1"] for key in ("bicubic", "baseline_cnn")},
        boundary_evaluation=BOUNDARY_EVALUATION, split_audit=audit, history=history,
        custom_checkpoint_sha256=sha256(custom_path), baseline_checkpoint_sha256=sha256(args.baseline_checkpoint),
        reference_frames_sha256=sha256(args.reference_frames), custom_metadata=custom_metadata, baseline_metadata=baseline_metadata,
        device=str(device), torch_version=str(torch.__version__), comparison_wall_seconds=time.perf_counter()-started,
        caveat="Loss includes uncertainty only for custom network, so total losses are not identical objectives. Boundary confidence head is not directly supervised by the approved loss.")
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "per_frame.json").write_text(json.dumps(frames, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Custom dense 4x two-epoch pilot comparison", "", "Validation only: same 123 frames, reference masks and files. No test evaluation or weight export.", "", "| Method | RMSE (m) | MAE (m) | Boundary F1 (frame mean) |", "| --- | ---: | ---: | ---: |"]
    for key, v in aggregate.items():
        lines.append(f"| {key} | {v['rmse_m']:.10f} | {v['mae_m']:.10f} | {v['boundary_f1']:.8f} |")
    lines += ["", report["caveat"], "", "Full provenance and both epoch logs, including losses, runtime and CUDA peak memory, are in comparison.json."]
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
