"""Separately gated two-epoch A1/B0 runner. Preparation does not authorize execution."""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ..config import PipelineConfig
from ..logging_utils import configure_logging
from ..models.checkpoint import MODELS, save_checkpoint, resume_checkpoint
from .state import seed_everything, capture_state
from .validation import audit_training_manifest, validation_errors
from .dataset import TorchDepthDataset
from .corrective import corrective_loss as model_loss, verify_configuration, make_model
from .colab import audit_pilot_dataset
from .corrective_monitoring import fixed_training_probe
from .corrective_data import audit_validation_reference, REFERENCE_SHA256
from .portable import select_device, recover_history, atomic_json
from .boundaries import boundary_f1_batch, BOUNDARY_EVALUATION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--execute-training", action="store_true", help="required deliberate opt-in")
    parser.add_argument("--resume", type=Path, help="resume a complete epoch-boundary checkpoint after approval")
    parser.add_argument("--manifest", type=Path, help="runtime dataset location; leaves experiment identity unchanged")
    parser.add_argument("--output", type=Path, help="runtime checkpoint directory, e.g. mounted Google Drive")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="runtime device (default: auto)")
    parser.add_argument("--record-boundary-f1", action="store_true", help="record existing boundary metric on validation after each epoch")
    parser.add_argument("--provenance", type=Path, help="notebook environment/source record to embed in each checkpoint")
    args = parser.parse_args(argv)
    if not args.execute_training:
        parser.error("training is disabled unless --execute-training is supplied")
    cfg, pipeline = verify_configuration(args.experiment)
    args.record_boundary_f1 = True
    base = args.experiment.parent
    manifest = args.manifest.resolve() if args.manifest else (base / cfg["manifest"]).resolve()
    pipeline = PipelineConfig.load(base / cfg["pipeline_config"])
    if pipeline.seed != cfg["seed"]:
        parser.error("experiment and pipeline seed must match")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(cfg["seed"])
    device = select_device(args.device)
    split_audit = audit_pilot_dataset(manifest)
    reference = audit_validation_reference(manifest)
    reference_by_id = {r["id"]:r for r in reference}
    initializer = cfg.get("initializer", "classical")
    train = TorchDepthDataset(manifest, pipeline, "train", initializer)
    val = TorchDepthDataset(manifest, pipeline, "validation", initializer)
    if not len(train) or not len(val):
        parser.error("nonempty scene-disjoint train and validation splits are required")
    output = args.output.resolve() if args.output else (base / cfg["output"]).resolve()
    if output.exists() and args.resume is None:
        parser.error("output exists; use a new run directory")
    if args.resume is not None and (not output.is_dir() or args.resume.resolve().parent != output):
        parser.error("resume checkpoint must belong to the configured existing run directory")
    # Windows/Colab-safe: no worker processes; seeded shuffle independent of iteration.
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0))
    # Constant LR preserves the approved experiment; state is explicit for resume.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    configure_logging()
    provenance = dict(dataset=cfg["dataset"], split_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      config=dict(experiment=cfg, pipeline=asdict(pipeline)), torch_version=str(torch.__version__),
                      device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      deterministic_algorithms=True)
    provenance["split_audit"] = split_audit
    provenance["runtime_paths"] = dict(manifest=str(manifest), output=str(output))
    provenance["scheduler"] = "constant learning rate"
    provenance["model_configuration"] = model.configuration()
    provenance["uncertainty_available"] = cfg["uncertainty_enabled"]
    provenance["boundary_head_supervised"] = False
    provenance["residual_output_mapping"] = "clamp_min(bicubic + alpha * raw_residual, 1e-6)"
    provenance["initialization_applies_only_to_fresh_construction"] = True
    provenance["validation_reference_sha256"] = REFERENCE_SHA256
    if args.record_boundary_f1:
        provenance["boundary_evaluation"] = BOUNDARY_EVALUATION
    if args.provenance:
        record = json.loads(args.provenance.read_text(encoding="utf-8"))
        if (record.get("experiment_sha256") != hashlib.sha256(args.experiment.read_bytes()).hexdigest()
                or record.get("pipeline_sha256") != hashlib.sha256((base / cfg["pipeline_config"]).read_bytes()).hexdigest()
                or record.get("environment", {}).get("torch") != str(torch.__version__)):
            parser.error("notebook provenance configuration or PyTorch version mismatch")
        provenance["environment_record"] = record
        provenance["environment_record_sha256"] = hashlib.sha256(args.provenance.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    start_epoch, best_rmse, best_epoch = 1, float("inf"), 0
    history_rows = []
    if args.resume is not None:
        previous_metadata = torch.load(args.resume, map_location="cpu", weights_only=True)["metadata"]
        if previous_metadata.get("boundary_evaluation") != provenance.get("boundary_evaluation"):
            parser.error("resume must preserve the recorded boundary evaluation configuration")
        if previous_metadata.get("environment_record") != provenance.get("environment_record"):
            parser.error("resume must preserve the recorded source/environment provenance")
        state = resume_checkpoint(args.resume, model, optimizer, generator,
            manifest_sha256=provenance["split_manifest_sha256"], experiment=cfg, pipeline=asdict(pipeline), scheduler=scheduler)
        start_epoch, best_rmse, best_epoch = state["epoch"] + 1, state["best_rmse"], state["best_epoch"]
        if start_epoch > cfg["epochs"]:
            parser.error("no epochs remain in the requested resume budget")
        history_rows = recover_history(output / "history.jsonl", state)
    else:
        (output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        # A fixed training frame, independent of shuffled iteration and RNG state.
        from torch.utils.data import default_collate
        diagnostic = fixed_training_probe(model, default_collate([train[0]]), pipeline, cfg, device)
        train_losses, val_losses = [], []
        train_terms, val_terms = {}, {}
        for batch in train_loader:
            if not batch["target_valid"].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, components = model_loss(model, batch, pipeline, device, weights=cfg.get("loss_weights"))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            train_terms.setdefault("preclip_gradient_norm", []).append(float(norm))
            train_terms.setdefault("clipping_applied", []).append(float(norm > 5))
            optimizer.step()
            train_losses.append(float(loss.detach()))
            for key,value in components.items():
                train_terms.setdefault(key, []).append(float(value.detach()))
        model.eval()
        sum_squared, sum_absolute, pixels = 0.0, 0.0, 0
        boundary_scores = []
        validation_frames = []
        with torch.inference_mode():
            for batch in val_loader:
                if batch["target_valid"].any():
                    loss, components, prediction = model_loss(model, batch, pipeline, device,
                        weights=cfg.get("loss_weights"), return_prediction=True)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite validation loss")
                    val_losses.append(float(loss))
                    for key,value in components.items():
                        val_terms.setdefault(key, []).append(float(value))
                    squared, absolute, count = validation_errors(prediction, batch["target"].to(device), batch["target_valid"].to(device))
                    sum_squared += squared
                    sum_absolute += absolute
                    pixels += count
                    frame_id = batch["sample_id"][0]
                    frame_reference = reference_by_id[frame_id]
                    mask_sha256 = hashlib.sha256(batch["target_valid"].contiguous().numpy().tobytes()).hexdigest()
                    if mask_sha256 != frame_reference["mask_sha256"]:
                        raise ValueError("validation mask differs from original pilot")
                    if args.record_boundary_f1:
                        scores = boundary_f1_batch(prediction, batch["target"], batch["target_valid"])
                        boundary_scores.extend(scores)
                    validation_frames.append(dict(id=frame_id, scene_id=frame_reference["scene_id"],
                        mask_sha256=mask_sha256, sum_squared_error_m2=squared,
                        sum_absolute_error_m=absolute, pixels=count, rmse_m=float(np.sqrt(squared/count)),
                        mae_m=absolute/count, boundary_f1=scores[0]))
        if not train_losses or not val_losses:
            raise ValueError("train/validation has no valid supervised batches")
        rmse = float(np.sqrt(sum_squared / pixels))
        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
        row = dict(epoch=epoch, train_loss=float(np.mean(train_losses)), validation_loss=float(np.mean(val_losses)),
                   validation_rmse_m=rmse, validation_mae_m=sum_absolute / pixels,
                   best_epoch=best_epoch, best_validation_rmse_m=best_rmse)
        row["fixed_training_probe"] = diagnostic
        row["validation_frames"] = validation_frames
        if args.record_boundary_f1:
            row.update(validation_boundary_f1=float(np.mean(boundary_scores)),
                       validation_boundary_frames=len(boundary_scores))
        row.update(train_components={k:float(np.mean(v)) for k,v in train_terms.items()},
                   validation_components={k:float(np.mean(v)) for k,v in val_terms.items()},
                   learning_rate=optimizer.param_groups[0]["lr"],
                   epoch_seconds=time.perf_counter()-started,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)
        scheduler.step()
        history_rows.append(row)
        training_state = capture_state(generator, epoch, best_rmse, best_epoch)
        training_state["history"] = history_rows
        save_checkpoint(output / f"epoch_{epoch:03d}.pt", model, cfg["model"],
                        dict(provenance, trained=True, epochs=epoch), optimizer,
                        training_state=training_state, scheduler=scheduler)
        atomic_json(output / "history.jsonl", history_rows, lines=True)
        atomic_json(output / "best_checkpoint.json", dict(epoch=best_epoch,
            checkpoint=f"epoch_{best_epoch:03d}.pt", validation_rmse_m=best_rmse))
        logging.info("Epoch %s: %s", epoch, row)


if __name__ == "__main__":
    main()
