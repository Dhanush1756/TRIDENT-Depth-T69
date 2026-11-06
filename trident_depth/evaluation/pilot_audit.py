"""Audit an existing dense pilot and evaluate validation only; never train."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import zipfile
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from ..config import PipelineConfig
from ..models.baseline import LightweightCNN
from ..training.colab import audit_pilot_dataset, sha256
from ..training.dataset import TorchDepthDataset
from ..training.train import model_loss
from ..training.validation import validation_errors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_hash(value):
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def audit_checkpoint(state, epoch, run, history, expected_loader_state):
    """Load component state without stepping an optimizer or resuming training."""
    require(state["format_version"] == 1 and state["model_name"] == "lightweight_cnn", "unexpected model format")
    require(state["model_configuration"] == {}, "unexpected model configuration")
    require(state["metadata"] == dict(run, trained=True, epochs=epoch), "checkpoint metadata differs from run")
    saved = state["training_state"]
    require(saved["epoch"] == epoch and saved["history"] == history[:epoch], "checkpoint history mismatch")
    best = min(history[:epoch], key=lambda row: row["validation_rmse_m"])
    require(saved["best_epoch"] == best["epoch"] and saved["best_rmse"] == best["validation_rmse_m"], "best-epoch metadata mismatch")
    model = LightweightCNN()
    model.load_state_dict(state["model_state"], strict=True)
    require(all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite model parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=run["config"]["experiment"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    groups = state["optimizer_state"]["param_groups"]
    require(len(groups) == 1, "expected one Adam parameter group")
    group = groups[0]
    require(group["lr"] == 1e-4 and tuple(group["betas"]) == (.9, .999)
            and group["eps"] == 1e-8 and group["weight_decay"] == 0, "Adam configuration mismatch")
    require(len(optimizer.state) == len(list(model.parameters())) == 6, "missing optimizer states")
    steps = []
    for parameter, moments in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            require(moments[key].shape == parameter.shape and torch.isfinite(moments[key]).all(), "invalid Adam moment")
        require((moments["exp_avg_sq"] >= 0).all(), "negative Adam second moment")
        require(moments["exp_avg_sq"].sum() > 0, "empty Adam second moment")
        steps.append(float(moments["step"]))
    require(steps == [672. * epoch] * 6, "Adam step count does not match training split")
    schedule = scheduler.state_dict()
    require(schedule == state["scheduler_state"] and schedule["last_epoch"] == epoch
            and schedule["_step_count"] == epoch + 1 and schedule["_last_lr"] == [1e-4]
            and schedule["base_lrs"] == [1e-4], "scheduler state mismatch")
    python_rng = random.Random()
    python_rng.setstate(saved["python_rng"])
    numpy_rng = np.random.RandomState()
    n = saved["numpy_rng"]
    numpy_rng.set_state((n["kind"], np.asarray(n["keys"], np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    for key in ("torch_rng", "loader_rng"):
        generator = torch.Generator()
        generator.set_state(saved[key])
        first = torch.rand(4, generator=generator)
        generator.set_state(saved[key])
        require(torch.equal(first, torch.rand(4, generator=generator)), "CPU RNG replay mismatch")
    require(torch.equal(saved["loader_rng"], expected_loader_state), "DataLoader RNG differs from expected epoch shuffle")
    require(len(saved["cuda_rng"]) == 1, "expected single-GPU RNG state")
    cuda = saved["cuda_rng"][0]
    require(cuda.dtype == torch.uint8 and cuda.numel() == 16, "invalid recorded CUDA Philox state")
    cuda_bytes = cuda.numpy().tobytes()
    require(int.from_bytes(cuda_bytes[:8], "little") == 69, "CUDA seed mismatch")
    return model.eval(), dict(epoch=epoch, model_name=state["model_name"], parameters=sum(p.numel() for p in model.parameters()),
        finite_model_and_moments=True, optimizer_steps=steps, scheduler=schedule,
        history_matches=True, python_numpy_cpu_rng_loadable=True, loader_rng_matches_seeded_epoch=True,
        cpu_rng_sha256=tensor_hash(saved["torch_rng"]), loader_rng_sha256=tensor_hash(saved["loader_rng"]),
        cuda_rng=dict(bytes=16, seed=69, offset=int.from_bytes(cuda_bytes[8:], "little"),
                      sha256=tensor_hash(cuda), structurally_valid=True, restored_on_cuda=False))


def comparison_metrics(bicubic, cnn):
    return {metric: dict(bicubic=bicubic[metric], cnn=cnn[metric],
            cnn_minus_bicubic=cnn[metric] - bicubic[metric],
            reduction_percent=100 * (bicubic[metric] - cnn[metric]) / bicubic[metric])
            for metric in ("rmse_m", "mae_m")}


def verify_validation_payloads(dataset_archive, records, per_frame):
    """Compare evaluated files to training ZIP bytes without reading test entries."""
    require(len(records) == len(per_frame), "validation provenance count mismatch")
    verified = 0
    with zipfile.ZipFile(dataset_archive) as source:
        for row, frame in zip(records, per_frame):
            require(row["split"] == "validation" and row["official_split"] == "train" and row["id"] == frame["id"], "unexpected provenance frame")
            for key in ("rgb", "depth", "mask"):
                digest = hashlib.sha256(source.read(row[key])).hexdigest()
                require(digest == frame["source_files_sha256"][key], "evaluated file differs from training archive")
                verified += 1
    return dict(validation_frames=len(per_frame), matched_files=verified,
                rgb_depth_mask_byte_identical_to_training_archive=True, test_entries_decoded=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/nyu/converted/manifest_development.json"))
    parser.add_argument("--source-archive", type=Path, default=Path("artifacts/colab/trident_source.zip"))
    parser.add_argument("--dataset-archive", type=Path, default=Path("artifacts/colab/nyu_converted.zip"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot_dense4_audit"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/pilot_dense4"))
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.report_dir.exists(), "audit destinations must be new")
    torch.set_num_threads(2)
    protected = sorted(Path("configs").rglob("*.json")) + sorted(Path("trident_depth/models").glob("*.py"))
    protected += sorted(Path("artifacts/nyu/validation_baselines").rglob("results.json"))
    protected_hashes = {str(p): sha256(p) for p in protected}
    prefix = "runs/pilot_dense4_lightweight_seed69/"
    with zipfile.ZipFile(args.archive) as archive:
        require(archive.testzip() is None, "archive CRC failed")
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate archive entries")
        run = json.loads(archive.read(prefix + "run.json"))
        history = [json.loads(line) for line in archive.read(prefix + "history.jsonl").decode().splitlines() if line]
        environment = json.loads(archive.read("runs/pilot_dense4_lightweight_seed69_environment.json"))
        best_pointer = json.loads(archive.read(prefix + "best_checkpoint.json"))
        checkpoints = {e: torch.load(io.BytesIO(archive.read(prefix + f"epoch_{e:03d}.pt")), map_location="cpu", weights_only=True) for e in (1, 2)}
        archive_payload = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    require([row["epoch"] for row in history] == [1, 2], "expected exactly two epochs")
    cfg, pipeline_dict = run["config"]["experiment"], run["config"]["pipeline"]
    experiment_path = Path("configs/phase5/pilot_dense4_lightweight.json")
    pipeline_path = experiment_path.parent / cfg["pipeline_config"]
    require(json.loads(experiment_path.read_text()) == cfg, "local approved experiment mismatch")
    pipeline = PipelineConfig.load(pipeline_path)
    require(asdict(pipeline) == pipeline_dict and pipeline.protocol == "dense_sr" and pipeline.scale == 4, "pipeline mismatch")
    require(cfg["initializer"] == "bicubic" and cfg["batch_size"] == 1 and cfg["epochs"] == 2 and cfg["seed"] == 69, "unexpected pilot settings")
    require(sha256(experiment_path) == environment["experiment_sha256"] and sha256(pipeline_path) == environment["pipeline_sha256"], "configuration SHA256 mismatch")
    require(sha256(args.source_archive) == environment["source_id"], "source archive SHA256 mismatch")
    require(sha256(args.dataset_archive) == environment["dataset_archive_sha256"], "dataset archive SHA256 mismatch")
    with zipfile.ZipFile(args.source_archive) as source:
        archived_modules = [n for n in source.namelist() if n.startswith("trident_depth/") and n.endswith(".py")]
        require(all(Path(n).read_bytes() == source.read(n) for n in archived_modules), "local code differs from training source")
    split_audit = audit_pilot_dataset(args.manifest)
    require(split_audit == run["split_audit"] and split_audit["manifest_sha256"] == run["split_manifest_sha256"], "split provenance mismatch")
    require(environment["environment"]["torch"] == run["torch_version"] and environment["environment"]["gpu"] == run["gpu"], "environment metadata mismatch")
    for row in history:
        for phase in ("train", "validation"):
            terms = row[phase + "_components"]
            require(terms["uncertainty"] == 0, "unexpected uncertainty term")
            total = sum(cfg["loss_weights"][key] * value for key, value in terms.items())
            require(math.isclose(total, row[phase + "_loss"], abs_tol=1e-7), "weighted loss/history mismatch")
    generator = torch.Generator().manual_seed(69)
    simulated_loader = DataLoader(range(672), batch_size=1, shuffle=True, num_workers=0, generator=generator)
    models, audits = {}, []
    for epoch in (1, 2):
        for _ in simulated_loader:
            pass  # Reproduce shuffle RNG only; no training pixels or optimizer steps.
        models[epoch], audit = audit_checkpoint(checkpoints[epoch], epoch, run, history, generator.get_state())
        audit["checkpoint_sha256"] = hashlib.sha256(archive_payload[prefix + f"epoch_{epoch:03d}.pt"]).hexdigest()
        audits.append(audit)
    require(any(not torch.equal(checkpoints[1]["model_state"][key], value) for key, value in checkpoints[2]["model_state"].items()), "weights did not change between epochs")
    best = min(history, key=lambda row: row["validation_rmse_m"])
    require(best_pointer == dict(epoch=best["epoch"], checkpoint=f"epoch_{best['epoch']:03d}.pt", validation_rmse_m=best["validation_rmse_m"]), "best pointer/history mismatch")
    dataset = TorchDepthDataset(args.manifest, pipeline, "validation", initializer="bicubic")
    require(len(dataset) == 123 and all(r["official_split"] == "train" for r in dataset.source.records), "validation selection changed")
    totals = {key: dict(squared=0., absolute=0., pixels=0, losses=[]) for key in ("bicubic", "cnn_epoch_1", "cnn_epoch_2")}
    per_frame = []
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
            target, mask = batch["target"], batch["target_valid"]
            require(tuple(target.shape) == (1, 1, 480, 640) and tuple(batch["observations"].shape) == (1, 1, 120, 160), "unexpected frame dimensions")
            row = dataset.source.records[index]
            item = dict(id=row["id"], scene_id=row["scene_id"], shape=[480, 640], valid_pixels=int(mask.sum()),
                        mask_sha256=tensor_hash(mask), source_files_sha256={key:sha256(args.manifest.parent / row[key]) for key in ("rgb", "depth", "mask")})
            predictions = {"bicubic": batch["initial"]}
            for epoch, model in models.items():
                loss, terms, predictions[f"cnn_epoch_{epoch}"] = model_loss(model, batch, pipeline, torch.device("cpu"), weights=cfg["loss_weights"], return_prediction=True)
                require(float(terms["uncertainty"]) == 0 and torch.isfinite(loss), "invalid inference loss")
                totals[f"cnn_epoch_{epoch}"]["losses"].append(float(loss))
            for key, prediction in predictions.items():
                squared, absolute, count = validation_errors(prediction, target, mask)
                total = totals[key]
                total["squared"] += squared
                total["absolute"] += absolute
                total["pixels"] += count
                item[key] = dict(sum_squared_error_m2=squared, sum_absolute_error_m=absolute, pixels=count,
                                 rmse_m=math.sqrt(squared/count), mae_m=absolute/count)
            per_frame.append(item)
            if (index + 1) % 20 == 0:
                print(f"Evaluated {index+1}/123 validation frames", flush=True)
    payload_identity = verify_validation_payloads(args.dataset_archive, dataset.source.records, per_frame)
    aggregate = {key:dict(rmse_m=math.sqrt(value["squared"]/value["pixels"]), mae_m=value["absolute"]/value["pixels"],
                         valid_pixels=value["pixels"], frames=len(per_frame)) for key, value in totals.items()}
    replay = []
    for epoch in (1, 2):
        actual = aggregate[f"cnn_epoch_{epoch}"]
        recorded = history[epoch-1]
        differences = {metric: actual[metric] - recorded["validation_" + metric] for metric in ("rmse_m", "mae_m")}
        loss_delta = float(np.mean(totals[f"cnn_epoch_{epoch}"]["losses"])) - recorded["validation_loss"]
        require(all(abs(d) <= 1e-6 for d in differences.values()) and abs(loss_delta) <= 1e-6, "CPU replay differs beyond 1e-6 tolerance")
        replay.append(dict(epoch=epoch, cpu_minus_recorded=differences, loss_delta=loss_delta, tolerance=1e-6, passed=True))
    require(len({p["scene_id"] for p in per_frame}) == 30, "validation scene count mismatch")
    require(all(sha256(p) == digest for p, digest in protected_hashes.items()), "protected file changed")
    report = dict(status="audited; no new training", archive_sha256=sha256(args.archive),
        training_environment=environment, run=run, history=history, best_checkpoint=best_pointer,
        checkpoint_audits=audits, local_source_matches_training_archive=True, split_audit=split_audit,
        validation_payload_identity=payload_identity, audit_module_sha256=sha256(Path(__file__)),
        evaluation=dict(split="validation", frames=123, scenes=30, full_frame=True, crop=None,
            depth_units="metres", input_shape=[120,160], target_shape=[480,640],
            mask="identical target_valid from TorchDepthDataset for every method", aggregation="global valid-pixel double-precision error sums",
            metrics_function="trident_depth.training.validation.validation_errors", protocol=pipeline_dict,
            local_environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, pillow=PIL.__version__, device="cpu")),
        aggregate=aggregate, cpu_replay=replay,
        comparison_cpu=comparison_metrics(aggregate["bicubic"], aggregate[f"cnn_epoch_{best['epoch']}"]),
        frame_comparison={metric:dict(cnn_better=sum(f[f"cnn_epoch_{best['epoch']}"][metric] < f["bicubic"][metric] for f in per_frame),
                                     cnn_worse=sum(f[f"cnn_epoch_{best['epoch']}"][metric] > f["bicubic"][metric] for f in per_frame),
                                     equal=sum(f[f"cnn_epoch_{best['epoch']}"][metric] == f["bicubic"][metric] for f in per_frame)) for metric in ("rmse_m", "mae_m")},
        comparison_recorded=comparison_metrics(aggregate["bicubic"], {m:best["validation_"+m] for m in ("rmse_m", "mae_m")}),
        uncertainty=dict(expected_zero=True, reason="LightweightCNN returns depth and residual only; no log_scale head. combined_loss uses pred.sum()*0. The 0.05 weight therefore contributes zero; this run learns no uncertainty calibration."),
        limitations=["CPU replay uses different Python/PyTorch/NumPy/Pillow versions from Colab; numerical agreement is checked to 1e-6 m, not bitwise equivalence.",
                     "CUDA Philox bytes and seed checked structurally; GPU RNG restoration and exact resumed CUDA execution were not performed on this CPU host.",
                     "Validation evidence only; official test pixels were never loaded. No optimizer steps were performed during this audit."],
        protected_files_sha256=protected_hashes, protected_files_unchanged=True)
    args.output.mkdir(parents=True)
    # Write only known imported files into our new destination, never arbitrary ZIP paths.
    for name, payload in archive_payload.items():
        if name.startswith(prefix):
            (args.output / Path(name).name).write_bytes(payload)
    (args.output / "training_environment.freeze.txt").write_bytes(archive_payload["runs/pilot_dense4_lightweight_seed69_environment.freeze.txt"])
    args.report_dir.mkdir(parents=True)
    (args.report_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.report_dir / "per_frame.json").write_text(json.dumps(per_frame, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Dense 4x baseline CNN pilot audit", "", "No new training. Official test pixels were not loaded. Existing classical benchmark results were preserved.", "", "| Method | Validation RMSE (m) | Validation MAE (m) |", "| --- | ---: | ---: |"]
    for key, metrics in aggregate.items():
        lines.append(f"| {key} (CPU replay) | {metrics['rmse_m']:.10f} | {metrics['mae_m']:.10f} |")
    lines += ["", f"Recorded Colab best: epoch {best['epoch']}, RMSE {best['validation_rmse_m']:.10f} m, MAE {best['validation_mae_m']:.10f} m.", ""]
    for metric, values in report["comparison_cpu"].items():
        lines.append(f"- {metric}: CNN minus bicubic {values['cnn_minus_bicubic']:.10f} m; reduction {values['reduction_percent']:.4f}%.")
    lines += ["", "Both checkpoints strictly load 11,873 model parameters, six Adam moment pairs and a constant scheduler. Adam steps are 672/1,344; scheduler epochs are 1/2. Saved history exactly matches the external log and best-epoch selection. Python, NumPy and CPU RNG states load; DataLoader RNG matches a seeded replay of each epoch's shuffle. CUDA state has the expected 16-byte layout and seed 69, but was not restored on a GPU.", "", "Uncertainty loss is expected to be zero: the baseline has no log_scale head. The loss explicitly supplies a connected zero for that optional term. A nonzero configured uncertainty weight does not create an uncertainty head or imply learned uncertainty.", "", "Evaluation uses exactly 123 scene-safe development validation frames (30 scenes), full 480x640, area-mean 4x degradation, identical target_valid masks, and the training runner's global valid-pixel double-precision RMSE/MAE. Bicubic is the unchanged initializer computation, without CNN refinement. The earlier eight-frame classical summary is not used as the comparator.", "", "CPU replay agrees with both recorded validation epochs within 1e-6 m and validation loss within 1e-6. CPU software versions differ from Colab, so this is numerical verification rather than bitwise reproduction. CUDA resume remains unverified locally.", "", "`report.json` contains full training/evaluation configurations, history, environment, source/dataset/config/checkpoint hashes, audit details and protected-file hashes. `per_frame.json` contains exact IDs, scenes, file/mask hashes and paired error sums. Imported checkpoints and original logs are preserved under `" + args.output.as_posix() + "`.", ""]
    (args.report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.report_dir / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\nAll 369 evaluated RGB/depth/mask files match the checksummed training dataset archive byte-for-byte; test entries were not decoded.\n")
        stream.write("\nThe small RMSE reduction accompanies a MAE regression; this pilot does not show consistent improvement across both metrics. Further training requires approval.\n")
    print(json.dumps(dict(aggregate=aggregate, comparison=report["comparison_cpu"], report=str(args.report_dir / "report.md")), indent=2))


if __name__ == "__main__":
    main()
