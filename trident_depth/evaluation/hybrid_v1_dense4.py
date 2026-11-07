"""Explicitly gated validation-only B0 hybrid comparison; never trains or selects data."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import importlib.metadata
import numpy as np
import torch
from torch.utils.data import default_collate
from ..config import PipelineConfig
from ..hybrid_v1_dense4 import HybridConfig, HybridV1Dense4, METHODS, B0_SHA256, comparison_outputs, area_mean
from ..training.colab import sha256, NYU_MANIFEST_SHA256
from ..training.corrective_data import REFERENCE_SHA256
from ..training.state import seed_everything
from ..training.portable import select_device
from ..training.validation import validation_errors
from .metrics import evaluate
from .benchmark import process_peak_rss_bytes
from .hybrid_v1_data import ValidationOnly


def write_new(path, value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def source_identity(root):
    hashes = {p.relative_to(root).as_posix():sha256(p) for p in sorted((root/"trident_depth").rglob("*.py"))}
    try:
        revision = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True,stderr=subprocess.DEVNULL).splitlines()
    except (OSError,subprocess.CalledProcessError):
        revision, dirty = None, None
    return dict(git_commit=revision, git_status=dirty, source_files_sha256=hashes,
                identity_note="File hashes identify actual code, including uncommitted additions")


def frame_metrics(prediction, target, mask):
    values = evaluate(prediction,target,mask,data_range_m=10.0,edge_threshold_m=.05,boundary_tolerance_px=1)
    p = torch.from_numpy(np.ascontiguousarray(prediction))[None,None]
    t = torch.from_numpy(np.ascontiguousarray(target))[None,None]
    m = torch.from_numpy(np.ascontiguousarray(mask))[None,None]
    squared, absolute, count = validation_errors(p,t,m)
    return dict(metrics=values,sum_squared_error_m2=squared,sum_absolute_error_m=absolute,valid_pixels=count)


def aggregate_frames(frames):
    output = {}
    for method in METHODS:
        rows = [f["methods"][method] for f in frames]
        count = sum(r["valid_pixels"] for r in rows)
        if count <= 0:
            raise ValueError("no valid evaluation pixels")
        squared = sum(r["sum_squared_error_m2"] for r in rows)
        absolute = sum(r["sum_absolute_error_m"] for r in rows)
        mse = squared/count
        macro = {}
        eligible = {}
        for key in ("psnr_db","ssim","boundary_precision","boundary_recall","boundary_f1","edge_preservation"):
            values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
            macro[key] = float(np.mean(values)) if values else None
            eligible[key] = len(values)
        output[method] = dict(frames=len(rows),valid_pixels=count,global_rmse_m=math.sqrt(mse),
            global_mae_m=absolute/count,global_psnr_db=None if mse==0 else 10*math.log10(100/mse),
            global_psnr_infinite=mse==0,mean_per_frame=macro,eligible_frame_counts=eligible)
    return output


def replay_checks(aggregate):
    expected = {"b0":(.04550963874424752,.012129690590384606,.8178656134105969),
                "bicubic":(.0585272133,.0162030076,None)}
    checks = {}
    for name,(rmse,mae,f1) in expected.items():
        measured = aggregate[name]
        values = {"global_rmse_m":(measured["global_rmse_m"],rmse,1e-6),
                  "global_mae_m":(measured["global_mae_m"],mae,1e-6)}
        if f1 is not None:
            values["mean_per_frame_boundary_f1"]=(measured["mean_per_frame"]["boundary_f1"],f1,1e-5)
        checks[name] = {k:dict(actual=a,expected=b,absolute_difference=abs(a-b),absolute_tolerance=t,
            passed=abs(a-b)<=t) for k,(a,b,t) in values.items()}
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-validation",action="store_true",help="required separate benchmark approval")
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True,help="original B0 epoch_002.pt or supplied B0 ZIP")
    parser.add_argument("--output",type=Path,required=True,help="new directory within a hybrid_v1_dense4 namespace")
    parser.add_argument("--config",type=Path,default=Path("configs/hybrid_v1_dense4.json"))
    parser.add_argument("--reference",type=Path,default=Path("docs/pilot_dense4/per_frame.json"))
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--save-predictions",action="store_true")
    args=parser.parse_args(argv)
    if not args.execute_validation:
        parser.error("benchmark disabled; separate approval and --execute-validation are required")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    output=args.output.resolve()
    if output.exists() or "hybrid_v1_dense4" not in output.parts:
        parser.error("choose a new directory under a separate hybrid_v1_dense4 namespace")
    if output.is_relative_to(args.manifest.resolve().parent):
        parser.error("results must be outside dataset storage")
    if args.checkpoint.suffix.lower() != ".zip" and output.is_relative_to(args.checkpoint.resolve().parent):
        parser.error("results must be outside the original checkpoint run directory")
    dataset_root=args.manifest.resolve().parent
    if args.config.resolve().is_relative_to(dataset_root) or args.checkpoint.resolve().is_relative_to(dataset_root):
        parser.error("config/checkpoint must be outside dataset payload storage")
    config=HybridConfig.load(args.config)
    isolation=ValidationOnly(args.manifest,args.reference)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(2)
    seed_everything(69)
    device=select_device(args.device)
    engine=HybridV1Dense4(args.checkpoint,config,device)
    root=Path(__file__).resolve().parents[2]
    pipeline_path=root/"configs/phase5/dense_sr_4x_pipeline.json"
    pipeline=PipelineConfig.load(pipeline_path)
    if asdict(pipeline)!=engine.metadata["config"]["pipeline"]:
        raise ValueError("current preprocessing differs from the original B0 pipeline")
    source=source_identity(root)
    checkpoint_container_sha=sha256(args.checkpoint)
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        cpu=platform.processor(),logical_cpus=os.cpu_count(),device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,torch_cuda=torch.version.cuda,
        packages={n:importlib.metadata.version(n) for n in ("numpy","Pillow","scipy","opencv-python-headless","h5py")},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    provenance=dict(version=config.version,config=asdict(config),config_sha256=sha256(args.config),
        checkpoint_sha256=B0_SHA256,checkpoint_container_sha256=checkpoint_container_sha,
        checkpoint_metadata=engine.metadata,source=source,environment=environment,
        validation_manifest_sha256=NYU_MANIFEST_SHA256,reference_sha256=REFERENCE_SHA256,
        pipeline=asdict(pipeline),pipeline_sha256=sha256(pipeline_path),
        evaluation=dict(split="validation",frames=123,scale=4,protocol="dense_sr",seed=69,units="metres",
            mask="immutable original GT mask; no predicted-mask intersection",depth_range_m=10.0,
            edge_threshold_m=.05,boundary_tolerance_pixels=1,ssim_window=7),
        aggregation=dict(rmse="sqrt(total squared error / total valid pixels)",mae="total absolute error / total valid pixels",
            global_psnr="10*log10(10m^2/global_MSE)",macro_metrics="arithmetic frame means; counts saved for eligible SSIM/finite PSNR",
            boundary_f1="mean frame F1; not F1 of averaged precision/recall"),
        timing=dict(repeats=args.repeats,warmup=args.warmup,
            scope="standalone complete hybrid pipeline, synchronized GPU; no tracemalloc during timing",
            includes="input tensor bridge/H2D, B0, D2H, classical, geometry, heuristic scores, full hybrid blend/projection",
            excludes="dataset loading/preprocessing, metrics, component comparison blends and file writes",
            per_method_latency="only full pipeline and its components are timed; no claim of independently timed baseline/ablation latencies"),
        confidence="documented heuristic reliability, never trained uncertainty or boundary probability",
        measurement_consistency="bounded 4x4 block area-mean projection; no HR point-observation substitution",
        optimizer_steps=0,benchmark_approval_required=True)
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/"provenance.json",provenance)
    write_new(output/"config.json",asdict(config))
    model_hash={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in engine.model.state_dict().items()}
    frames=[]
    try:
        with isolation.isolated():
            dataset,projected_path=isolation.prepare(output,pipeline)
            if args.save_predictions:
                (output/"predictions").mkdir()
            for index in range(123):
                loaded_at=time.perf_counter()
                batch=default_collate([dataset[index]])
                dataset_seconds=time.perf_counter()-loaded_at
                isolation.verify_batch(batch,index)
                observations=batch["observations"][0,0].numpy()
                valid=batch["input_valid"][0,0].numpy()
                rgb=np.transpose(batch["rgb"][0].numpy(),(1,2,0))
                rgb=np.rint(rgb*255).astype(np.uint8)
                prepared=(batch["rgb"],batch["initial"],batch["observed_support"])
                for _ in range(args.warmup):
                    engine.run(observations,valid,rgb,prepared=prepared)
                if device.type=="cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timings=[]
                for _ in range(args.repeats):
                    result=engine.run(observations,valid,rgb,prepared=prepared)
                    timings.append(result["timing"])
                started=time.perf_counter()
                variants,weights,deltas=comparison_outputs(result["candidates"],result["reliability"]["scores"],observations,valid,config)
                if not np.array_equal(variants["hybrid_v1_dense4"],result["depth"]):
                    raise ValueError("timed hybrid differs from comparison hybrid")
                predictions=dict(result["outputs"],**{k:v for k,v in variants.items() if k!="hybrid_v1_dense4"})
                comparison_seconds=time.perf_counter()-started
                target=batch["target"][0,0].numpy()
                gt_mask=batch["target_valid"][0,0].numpy()
                metrics={name:frame_metrics(predictions[name],target,gt_mask) for name in METHODS}
                drift={name:float(np.abs(area_mean(p)[valid]-observations[valid]).max()) for name,p in predictions.items()}
                if any(drift[name]>2e-6 for name in variants):
                    raise ValueError("projected candidate violates LR area-mean consistency tolerance")
                row=dict(id=batch["sample_id"][0],scene_id=isolation.rows[index]["scene_id"],
                    mask_sha256=isolation.reference[index]["mask_sha256"],methods=metrics,
                    source_files_sha256=isolation.reference[index]["source_files_sha256"],
                    prediction_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in predictions.items()},
                    dataset_preprocessing_seconds=dataset_seconds,comparison_blends_seconds=comparison_seconds,
                    timing_samples=timings,timing_median={k:statistics.median(t[k] for t in timings) for k in timings[0]},
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
                    process_lifetime_peak_rss_bytes=process_peak_rss_bytes(),
                    memory_scope="CUDA pipeline-plus-comparison peak after warmup; CPU process lifetime RSS, not isolated native peak",
                    projection_mean_abs_correction_m=deltas)
                row["max_lr_area_mean_error_m"]=drift
                row["b0_range_clip_mean_abs_correction_m"]=result["b0_range_clip_mean_abs_correction_m"]
                row["b0_range_clip_fraction"]=result["b0_range_clip_fraction"]
                frames.append(row)
                if args.save_predictions:
                    path=output/"predictions"/(row["id"]+".npz")
                    with path.open("xb") as stream:
                        np.savez_compressed(stream,**predictions,ground_truth=target,ground_truth_valid=gt_mask,rgb=rgb,
                            input_depth=observations,input_valid=valid,**{name+"_weights":w for name,w in weights.items()})
                print(f"Validation {index+1}/123: {row['id']}",flush=True)
            isolation_report=isolation.audit()
            if sha256(args.checkpoint)!=checkpoint_container_sha:
                raise ValueError("checkpoint container changed during evaluation")
            if any(hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()!=model_hash[k] for k,v in engine.model.state_dict().items()):
                raise ValueError("B0 model tensors changed")
            if sha256(args.manifest)!=NYU_MANIFEST_SHA256 or sha256(args.reference)!=REFERENCE_SHA256:
                raise ValueError("original manifest/reference changed")
            if source_identity(root)["source_files_sha256"]!=source["source_files_sha256"]:
                raise ValueError("source changed during evaluation")
            aggregate=aggregate_frames(frames)
            checks=replay_checks(aggregate)
            passed=all(v["passed"] for method in checks.values() for v in method.values())
            report=dict(status="COMPLETE" if passed else "INVALID_BASELINE_REPLAY",provenance=provenance,
                aggregate=aggregate,baseline_replay=checks,isolation=isolation_report,
                timing_summary={k:statistics.median(r["timing_median"][k] for r in frames) for k in frames[0]["timing_median"]},
                validation_projection_sha256=sha256(projected_path),
                interpretation="No outcome is asserted by the implementation; examine all metrics and the projection-only control. Heuristics are not calibrated confidence.")
            write_new(output/"per_frame.json",frames)
            write_new(output/"report.json",report)
            write_new(output/"access_audit.json",dict(summary=isolation_report,dataset_open_ledger=isolation.opened))
            lines=["# hybrid_v1_dense4 validation", "", "Status: "+report["status"], "",
                "123 original validation frames only; no training or official test sample access.", "",
                "| Method | Global RMSE m | Global MAE m | Global PSNR dB | Mean-frame SSIM | Mean-frame boundary P | R | F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for name,a in aggregate.items():
                m=a["mean_per_frame"]
                lines.append(f"| {name} | {a['global_rmse_m']:.10f} | {a['global_mae_m']:.10f} | {a['global_psnr_db']} | {m['ssim']} | {m['boundary_precision']} | {m['boundary_recall']} | {m['boundary_f1']} |")
            lines += ["", "Timing medians (seconds): `"+json.dumps(report["timing_summary"])+"`", "",
                "Total timing is the standalone full hybrid pipeline. Component ablation blends, metrics, source I/O and reports are excluded.",
                "Dataset preprocessing time is reported separately per frame. Memory scope and runtime are recorded in JSON.", "",
                "All blends use area consistency; b0_consistency_only exposes that effect. B0/bicubic references are unmodified.",
                "Global PSNR uses global MSE; mean frame PSNR is separately available in JSON. SSIM and boundary values are frame means.",
                "Reliability is heuristic. Do not infer calibrated uncertainty or improvements on other scales/protocols."]
            with (output/"report.md").open("x",encoding="utf-8") as stream:
                stream.write("\n".join(lines)+"\n")
            if not passed:
                raise ValueError("B0/bicubic replay outside predefined tolerances; inspect saved INVALID_BASELINE_REPLAY report")
    except Exception as exc:
        path=output/"failure.json"
        if not path.exists():
            write_new(path,dict(status="FAILED",error=type(exc).__name__,message=str(exc),completed_frames=len(frames),
                forbidden_access_attempts=isolation.denied,optimizer_steps=0))
        raise


if __name__=="__main__":
    main()
