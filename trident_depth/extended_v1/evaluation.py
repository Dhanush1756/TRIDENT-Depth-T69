"""Frozen seven-task, 123-frame validation benchmark; inference only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import default_collate
from .evaluation_audit import inspect,EPOCH12_SHA256
from .evaluation_diagnostics import error_sums,summary,regions,correction_diagnostics,texture_diagnostic,panel
from .data import ProtocolDataset,tensor_sample,pipeline
from .contract import environment_identity,source_hashes
from .inference import ExtendedInference
from ..data.io import load_sample
from ..data.degradation import degrade
from ..classical.interpolation import interpolate
from ..classical.hole_filling import fill_holes
from ..classical.pipeline import reconstruct
from ..evaluation.metrics import evaluate
from ..evaluation.benchmark import process_peak_rss_bytes
from ..evaluation.hybrid_v1_data import ValidationOnly
from ..hybrid_v1_dense4 import load_b0,B0_SHA256
from ..training.state import seed_everything
from ..training.colab import sha256
from ..training.corrective_data import audit_validation_reference

BASELINES=("nearest_fill","bilinear_fill","bicubic_fill","joint_bilateral","guided","classical")


def write_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)


def methods_for(task):
    return BASELINES+("extended","extended_hybrid")+(("b0",) if task["name"]=="dense4" else ())


def neural_prediction(model,sample,task,device):
    # Call the ORIGINAL tensor/preprocessing function. Only RGB, initializer,
    # observed support and protocol reach the model, never target/mask tensors.
    b=tensor_sample(sample,task,sample.target.sample_id)
    to=lambda key:b[key][None].to(device)
    c=task["pipeline"]
    with torch.inference_mode():
        out=model(to("rgb"),to("initial"),to("observed_support"),c["scale"],c["protocol"])
    return out["depth"].float().cpu().numpy()[0,0].copy()


def baseline(method,sample):
    if method.endswith("_fill"):
        d,m=interpolate(sample.input_depth,sample.input_valid,sample.target.depth.shape,method[:-5])
        return fill_holes(d,m)[0]
    return reconstruct(sample.input_depth,sample.input_valid,sample.target.depth.shape,sample.target.rgb,method)[0]


def measure(call,device,warmup,repeats):
    for _ in range(warmup):call()
    if device.type=="cuda":
        torch.cuda.synchronize(device)
        base=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:base=0
    times=[]
    previous=None
    for _ in range(repeats):
        if device.type=="cuda":torch.cuda.synchronize(device)
        start=time.perf_counter();value=np.ascontiguousarray(call(),dtype=np.float32)
        if device.type=="cuda":torch.cuda.synchronize(device)
        times.append(time.perf_counter()-start)
        if not np.isfinite(value).all():raise ValueError("nonfinite prediction")
        if previous is not None and not np.array_equal(previous,value):
            raise ValueError("inference repetitions are not bitwise reproducible")
        previous=value
    memory=dict(resident_allocated_before_bytes=base,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-base if device.type=="cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
        process_lifetime_peak_rss_bytes=process_peak_rss_bytes())
    return previous,dict(seconds=times,median_seconds=statistics.median(times),memory=memory)


def aggregate(rows):
    total={k:sum(r["errors"][k] for r in rows) for k in ("pixels","squared_error_m2","absolute_error_m")}
    result=summary(total)
    mse=total["squared_error_m2"]/total["pixels"]
    result["global_psnr_db"]=10*math.log10(100/mse) if mse else None
    result["global_psnr_infinite"]=mse==0
    result["frames"]=len(rows)
    for k in ("ssim","boundary_precision","boundary_recall","boundary_f1","edge_preservation","psnr_db"):
        values=[r["metrics"][k] for r in rows if r["metrics"][k] is not None]
        result["mean_frame_"+k]=float(np.mean(values)) if values else None
        result[k+"_eligible_frames"]=len(values)
    result["regions"]={name:summary({k:sum(r["regions"][name][k] for r in rows) for k in total}) for name in rows[0]["regions"]}
    result["latency_median_frame_seconds"]=statistics.median(r["timing"]["median_seconds"] for r in rows)
    result["latency_p95_frame_seconds"]=float(np.percentile([r["timing"]["median_seconds"] for r in rows],95))
    result["peak_allocated_bytes"]=max((r["timing"]["memory"]["peak_allocated_bytes"] or 0) for r in rows)
    result["peak_reserved_bytes"]=max((r["timing"]["memory"]["peak_reserved_bytes"] or 0) for r in rows)
    if "correction_vs_bicubic_fill" in rows[0]:
        result["correction_vs_bicubic_fill"]={}
        for region in rows[0]["regions"]:
            active=[r["correction_vs_bicubic_fill"][region] for r in rows if region in r["correction_vs_bicubic_fill"]]
            n=sum(r["pixels"] for r in active)
            result["correction_vs_bicubic_fill"][region]=dict(pixels=n,
                correction_mean_m=sum(r["correction_mean_m"]*r["pixels"] for r in active)/n if n else None,
                correction_abs_mean_m=sum(r["correction_abs_mean_m"]*r["pixels"] for r in active)/n if n else None,
                improved_fraction=sum(r["improved_pixels"] for r in active)/n if n else None,
                worsened_fraction=sum(r["worsened_pixels"] for r in active)/n if n else None)
    return result


def replay_checks(aggregate_by_task,recorded,protocol):
    checks={}
    for name,t in recorded.items():
        actual=aggregate_by_task[name]["extended"]
        checks[name]={}
        for a,b,tol in (("rmse_m","rmse_m",protocol["rmse_mae_replay_tolerance_m"]),
                        ("mae_m","mae_m",protocol["rmse_mae_replay_tolerance_m"]),
                        ("mean_frame_ssim","ssim",protocol["ssim_boundary_replay_tolerance"]),
                        ("mean_frame_boundary_f1","boundary_f1",protocol["ssim_boundary_replay_tolerance"])):
            difference=abs(actual[a]-t[b])
            checks[name][a]=dict(actual=actual[a],expected=t[b],absolute_difference=difference,tolerance=tol,passed=difference<=tol)
    actual=aggregate_by_task["dense4"]["b0"]
    checks["b0_dense4"]={}
    for k,value in (("rmse_m",.04550963874424752),("mae_m",.012129690590384606),("mean_frame_boundary_f1",.8178656134105969)):
        tol=1e-5 if "boundary" in k else 1e-6
        diff=abs(actual[k]-value)
        checks["b0_dense4"][k]=dict(actual=actual[k],expected=value,absolute_difference=diff,tolerance=tol,passed=diff<=tol)
    return checks


def build_panels(output,frames,tasks,representatives,worst_count):
    directory=output/"panels";directory.mkdir()
    selection={}
    for task in tasks:
        name=task["name"];rows=[r for r in frames if r["task"]==name]
        worst=sorted(rows,key=lambda r:(-r["methods"]["extended"]["errors"]["squared_error_m2"]/r["methods"]["extended"]["errors"]["pixels"],r["id"]))[:worst_count]
        selection[name]=dict(representative_ids=representatives,worst_extended_rmse_ids=[r["id"] for r in worst])
        for sample_id in sorted(set(representatives)|{r["id"] for r in worst}):
            with np.load(output/"predictions"/name/(sample_id+".npz"),allow_pickle=False) as z:
                names=["bicubic_fill","joint_bilateral","extended","extended_hybrid"]
                if "b0" in z.files:names.append("b0")
                roles=[]
                if sample_id in representatives:roles.append("representative")
                if sample_id in {r["id"] for r in worst}:roles.append("worst extended RMSE (post hoc)")
                rank=" + ".join(roles)
                panel(directory/f"{name}_{sample_id}.png",z["rgb"],z["observations"],z["input_valid"],z["target"],z["target_valid"],
                      {k:z[k] for k in names},f"{name} / {sample_id} / {rank}; depth 0..10m, error 0..0.5m (saturated)")
    return selection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","training-log","manifest","b0-checkpoint","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("configs/extended_v1/train.json"))
    p.add_argument("--protocol",type=Path,default=Path("configs/extended_v1/validation_epoch012.json"))
    p.add_argument("--reference",type=Path,default=Path("docs/pilot_dense4/per_frame.json"))
    p.add_argument("--execute-validation",action="store_true")
    args=p.parse_args(argv)
    if not args.execute_validation:p.error("evaluation disabled; inspect audit first, then --execute-validation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    output=args.output.resolve()
    if output.exists() or "extended_v1_epoch012" not in output.parts:
        p.error("choose a NEW directory in an extended_v1_epoch012 namespace")
    if output.is_relative_to(args.checkpoint.resolve().parent) or output.is_relative_to(args.manifest.resolve().parent):
        p.error("output must be separate from dataset and original training artifacts")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="Tesla T4":
        p.error("benchmark requires one CUDA Tesla T4; no CPU substitution")
    protocol=json.loads(args.protocol.read_text())
    # Freeze protocol independently of runtime flags; thresholds are not tunable.
    if sha256(args.protocol)!=EVALUATION_PROTOCOL_SHA256:raise ValueError("frozen evaluation protocol SHA256 differs")
    model,payload,audit=inspect(args.checkpoint,args.training_log,args.config,args.reference)
    if environment_identity()!=payload["provenance"]["environment"]:
        raise ValueError("evaluation must use the original Python/Torch/dependency environment")
    cfg=payload["provenance"]["config"];tasks=cfg["tasks"]
    if [t["name"] for t in tasks]!=protocol["tasks"]:raise ValueError("task order mismatch")
    isolation=ValidationOnly(args.manifest,args.reference)
    torch.set_num_threads(2);seed_everything(69)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda")
    model.requires_grad_(False);model=model.to(device).eval()
    b0,_=load_b0(args.b0_checkpoint,device)
    immutable_paths={str(path):sha256(path) for path in (args.checkpoint,args.b0_checkpoint,args.manifest,args.reference,args.config,args.protocol,args.training_log)}
    adapter=ExtendedInference(model,device)
    weights_before={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
    identity=source_hashes()
    output.mkdir(parents=True,exist_ok=False)
    (output/"predictions").mkdir()
    provenance=dict(checkpoint_audit=audit,checkpoint_sha256=EPOCH12_SHA256,b0_checkpoint_sha256=B0_SHA256,
        source_hashes=identity,original_files_sha256=immutable_paths,config_sha256=sha256(args.config),protocol_sha256=sha256(args.protocol),
        environment=environment_identity(),gpu=torch.cuda.get_device_name(0),protocol=protocol,
        timing="per-method end-to-end CPU input preprocessing/H2D/inference/D2H/postprocessing; synchronized CUDA wall-clock; excludes dataset loading/degradation/metrics/storage/panels",
        memory="GPU absolute allocated/reserved peaks include resident models; incremental allocated peak also recorded; CPU RSS is process lifetime, not isolated per-method",
        units="metres",precision="FP32; TF32 disabled",optimizer_steps=0,
        completion_scale="target-grid spatial factor 1; conditioning scale 4 is not upsampling",
        baseline_note="interpolation baselines use nearest valid hole filling; on same-resolution completion these may be identical, not independent gains",
        unsupported_b0={t["name"]:"N/A: original B0 supports dense_sr 4x only" for t in tasks if t["name"]!="dense4"})
    write_new(output/"provenance.json",provenance)
    frames=[]
    representatives=[];seen_scenes=set()
    for r in isolation.rows:
        if r["scene_id"] not in seen_scenes and len(representatives)<protocol["panels_representative_scenes"]:
            representatives.append(r["id"]);seen_scenes.add(r["scene_id"])
    try:
        with isolation.isolated():
            # Existing guard checks symlinks BEFORE hashes and audits only 369
            # validation files. The temporary legacy dense4 loader is NOT iterated.
            isolation.prepare(output,pipeline(tasks[1]))
            view=SimpleNamespace(root=isolation.root,rows={"validation":isolation.rows},enabled=True)
            dataset=ProtocolDataset(view,tasks,"validation")
            with (output/"per_frame.jsonl").open("x",encoding="utf-8") as stream:
                for index in range(len(dataset)):
                    ti,ri=divmod(index,123);task=tasks[ti]
                    row,reference=isolation.rows[ri],isolation.reference[ri]
                    # ORIGINAL loader + degradation + preprocessing used at training validation.
                    batch=dataset[index]
                    if (batch["sample_id"]!=reference["id"] or batch["target"].shape!=(1,480,640)
                            or hashlib.sha256(batch["target_valid"].numpy().tobytes()).hexdigest()!=reference["mask_sha256"]):
                        raise ValueError("validation ID/dimensions/mask changed")
                    if ti==0:isolation.loaded.append(batch["sample_id"])
                    sample=degrade(load_sample(row,isolation.root,min_depth=0,max_depth=10),pipeline(task))
                    if not np.array_equal(batch["observations"].numpy()[0],sample.input_depth):
                        raise ValueError("original degradation is not reproducible")
                    target,valid,rgb=sample.target.depth,sample.target.valid,sample.target.rgb
                    masks=regions(valid,sample.input_valid,task,protocol["distance_bins_px"])
                    methods={};predictions={}
                    for name in methods_for(task):
                        if name in BASELINES:call=lambda n=name:baseline(n,sample)
                        elif name in {"extended","b0"}:call=lambda m=(b0 if name=="b0" else model):neural_prediction(m,sample,task,device)
                        else:
                            c=task["pipeline"]
                            call=lambda:adapter.run(sample.input_depth,sample.input_valid,rgb,scale=c["scale"],protocol=c["protocol"],noisy=c["noise_std_m"]>0)["depth"]
                        pred,timing=measure(call,device,protocol["warmup"],protocol["repeats"])
                        if pred.shape!=target.shape:raise ValueError("prediction dimensions mismatch")
                        predictions[name]=pred
                        methods[name]=dict(errors=error_sums(pred,target,valid),
                            metrics=evaluate(pred,target,valid,data_range_m=10.,edge_threshold_m=.05,boundary_tolerance_px=1),
                            regions={k:error_sums(pred,target,m) for k,m in masks.items()},timing=timing,
                            prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),
                            min_depth_m=float(pred.min()),max_depth_m=float(pred.max()),
                            texture=texture_diagnostic(pred,target,valid,rgb))
                    for name in ("extended","extended_hybrid"):
                        methods[name]["correction_vs_bicubic_fill"]={k:correction_diagnostics(predictions[name],predictions["bicubic_fill"],target,m) for k,m in masks.items() if m.any()}
                    frame=dict(task=task["name"],id=row["id"],scene_id=row["scene_id"],
                        input_shape=list(sample.input_depth.shape),target_shape=list(target.shape),
                        mask_sha256=reference["mask_sha256"],source_files_sha256=reference["source_files_sha256"],
                        observations_sha256=hashlib.sha256(sample.input_depth.tobytes()).hexdigest(),methods=methods)
                    frames.append(frame);stream.write(json.dumps(frame,allow_nan=False)+"\n");stream.flush()
                    directory=output/"predictions"/task["name"];directory.mkdir(exist_ok=True)
                    with (directory/(row["id"]+".npz")).open("xb") as f:
                        np.savez_compressed(f,**predictions,target=target,target_valid=valid,rgb=rgb,
                            observations=sample.input_depth,input_valid=sample.input_valid)
                    print(f"{task['name']} {ri+1}/123 ({index+1}/861)",flush=True)
            access=isolation.audit()
            if audit_validation_reference(args.manifest)!=isolation.reference:raise ValueError("validation data changed")
        aggregate_by_task={t["name"]:{m:aggregate([f["methods"][m] for f in frames if f["task"]==t["name"]]) for m in methods_for(t)} for t in tasks}
        checks=replay_checks(aggregate_by_task,audit["validation"],protocol)
        passed=all(c["passed"] for task in checks.values() for c in task.values())
        selection=build_panels(output,frames,tasks,representatives,protocol["panels_worst_frames_per_task"])
        if (any(sha256(Path(path))!=digest for path,digest in immutable_paths.items()) or source_hashes()!=identity
                or any(hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()!=weights_before[k] for k,v in model.state_dict().items())):
            raise ValueError("checkpoint, model tensors or source changed during inference")
        report=dict(status="COMPLETE" if passed else "INVALID_REPLAY",aggregate=aggregate_by_task,
                    replay_checks=checks,isolation=access,unique_validation_frames=123,task_frame_pairs=861,
                    panels=selection,provenance=provenance)
        write_new(output/"report.json",report)
        write_new(output/"access_audit.json",dict(summary=access,dataset_open_ledger=isolation.opened))
        write_markdown(output,report)
        if not passed:raise ValueError("reproduction tolerances failed; preserve evidence and investigate without tuning")
    except Exception as e:
        write_new(output/"failure.json",dict(status="FAILED",error=type(e).__name__,message=str(e),completed_pairs=len(frames),official_test_access_attempts=isolation.denied))
        raise


def write_markdown(output,report):
    lines=["# Extended epoch-12 validation", "", "Status: "+report["status"],"",
           "123 validation frames per task; no official test samples. PSNR range 10m; SSIM/boundary scores are mean per frame.","",
           "| Task | Method | RMSE m | MAE m | PSNR dB | SSIM | Boundary P | R | F1 | Median latency ms |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for task,methods in report["aggregate"].items():
        for name,r in methods.items():
            values=[r[k] for k in ("rmse_m","mae_m","global_psnr_db","mean_frame_ssim","mean_frame_boundary_precision","mean_frame_boundary_recall","mean_frame_boundary_f1")]
            lines.append("| "+" | ".join([task,name]+[f"{v:.8f}" if v is not None else "N/A" for v in values]+[f"{r['latency_median_frame_seconds']*1000:.3f}"])+" |")
    lines += ["","B0 is N/A outside dense4. Filled interpolation methods can coincide on target-grid completion.","",
              "## Missing-region evidence (not just global scores)","",
              "| Task | Method | Region | Pixels | RMSE m | MAE m |","| --- | --- | --- | ---: | ---: | ---: |"]
    for task in ("incomplete","sparse_noisy_holes"):
        for name in ("bicubic_fill","joint_bilateral","extended","extended_hybrid"):
            for region,r in report["aggregate"][task][name]["regions"].items():
                lines.append(f"| {task} | {name} | {region} | {r['pixels']} | {r['rmse_m']} | {r['mae_m']} |")
    lines += ["","## Interpretation and limitations","",
        "Panels include five scenes chosen by manifest order before inference and three post-hoc worst-RMSE extended frames per task. All frame scores and predictions remain available; no cases are discarded.",
        "Compare central-hole/missing-region errors with filled bicubic and their distance profiles before attributing failure. Clean hybrid point retention can improve global scores outside a hole without solving the hole.",
        "Confirmed implementation facts: central holes remove 25% of image area; initialization fills from nearest valid depth; supervision reaches the GT everywhere valid, but measurement consistency cannot constrain unobserved pixels. The convolutional model has no explicit global geometry module.",
        "Hypotheses requiring evidence: long-distance surface ambiguity, local receptive-field limitations and insufficient diverse hole supervision. These are not proven causes or validated fixes. Fixed synthetic central holes are not a real-sensor completion benchmark.",
        "High whole-image SSIM can conceal severe missing-region errors. Inspect region-weighted sums, correction diagnostics, RGB-texture counts and saturation-labelled error panels. No thresholds were tuned and no optimizer was constructed."]
    for task in ("incomplete","sparse_noisy_holes"):
        a=report["aggregate"][task]["extended"]
        hole,total=a["regions"]["central_hole"],a["regions"]["all"]
        share=hole["squared_error_m2"]/total["squared_error_m2"] if total["squared_error_m2"] else 0.
        initial=report["aggregate"][task]["bicubic_fill"]["regions"]["central_hole"]
        change=a["correction_vs_bicubic_fill"]["central_hole"]
        lines += ["",f"Measured {task}: central-hole pixels are {hole['pixels']/total['pixels']:.2%} of valid pixels and account for {share:.2%} of squared error. Hole RMSE is {hole['rmse_m']:.6f} m versus filled-bicubic {initial['rmse_m']:.6f} m. Corrections improve {change['improved_fraction']:.2%} and worsen {change['worsened_fraction']:.2%} of hole pixels. These are observations, not causal proof."]
    with (output/"report.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")


# Set from the reviewed immutable JSON file; not a self-comparison at runtime.
EVALUATION_PROTOCOL_SHA256="f147821bfe900263ac022b05df0c679e745978d5e6af5b4a66ec8f2ae09fae67"

if __name__=="__main__":main()
