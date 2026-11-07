"""Verified, scene-safe multi-task training; only this CLI performs real updates."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import DevelopmentData, ProtocolDataset
from .model import ExtendedTRIDENT
from .loss import objective
from .contract import read_config,source_hashes,fingerprint,environment_identity
from .checkpoint import save,resume
from ..training.state import seed_everything,capture_state
from ..training.colab import sha256
from ..training.portable import select_device
from ..training.validation import validation_errors
from ..evaluation.metrics import evaluate


def write_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False)


def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def predict(model,batch,task):
    cfg=task["pipeline"]
    return model(batch["rgb"],batch["initial"],batch["observed_support"],cfg["scale"],cfg["protocol"])


def validate(model,loader,tasks,weights,device,reference):
    """No optimizer reference; all seven tasks retain the original 123 masks."""
    model.eval()
    frames=[]
    by_name={t["name"]:t for t in tasks}
    by_id={r["id"]:r for r in reference}
    with torch.inference_mode():
        for raw in loader:
            name=raw["task"][0]
            sample_id=raw["sample_id"][0]
            if sample_id not in by_id:
                raise ValueError("non-validation sample in validation")
            import hashlib
            if hashlib.sha256(raw["target_valid"].numpy().tobytes()).hexdigest()!=by_id[sample_id]["mask_sha256"]:
                raise ValueError("original validation mask changed")
            b=to_device(raw,device)
            out=predict(model,b,by_name[name])
            loss,terms=objective(out,b,by_name[name],weights)
            squared,absolute,count=validation_errors(out["depth"],b["target"],b["target_valid"])
            arrays=[v.detach().cpu().numpy()[0,0] for v in (out["depth"],b["target"],b["target_valid"])]
            metrics=evaluate(*arrays,data_range_m=10.,edge_threshold_m=.05,boundary_tolerance_px=1)
            frames.append(dict(task=name,id=sample_id,loss=float(loss),squared_error_m2=squared,
                               absolute_error_m=absolute,valid_pixels=count,metrics=metrics))
            if len(frames)%123==0:
                print(f"validation {len(frames)}/{123*len(tasks)} task={name}",flush=True)
    aggregate={}
    for task in tasks:
        rows=[r for r in frames if r["task"]==task["name"]]
        if len(rows)!=123 or [r["id"] for r in rows]!=[r["id"] for r in reference]:
            raise ValueError("validation must contain exactly original 123 frames per task")
        n=sum(r["valid_pixels"] for r in rows)
        mse=sum(r["squared_error_m2"] for r in rows)/n
        aggregate[task["name"]]=dict(frames=123,pixels=n,rmse_m=math.sqrt(mse),
            mae_m=sum(r["absolute_error_m"] for r in rows)/n,
            psnr_db=10*math.log10(100/mse) if mse>0 else None,
            ssim=float(np.mean([r["metrics"]["ssim"] for r in rows])),
            boundary_f1=float(np.mean([r["metrics"]["boundary_f1"] for r in rows])),
            loss=float(np.mean([r["loss"] for r in rows])))
    score=float(np.mean([a["rmse_m"] for a in aggregate.values()]))
    return score,aggregate,frames


def verify_preflight(report,cfg_path,checkpoint,data_hashes,device):
    expected=dict(status="PASS",optimizer_steps=0,device=str(device),full_resolution=True,
                  config_sha256=sha256(cfg_path),checkpoint_sha256=sha256(checkpoint),
                  source_hashes=source_hashes(),dataset_fingerprint=fingerprint(data_hashes),
                  torch_version=str(torch.__version__),environment=environment_identity())
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError("preflight does not qualify this exact code/config/data/device/environment")
    if device.type!="cuda" or report.get("gpu")!="Tesla T4" or torch.cuda.get_device_name(device)!="Tesla T4":
        raise ValueError("real training requires qualified Tesla T4; CPU is for tests/preflight")
    if len(report["results"])!=7 or not all(r["finite_gradients"] for r in report["results"]):
        raise ValueError("all seven tasks must pass backward qualification")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path,help="immutable original B0 warm-start")
    p.add_argument("--preflight",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--resume",type=Path)
    p.add_argument("--execute-training",action="store_true")
    args=p.parse_args(argv)
    if not args.execute_training:
        p.error("training requires explicit --execute-training")
    cfg=read_config(args.config)
    output=args.output.resolve()
    if "extended_v1" not in output.parts or output.is_relative_to(args.manifest.resolve().parent):
        p.error("use separate extended_v1 results namespace outside dataset")
    if not args.resume and output.exists():
        p.error("new run requires a fresh output directory")
    if args.resume and (args.resume.resolve().parent!=output or not output.is_dir()):
        p.error("resume checkpoint must belong to this extended run")
    device=select_device("cuda")
    torch.set_num_threads(2)
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=DevelopmentData(args.manifest,Path(__file__).resolve().parents[2]/"docs/pilot_dense4/per_frame.json")
    with data.isolated():
        hashes=data.verify_files()
        qualified=json.loads(args.preflight.read_text())
        verify_preflight(qualified,args.config,args.checkpoint,hashes,device)
        provenance=dict(config=cfg,config_sha256=sha256(args.config),source_hashes=source_hashes(),
            dataset_fingerprint=fingerprint(hashes),manifest_sha256=sha256(args.manifest),
            initial_checkpoint_sha256=sha256(args.checkpoint),torch_version=str(torch.__version__),
            gpu=torch.cuda.get_device_name(device),cuda=torch.version.cuda,environment=environment_identity(),
            split=data.audit(),precision="fp32",amp_enabled=False,
            selection="unweighted mean of seven task-specific global pixel-weighted validation RMSEs",
            model_identity="trident_extended_v1",uncertainty_enabled=False,boundary_supervised=True)
        # Audit count must be stable across resume: all 2385 files already hashed.
        train=ProtocolDataset(data,cfg["tasks"],"train")
        validation=ProtocolDataset(data,cfg["tasks"],"validation")
        generator=torch.Generator().manual_seed(cfg["seed"])
        train_loader=DataLoader(train,batch_size=1,shuffle=True,generator=generator,num_workers=0)
        val_loader=DataLoader(validation,batch_size=1,shuffle=False,num_workers=0)
        model=ExtendedTRIDENT().warm_start(args.checkpoint).to(device)
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        start,best,best_epoch,history=1,float("inf"),0,[]
        if args.resume:
            state=resume(args.resume,model,optimizer,scheduler,scaler,generator,provenance)
            start,best,best_epoch=state["epoch"]+1,state["best_rmse"],state["best_epoch"]
            history=state["history"]
            if start>cfg["epochs"]:
                p.error("approved epoch budget already complete")
            # Do not permit branching over newer evidence in the same run directory.
            if any((output/f"epoch_{e:03d}.pt").exists() for e in range(start,cfg["epochs"]+1)):
                p.error("newer checkpoint evidence exists; resume the latest complete epoch")
        else:
            output.mkdir(parents=True)
            write_new(output/"run.json",provenance)
        by_name={t["name"]:t for t in cfg["tasks"]}
        for epoch in range(start,cfg["epochs"]+1):
            started=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            train.epoch=epoch
            model.train()
            losses=[]
            components={}
            counts={k:0 for k in by_name}
            training_ids=set()
            for raw in train_loader:
                name=raw["task"][0]
                sample_id=raw["sample_id"][0]
                if sample_id in training_ids:
                    raise ValueError("training frame repeated in epoch")
                training_ids.add(sample_id)
                b=to_device(raw,device)
                optimizer.zero_grad(set_to_none=True)
                loss,terms=objective(predict(model,b,by_name[name]),b,by_name[name],cfg["loss_weights"])
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"],error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                counts[name]+=1
                for k,v in terms.items():
                    components.setdefault(k,[]).append(float(v.detach()))
                if len(losses)%32==0:
                    print(f"epoch={epoch} train={len(losses)}/672 loss={losses[-1]:.6f}",flush=True)
            if len(training_ids)!=672 or set(counts.values())!={96}:
                raise ValueError("balanced one-pass training contract failed")
            score,aggregate,frames=validate(model,val_loader,cfg["tasks"],cfg["loss_weights"],device,data.frames)
            if not math.isfinite(score):
                raise ValueError("nonfinite validation selection score")
            if score<best:
                best,best_epoch=score,epoch
            torch.cuda.synchronize()
            row=dict(epoch=epoch,optimizer_steps=672,train_loss=float(np.mean(losses)),
                train_components={k:float(np.mean(v)) for k,v in components.items()},
                training_task_counts=counts,validation=aggregate,selection_score_m=score,
                best_epoch=best_epoch,best_score_m=best,seconds=time.perf_counter()-started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),split_audit=data.audit())
            scheduler.step()
            history.append(row)
            state=capture_state(generator,epoch,best,best_epoch)
            state["history"]=history
            save(output/f"epoch_{epoch:03d}.pt",model,optimizer,scheduler,scaler,state,provenance)
            write_new(output/f"epoch_{epoch:03d}_metrics.json",dict(summary=row,validation_frames=frames))
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
