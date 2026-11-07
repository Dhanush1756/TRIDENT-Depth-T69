"""Read-only epoch-12 audit. Hash the checkpoint BEFORE deserializing it."""
import argparse
import json
import math
from pathlib import Path
import torch
from .checkpoint import load
from .contract import read_config,source_hashes
from ..hybrid_v1_dense4 import B0_SHA256
from ..training.colab import sha256,NYU_MANIFEST_SHA256
from ..training.corrective_data import REFERENCE_SHA256

EPOCH12_SHA256="09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd"
TRAINING_LOG_SHA256="ee444bba51413d23e0261cfa45f893faea6dd92f505b7c5823789ede228d646c"


def log_rows(path):
    if sha256(path)!=TRAINING_LOG_SHA256:
        raise ValueError("original training console SHA256 mismatch")
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
          if line.startswith('{"epoch":')]
    if [r["epoch"] for r in rows]!=list(range(1,13)):
        raise ValueError("requires original twelve completed epochs")
    return rows


def inspect(checkpoint,training_log,config,reference):
    checkpoint=Path(checkpoint)
    if checkpoint.name!="epoch_012.pt" or sha256(checkpoint)!=EPOCH12_SHA256:
        raise ValueError("exact epoch_012.pt SHA256 mismatch; checkpoint was NOT loaded")
    cfg=read_config(config)
    if sha256(reference)!=REFERENCE_SHA256:
        raise ValueError("original validation reference SHA256 mismatch")
    history=log_rows(training_log)
    run_dir=checkpoint.parent
    for name in ("run.json","epoch_012_metrics.json"):
        if not (run_dir/name).is_file():
            raise FileNotFoundError("required original training artifact: "+str(run_dir/name))
    model,payload=load(checkpoint,"cpu")
    required={"optimizer_state","scheduler_state","scaler_state","training_state","provenance"}
    if not required<=payload.keys():
        raise ValueError("checkpoint missing complete training state")
    state,provenance=payload["training_state"],payload["provenance"]
    if state.get("epoch")!=12 or state.get("best_epoch")!=12 or state.get("history")!=history:
        raise ValueError("checkpoint epoch/history/best epoch differs from original console")
    if min(history,key=lambda r:r["selection_score_m"])["epoch"]!=12 or state["best_rmse"]!=history[-1]["selection_score_m"]:
        raise ValueError("epoch 12 is not the recorded validation-selected checkpoint")
    if (provenance.get("config")!=cfg or provenance.get("config_sha256")!=sha256(config)
            or provenance.get("manifest_sha256")!=NYU_MANIFEST_SHA256
            or provenance.get("initial_checkpoint_sha256")!=B0_SHA256
            or provenance.get("model_identity")!="trident_extended_v1"
            or provenance.get("precision")!="fp32" or provenance.get("amp_enabled") is not False):
        raise ValueError("training configuration/lineage mismatch")
    current=source_hashes()
    original=provenance.get("source_hashes",{})
    if not original or any(current.get(k)!=v for k,v in original.items()):
        raise ValueError("an original training source file differs; no implementation substitution allowed")
    if json.loads((run_dir/"run.json").read_text())!=provenance:
        raise ValueError("run.json differs from checkpoint provenance")
    metrics=json.loads((run_dir/"epoch_012_metrics.json").read_text())
    if metrics["summary"]!=history[-1]:
        raise ValueError("saved epoch metrics differ from checkpoint history")
    ids=[r["id"] for r in json.loads(Path(reference).read_text())]
    for task in cfg["tasks"]:
        name=task["name"]
        rows=[r for r in metrics["validation_frames"] if r["task"]==name]
        if [r["id"] for r in rows]!=ids or len(rows)!=123:
            raise ValueError("training validation IDs/order mismatch")
        n=sum(r["valid_pixels"] for r in rows)
        rmse=math.sqrt(sum(r["squared_error_m2"] for r in rows)/n)
        mae=sum(r["absolute_error_m"] for r in rows)/n
        recorded=history[-1]["validation"][name]
        if abs(rmse-recorded["rmse_m"])>1e-12 or abs(mae-recorded["mae_m"])>1e-12:
            raise ValueError("per-frame error sums do not reproduce training aggregates")
    if len(metrics["validation_frames"])!=861:
        raise ValueError("requires all 861 task/frame pairs in original epoch metrics")
    for row in history:
        if (row["optimizer_steps"]!=672 or set(row["training_task_counts"].values())!={96}
                or row["split_audit"]["official_test_sample_files_opened"]!=0
                or row["split_audit"]["forbidden_access_attempts"]):
            raise ValueError("original training step/split audit mismatch")
    for key,value in payload["model_state"].items():
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite checkpoint tensor: "+key)
    rng={"python_rng","numpy_rng","torch_rng","loader_rng","cuda_rng"}
    if not rng<=state.keys() or payload["scheduler_state"].get("last_epoch")!=12:
        raise ValueError("missing RNG or inconsistent scheduler epoch")
    if not payload["optimizer_state"].get("state"):
        raise ValueError("missing Adam state")
    adam=payload["optimizer_state"]["state"]
    if len(adam)!=sum(p.requires_grad for p in model.parameters()):
        raise ValueError("Adam state does not cover all trainable tensors")
    for item in adam.values():
        if float(item["step"])!=12*672 or any(not torch.isfinite(item[k]).all() for k in ("exp_avg","exp_avg_sq")):
            raise ValueError("Adam step or moment state is inconsistent/nonfinite")
    if sum(p.numel() for p in model.parameters())!=473059:
        raise ValueError("unexpected model parameter count")
    report=dict(status="PASS",checkpoint_sha256=EPOCH12_SHA256,training_log_sha256=TRAINING_LOG_SHA256,
        original_training_sources_verified=len(original),additional_evaluation_sources=len(set(current)-set(original)),
        epoch=12,best_epoch=12,parameters=sum(p.numel() for p in model.parameters()),
        validation=history[-1]["validation"],selection_score_m=state["best_rmse"],
        complete_model_optimizer_scheduler_scaler_rng=True,training_epochs=12,
        checkpoint_model_tensors_finite=True,official_test_sample_files_opened=0,
        note="Historical split audit is checked; no dataset sample is opened by this checkpoint audit")
    return model.eval(),payload,report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--training-log",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/extended_v1/train.json"))
    p.add_argument("--reference",type=Path,default=Path("docs/pilot_dense4/per_frame.json"))
    args=p.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.checkpoint.resolve().parent):
        p.error("audit must use a fresh path outside the original training run")
    _,_,report=inspect(args.checkpoint,args.training_log,args.config,args.reference)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
