"""Complete epoch-boundary state under an isolated extended-model format."""
import os
from pathlib import Path
import tempfile
import torch
from .model import ExtendedTRIDENT
from ..training.state import restore_state


def save(path,model,optimizer,scheduler,scaler,state,provenance):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(format_version="trident_extended_v1",model_name=model.identity,
                 model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                 scheduler_state=scheduler.state_dict(),scaler_state=scaler.state_dict(),
                 training_state=state,provenance=provenance)
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix=".part")
    os.close(fd)
    try:
        torch.save(payload,temporary)
        # Publish only a completely serialized file. Single-writer run directories
        # are required; the runner rejects every already-existing epoch filename.
        if path.exists():
            raise FileExistsError(path)
        Path(temporary).rename(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load(path,device="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("format_version")!="trident_extended_v1" or payload.get("model_name")!=ExtendedTRIDENT.identity:
        raise ValueError("requires separately versioned extended checkpoint")
    model=ExtendedTRIDENT().to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    return model,payload


def resume(path,model,optimizer,scheduler,scaler,generator,provenance):
    _,payload=load(path)
    if payload["provenance"]!=provenance:
        raise ValueError("resume source/config/data/environment identity mismatch")
    required={"optimizer_state","scheduler_state","scaler_state","training_state"}
    if not required<=payload.keys():
        raise ValueError("incomplete resumable checkpoint")
    model.load_state_dict(payload["model_state"],strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    restore_state(payload["training_state"],generator)
    return payload["training_state"]
