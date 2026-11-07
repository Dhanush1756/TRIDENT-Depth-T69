"""Fail-closed configuration identity and source provenance."""
import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path
import torch
from .data import pipeline
from ..training.colab import sha256


def read_config(path):
    cfg=json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["version"]!="trident_extended_v1" or cfg["precision"]!="fp32":
        raise ValueError("v1 uses FP32 until separate AMP numerical qualification")
    if (cfg["seed"],cfg["batch_size"],cfg["learning_rate"])!=(69,1,1e-4):
        raise ValueError("unexpected initial extended training contract")
    if type(cfg["epochs"]) is not int or not 1<=cfg["epochs"]<=12:
        raise ValueError("initial run budget is 1..12 epochs")
    if (cfg["optimizer"],cfg["scheduler"],cfg["weight_decay"],cfg["gradient_clip_norm"],
            cfg["initial_weights"],cfg["selection"],cfg["training_schedule"],cfg["initializer"]) != (
            "Adam","constant",0.0,5.0,"verified_B0_epoch002","equal_task_mean_validation_rmse_m",
            "one_task_per_training_frame_rotated_each_epoch","bicubic_then_nearest_hole_fill"):
        raise ValueError("unsupported extended optimizer/schedule/initialization contract")
    tasks=cfg["tasks"]
    if [t["name"] for t in tasks]!=["dense2","dense4","dense8","sparse","noisy","incomplete","sparse_noisy_holes"]:
        raise ValueError("all seven mandatory training tasks must be explicit")
    for task in tasks:
        p=pipeline(task)
        if task["name"].startswith("dense"):
            if p.protocol!="dense_sr" or p.scale!=int(task["name"][-1]):
                raise ValueError("dense task factor mismatch")
        elif p.protocol!="sparse_completion" or p.scale!=4:
            raise ValueError("completion uses canonical conditioning scale 4, spatial scale 1")
    weights=cfg["loss_weights"]
    if set(weights)!={"depth","gradient","edge","measurement","boundary","false_edge"} or any(
            not isinstance(v,(int,float)) or not __import__('math').isfinite(v) or v<0 for v in weights.values()):
        raise ValueError("invalid loss weights")
    return cfg


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted((root/"trident_depth").rglob("*.py"))}


def fingerprint(hashes):
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def environment_identity():
    return dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                packages={name:importlib.metadata.version(name) for name in
                          ("numpy","scipy","Pillow","opencv-python-headless","h5py")})
