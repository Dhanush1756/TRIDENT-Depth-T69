"""Tensor/state-only checkpoint I/O with explicit model and training provenance."""
from pathlib import Path
import os
import tempfile
import torch
from .baseline import LightweightCNN
from .residual_cnn import ResidualCNN
from .residual_v2 import ResidualCNNV2

MODELS = {"lightweight_cnn": LightweightCNN, "residual_cnn": ResidualCNN,
          "residual_cnn_v2": ResidualCNNV2}


def model_configuration(model, model_name):
    if model_name == "residual_cnn_v2":
        return model.configuration()
    return dict(upsampling=model.upsampling) if model_name == "residual_cnn" else {}


def save_checkpoint(path, model, model_name, metadata, optimizer=None, training_state=None, scheduler=None):
    if model_name not in MODELS or not isinstance(model, MODELS[model_name]):
        raise ValueError("unknown or mismatched model architecture")
    if isinstance(model, ResidualCNNV2) and model_name != "residual_cnn_v2":
        raise ValueError("versioned model requires its own checkpoint identity")
    if metadata.get("trained") and not all(metadata.get(k) for k in ("dataset", "split_manifest_sha256", "config", "epochs")):
        raise ValueError("trained checkpoints require dataset, split hash, config, and epoch provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    state = dict(format_version=1, model_name=model_name, model_state=model.state_dict(), metadata=metadata)
    state["model_configuration"] = model_configuration(model, model_name)
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        state["training_state"] = training_state
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    descriptor, partial = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, partial)
        Path(partial).rename(path)
    finally:
        Path(partial).unlink(missing_ok=True)


def resume_checkpoint(path, model, optimizer, generator, *, manifest_sha256, experiment, pipeline, scheduler=None):
    from ..training.state import restore_state
    state = torch.load(path, map_location="cpu", weights_only=True)
    metadata = state.get("metadata", {})
    if state.get("format_version") != 1 or "optimizer_state" not in state or "training_state" not in state:
        raise ValueError("checkpoint has no complete resumable training state")
    if state.get("model_name") != experiment["model"] or metadata.get("split_manifest_sha256") != manifest_sha256:
        raise ValueError("resume model or manifest hash mismatch")
    actual = model_configuration(model, experiment["model"])
    if state.get("model_configuration", {}) != actual:
        raise ValueError("resume decoder architecture mismatch")
    if metadata.get("torch_version", str(torch.__version__)) != str(torch.__version__):
        raise ValueError("resume requires the recorded PyTorch version; cross-version determinism is not guaranteed")
    stored = metadata.get("config", {})
    # Only extending total epoch budget is allowed; optimizer/seed/protocol/scale/units stay identical.
    identity = lambda cfg: {key:value for key,value in cfg.items() if key != "epochs"}
    if identity(stored.get("experiment", {})) != identity(experiment) or stored.get("pipeline") != pipeline:
        raise ValueError("resume configuration mismatch")
    if experiment.get("epochs", 0) < stored.get("experiment", {}).get("epochs", 0):
        raise ValueError("resume may only extend the epoch budget")
    if scheduler is not None and "scheduler_state" not in state:
        raise ValueError("checkpoint has no scheduler state; use the original runner for legacy resumes")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    restore_state(state["training_state"], generator)
    return state["training_state"]


def load_checkpoint(path, *, require_trained=True, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format_version") != 1 or state.get("model_name") not in MODELS:
        raise ValueError("unsupported checkpoint")
    if require_trained and not state.get("metadata", {}).get("trained", False):
        raise ValueError("untrained checkpoint cannot be used for application inference")
    if state["model_name"] == "residual_cnn_v2" and set(state.get("model_configuration", {})) != {
            "upsampling", "alpha", "uncertainty_enabled", "initialization"}:
        raise ValueError("versioned checkpoint requires complete model configuration")
    model = MODELS[state["model_name"]](**state.get("model_configuration", {})).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, state["metadata"]
