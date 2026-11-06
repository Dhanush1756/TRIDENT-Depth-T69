"""Portable runtime selection and recoverable epoch-boundary journal."""
import json
import os
from pathlib import Path
import tempfile
import torch


def select_device(requested="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("supported devices are auto, cpu, and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select CPU for local checks")
    return device


def atomic_json(path, value, *, lines=False):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            content = "".join(json.dumps(row, allow_nan=False) + "\n" for row in value) if lines else json.dumps(value, indent=2, allow_nan=False)
            stream.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def recover_history(path, state):
    """Checkpoint is authoritative only for a matching prefix, never newer logs."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    logged = []
    fragments = text.splitlines()
    for index, line in enumerate(fragments):
        try:
            logged.append(json.loads(line))
        except json.JSONDecodeError:
            # An interrupted last write can be repaired from the checkpoint.
            if index != len(fragments) - 1 or "history" not in state:
                raise ValueError("corrupt resume history")
    saved = state.get("history", logged)
    if [r["epoch"] for r in saved] != list(range(1, state["epoch"] + 1)):
        raise ValueError("resume history must end at checkpoint epoch")
    if len(logged) > len(saved) or logged != saved[:len(logged)]:
        raise ValueError("resume history conflicts with checkpoint; select the latest complete checkpoint")
    atomic_json(path, saved, lines=True)
    return list(saved)
