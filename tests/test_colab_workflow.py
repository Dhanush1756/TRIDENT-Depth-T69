"""CPU-only workflow checks; no training or benchmark writes."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import pytest
torch = pytest.importorskip("torch")
from trident_depth.training.portable import select_device, recover_history
from trident_depth.training.colab import stage_archive, sha256, export_checkpoint, best_checkpoint_path
from trident_depth.models.checkpoint import save_checkpoint, resume_checkpoint
from trident_depth.models.baseline import LightweightCNN
from trident_depth.training.state import capture_state


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")


def test_interrupted_history_recovers_and_conflicts_fail(tmp_path):
    path = tmp_path / "history.jsonl"
    first = {"epoch": 1, "validation_rmse_m": .2}
    second = {"epoch": 2, "validation_rmse_m": .1}
    state = dict(epoch=2, history=[first, second])
    path.write_text(json.dumps(first) + '\n{"epoch":')
    assert recover_history(path, state) == [first, second]
    assert [json.loads(line) for line in path.read_text().splitlines()] == state["history"]
    with pytest.raises(ValueError, match="conflicts"):
        recover_history(path, dict(epoch=1, history=[first]))
    path.write_text(json.dumps(dict(first, validation_rmse_m=99)))
    with pytest.raises(ValueError, match="conflicts"):
        recover_history(path, state)


def test_scheduler_state_resume_without_optimizer_steps(tmp_path):
    model = LightweightCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.)
    # Simulated scheduler counter only; no optimizer step or training.
    scheduler.last_epoch = 1
    cfg = dict(model="lightweight_cnn", epochs=2)
    metadata = dict(trained=False, split_manifest_sha256="unit", config=dict(experiment=cfg, pipeline={}))
    generator = torch.Generator().manual_seed(69)
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, model, cfg["model"], metadata, optimizer,
                    capture_state(generator, 1, .2, 1), scheduler=scheduler)
    other = LightweightCNN()
    other_optimizer = torch.optim.Adam(other.parameters(), lr=.5)
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(other_optimizer, lambda epoch: 1.)
    resume_checkpoint(path, other, other_optimizer, generator, manifest_sha256="unit",
                      experiment=cfg, pipeline={}, scheduler=other_scheduler)
    assert other_scheduler.last_epoch == 1
    assert other_optimizer.param_groups[0]["lr"] == 1e-4
    with pytest.raises(ValueError, match="untrained"):
        export_checkpoint(path, tmp_path / "rejected.pt")
    assert not (tmp_path / "rejected.pt").exists()
    epoch = tmp_path / "epoch_001.pt"
    path.rename(epoch)
    # Best selection still works if the separate pointer was never written.
    assert best_checkpoint_path(tmp_path) == epoch


def test_dataset_archive_hash_and_traversal_checks(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("depth/sample.txt", "real file structure, synthetic test contents")
    with pytest.raises(ValueError, match="SHA256"):
        stage_archive(archive, tmp_path / "bad_hash", "0" * 64)
    destination = stage_archive(archive, tmp_path / "cache", sha256(archive))
    assert (destination / "depth/sample.txt").is_file()
    with pytest.raises(FileExistsError):
        stage_archive(archive, destination, sha256(archive))
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("../escape", "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        stage_archive(archive, tmp_path / "new_cache", sha256(archive))
    assert not (tmp_path / "escape").exists()


def test_notebook_cells_compile_and_training_stays_guarded():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/colab_dense4_baseline_pilot.ipynb").read_text())
    code = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for source in code:
        ast.parse(source)
    assert "TRAINING_APPROVED = False" in code[0]
    launches = [source for source in code if "--execute-training" in source]
    assert len(launches) == 1 and "if not TRAINING_APPROVED:" in launches[0]
    # Exercise the actual cell: disabled execution needs no dataset, GPU or runner.
    exec(compile(launches[0], "notebook-training-cell", "exec"), {"TRAINING_APPROVED": False})


@pytest.mark.parametrize("package", ["scipy", "torch", "torchvision", "nvidia-cudnn-cu12", "triton"])
def test_colab_runtime_setup_preserves_gpu_packages(tmp_path, package):
    """Execute setup with mocked pip/GPU probes; never install packages or train."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks/colab_dense4_baseline_pilot.ipynb").read_text())
    setup = next("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and "TORCH_BEFORE =" in "".join(c["source"]))
    calls = []
    url = f"https://example.invalid/{package}-unit-test.whl"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--dry-run" in command:
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": [dict(metadata=dict(name=package), download_info=dict(url=url))]}))

    def check_output(command, **kwargs):
        # Synthetic environment responses are fixtures, not GPU measurements.
        assert command[0] == sys.executable
        if "torch.__file__" in command[-1]:
            return json.dumps(dict(version="unit-test", cuda="unit-test", location="colab/torch", cuda_available=True))
        return json.dumps(dict(gpu="synthetic unit-test GPU response"))

    namespace = dict(Path=Path, sys=sys, json=json, os=os, REPO=tmp_path,
                     subprocess=SimpleNamespace(run=run, check_output=check_output))
    if package != "scipy":
        with pytest.raises(AssertionError, match="Refusing to replace Colab GPU package"):
            exec(compile(setup, "colab-setup", "exec"), namespace)
        assert len(calls) == 1 and "--dry-run" in calls[0][0]
    else:
        exec(compile(setup, "colab-setup", "exec"), namespace)
        assert namespace["PY"] == sys.executable
        assert calls[1][0][-1] == url and "--no-deps" in calls[1][0]
        assert "--no-deps" in calls[2][0] and "--no-build-isolation" in calls[2][0]
        assert calls[-1][0] == [sys.executable, "-m", "pytest", "-q"]
        assert calls[-1][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert all(command[0] == sys.executable for command, _ in calls)
    assert "trident-venv" not in setup and '"venv"' not in setup
