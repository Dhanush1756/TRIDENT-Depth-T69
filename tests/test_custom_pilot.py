"""Custom pilot preparation tests: no training steps or inference exports."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from trident_depth.training.custom_pilot import verify_configuration
from trident_depth.training.boundaries import boundary_f1_batch
from trident_depth.evaluation.metrics import evaluate
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.training.losses import combined_loss

ROOT = Path(__file__).resolve().parents[1]


def test_two_epoch_custom_config_preserves_approved_architecture_and_loss():
    cfg, pipeline = verify_configuration(ROOT / "configs/phase5/pilot_dense4_residual.json")
    approved = json.loads((ROOT / "configs/phase5/dense_sr_4x_residual_cnn.json").read_text())
    assert {k for k in cfg if cfg[k] != approved[k]} == {"epochs", "output"}
    assert cfg["epochs"] == 2 and approved["epochs"] == 20
    assert pipeline.protocol == "dense_sr" and pipeline.scale == 4
    assert sum(p.numel() for p in ResidualCNN(upsampling=cfg["decoder_upsampling"]).parameters()) == 473059


def test_custom_configuration_rejects_longer_training(tmp_path):
    for name in ("pilot_dense4_residual.json", "dense_sr_4x_residual_cnn.json", "dense_sr_4x_pipeline.json"):
        (tmp_path / name).write_bytes((ROOT / "configs/phase5" / name).read_bytes())
    path = tmp_path / "pilot_dense4_residual.json"
    cfg = json.loads(path.read_text())
    cfg["epochs"] = 3
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="exactly two epochs"):
        verify_configuration(path)


def test_boundary_wrapper_matches_existing_metric_with_invalid_targets():
    target = np.ones((2, 1, 9, 11), np.float32)
    target[..., 5:] = 2
    mask = np.ones_like(target, bool)
    mask[:, :, 0] = False
    target[:, :, 0] = np.nan
    prediction = np.nan_to_num(target, nan=1)
    prediction[1] = 1
    actual = boundary_f1_batch(torch.from_numpy(prediction), torch.from_numpy(target), torch.from_numpy(mask))
    expected = [evaluate(prediction[i,0], target[i,0], mask[i,0])["boundary_f1"] for i in range(2)]
    assert actual == expected and actual == [1., 0.]


def test_custom_uncertainty_head_has_finite_gradients_without_boundary_supervision():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ResidualCNN(upsampling="nearest")
        depth = torch.ones((1,1,9,11))
        output = model(None, depth, depth.bool())
        loss, terms = combined_loss(output, depth * 2, depth.bool())
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(terms["uncertainty"])
        assert model.log_scale_head.weight.grad is not None
        assert torch.isfinite(model.log_scale_head.weight.grad).all()
        assert model.boundary_head.weight.grad is None
    finally:
        torch.set_num_threads(old)


def test_custom_notebook_has_safe_launch_comparison_and_no_export():
    nb = json.loads((ROOT / "notebooks/colab_dense4_custom_pilot.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for source in code:
        ast.parse(source)
    settings = code[0]
    assert "TRAINING_APPROVED = False" in settings and "EXPORT_APPROVED = False" in settings
    assert "pilot_dense4_residual_seed69" in settings
    joined = "\n".join(code)
    assert "pilot_dense4_residual.json" in joined and "pilot_dense4_lightweight.json" not in joined
    assert "export_checkpoint" not in joined and '"trident_depth.training.colab"' not in joined
    assert "trident-venv" not in joined
    launch = next(s for s in code if "--execute-training" in s)
    assert "--record-boundary-f1" in launch and "--provenance" in launch
    exec(compile(launch, "custom-launch", "exec"), dict(TRAINING_APPROVED=False))
    comparison = next(s for s in code if "comparison_command =" in s)
    exec(compile(comparison, "custom-comparison", "exec"), dict(EXPORT_APPROVED=False, RUN_COMPARISON=False))
