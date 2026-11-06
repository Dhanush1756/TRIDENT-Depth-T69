"""Audit mathematics and optional uncertainty checks; no optimization."""
import pytest
import hashlib
import zipfile
torch = pytest.importorskip("torch")
from trident_depth.evaluation.pilot_audit import comparison_metrics, audit_checkpoint, verify_validation_payloads
from trident_depth.models.baseline import LightweightCNN
from trident_depth.training.losses import combined_loss


def test_baseline_uncertainty_is_zero_because_head_is_absent():
    model = LightweightCNN().eval()
    depth = torch.ones((1, 1, 4, 4))
    with torch.inference_mode():
        output = model(None, depth, depth.bool(), 4, "dense_sr")
        assert "log_scale" not in output
        _, terms = combined_loss(output, depth * 2, depth.bool())
        assert terms["uncertainty"].item() == 0
        # The same loss activates uncertainty only when a head supplies log_scale.
        _, enabled = combined_loss(dict(output, log_scale=torch.zeros_like(depth)), depth * 2, depth.bool())
        assert enabled["uncertainty"].item() == terms["depth"].item()


def test_comparison_uses_reductions_and_preserves_regression_sign():
    # Unit arithmetic inputs, not benchmark observations.
    values = comparison_metrics(dict(rmse_m=.1, mae_m=.05), dict(rmse_m=.08, mae_m=.06))
    assert values["rmse_m"]["reduction_percent"] == pytest.approx(20)
    assert values["rmse_m"]["cnn_minus_bicubic"] == pytest.approx(-.02)
    assert values["mae_m"]["reduction_percent"] == pytest.approx(-20)
    assert values["mae_m"]["cnn_minus_bicubic"] == pytest.approx(.01)


def test_audit_rejects_wrong_checkpoint_architecture_before_loading():
    with pytest.raises(ValueError, match="unexpected model format"):
        audit_checkpoint(dict(format_version=1, model_name="residual_cnn"), 1, {}, [], torch.zeros(1))


def test_validation_payload_identity_checks_bytes_and_rejects_test_rows(tmp_path):
    path = tmp_path / "unit_dataset.zip"
    row = dict(id="unit", split="validation", official_split="train", rgb="rgb.png", depth="depth.npy", mask="mask.npy")
    frame = dict(id="unit", source_files_sha256={})
    with zipfile.ZipFile(path, "w") as archive:
        for key in ("rgb", "depth", "mask"):
            content = f"synthetic hash fixture for {key}".encode()
            archive.writestr(row[key], content)
            frame["source_files_sha256"][key] = hashlib.sha256(content).hexdigest()
    verified = verify_validation_payloads(path, [row], [frame])
    assert verified["matched_files"] == 3 and verified["test_entries_decoded"] == 0
    with pytest.raises(ValueError, match="unexpected provenance frame"):
        verify_validation_payloads(path, [dict(row, split="test", official_split="test")], [frame])
    frame["source_files_sha256"]["depth"] = "0" * 64
    with pytest.raises(ValueError, match="differs from training archive"):
        verify_validation_payloads(path, [row], [frame])
