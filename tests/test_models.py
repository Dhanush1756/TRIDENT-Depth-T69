import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="install CPU PyTorch for Phase 4 tests")
from trident_depth.models.baseline import LightweightCNN
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.models.checkpoint import save_checkpoint, load_checkpoint
from trident_depth.models.fusion import fuse
from trident_depth.training.losses import combined_loss, masked_l1
from trident_depth.inference import InferenceEngine


@pytest.fixture(autouse=True)
def limit_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("model_type", [LightweightCNN, ResidualCNN])
@pytest.mark.parametrize("shape", [(16, 24), (17, 19)])
def test_shapes_losses_and_gradients(model_type, shape):
    model = model_type()
    initial = torch.full((1, 1, *shape), 2.0)
    mask = torch.ones_like(initial, dtype=torch.bool)
    out = model(torch.zeros((1, 3, *shape)), initial, mask, 8, "sparse_completion")
    assert out["depth"].shape == initial.shape
    loss, terms = combined_loss(out, initial + 1, mask, observations=initial,
                                input_mask=mask, noisy=True)
    assert torch.isfinite(loss) and all(torch.isfinite(x) for x in terms.values())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_empty_and_invalid_target_losses():
    pred = torch.ones((1, 1, 8, 8), requires_grad=True)
    gt = torch.full_like(pred, float("nan"))
    mask = torch.zeros_like(pred, dtype=torch.bool)
    loss, _ = combined_loss({"depth": pred, "log_scale": pred * 0}, gt, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() == 0
    mask[:, :, 0, 0] = True
    gt[:, :, 0, 0] = 3
    assert masked_l1(pred, gt, mask).item() == 2


def test_checkpoint_roundtrip_and_untrained_guard(tmp_path):
    model = ResidualCNN().eval()
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, model, "residual_cnn", {"trained": False, "purpose": "unit test"})
    loaded, _ = load_checkpoint(path, require_trained=False)
    d = torch.ones((1, 1, 8, 8))
    with torch.inference_mode():
        torch.testing.assert_close(model(None, d, d)["depth"], loaded(None, d, d)["depth"])
    with pytest.raises(ValueError, match="untrained"):
        load_checkpoint(path)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, "residual_cnn", {"trained": False})
    with pytest.raises(ValueError, match="provenance"):
        save_checkpoint(tmp_path / "bad.pt", model, "residual_cnn", {"trained": True})


def test_fusion_weights_and_consistency():
    d = torch.ones((1, 1, 8, 8))
    fused, weights = fuse(d, d * 2, d * 3, d * 0.5, d * 0.5, d * 0)
    torch.testing.assert_close(weights.sum(dim=0), d)
    assert fused.min() >= 1 and fused.max() <= 3
    hard, _ = fuse(d, d * 2, d * 3, d, d, d, observations=d * 4,
                   observation_mask=d.bool(), observation_confidence=d)
    torch.testing.assert_close(hard, d * 4)
    soft, _ = fuse(d, d, d, d, d, d, observations=d * 4,
                   observation_mask=d.bool(), observation_confidence=d * 0.25)
    torch.testing.assert_close(soft, d * 1.75)


def test_engine_refuses_fake_neural_results():
    engine = InferenceEngine()
    d = np.ones((8, 8), np.float32)
    assert engine.run(d, d > 0, (16, 16), method="bilinear")["depth"].shape == (16, 16)
    with pytest.raises(ValueError, match="trained checkpoint"):
        engine.run(d, d > 0, (16, 16), method="hybrid")
