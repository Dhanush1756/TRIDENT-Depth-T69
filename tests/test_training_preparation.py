"""Preparation checks only: no optimizer steps or trained checkpoints."""
import json
import random
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from trident_depth.models.baseline import LightweightCNN
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.models.checkpoint import save_checkpoint, load_checkpoint, resume_checkpoint
from trident_depth.training.state import seed_everything, capture_state, restore_state
from trident_depth.training.validation import audit_training_manifest, validation_errors
from trident_depth.training.losses import combined_loss


def test_rng_and_shuffle_state_roundtrip():
    seed_everything(69)
    generator = torch.Generator().manual_seed(69)
    state = capture_state(generator, 1, .1, 1)
    draw = lambda: (random.random(), np.random.random(), torch.rand(3), torch.randperm(20, generator=generator))
    first = draw()
    restore_state(state, generator)
    second = draw()
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)


def test_optimizer_model_and_rng_resume_without_training(tmp_path):
    seed_everything(69)
    model = LightweightCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Populate synthetic optimizer moments to test serialization, without any step.
    parameter = next(model.parameters())
    optimizer.state[parameter] = dict(step=torch.tensor(2.), exp_avg=torch.full_like(parameter, .25), exp_avg_sq=torch.full_like(parameter, .5))
    generator = torch.Generator().manual_seed(69)
    cfg = dict(model="lightweight_cnn", epochs=2, initializer="bicubic")
    pipeline = dict(protocol="dense_sr", scale=4)
    metadata = dict(trained=False, purpose="synthetic resume unit test", split_manifest_sha256="hash",
                    config=dict(experiment=cfg, pipeline=pipeline))
    path = tmp_path / "unit_resume.pt"
    save_checkpoint(path, model, "lightweight_cnn", metadata, optimizer, capture_state(generator, 1, .2, 1))
    next_order = torch.randperm(10, generator=generator)
    restored = LightweightCNN()
    other_optimizer = torch.optim.Adam(restored.parameters(), lr=1.)
    result = resume_checkpoint(path, restored, other_optimizer, generator,
                              manifest_sha256="hash", experiment=dict(cfg, epochs=3), pipeline=pipeline)
    assert result["epoch"] == 1 and result["best_rmse"] == .2
    for left,right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert other_optimizer.param_groups[0]["lr"] == 1e-4
    torch.testing.assert_close(other_optimizer.state[next(restored.parameters())]["exp_avg"], torch.full_like(parameter, .25))
    torch.testing.assert_close(torch.randperm(10, generator=generator), next_order, rtol=0, atol=0)
    with pytest.raises(ValueError, match="manifest"):
        resume_checkpoint(path, restored, other_optimizer, generator, manifest_sha256="wrong", experiment=cfg, pipeline=pipeline)
    with pytest.raises(ValueError, match="configuration"):
        resume_checkpoint(path, restored, other_optimizer, generator, manifest_sha256="hash", experiment=dict(cfg, initializer="guided"), pipeline=pipeline)
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("log_scale", [-1e6, 1e6])
def test_extreme_uncertainty_and_partial_invalid_targets(log_scale):
    pred = torch.full((1, 1, 4, 4), 100., requires_grad=True)
    target = torch.full_like(pred, 2.)
    mask = torch.ones_like(pred, dtype=torch.bool)
    target[:, :, 0] = float("nan")
    mask[:, :, 0] = False
    scale = torch.full_like(pred, log_scale, requires_grad=True)
    loss, terms = combined_loss(dict(depth=pred, log_scale=scale), target, mask)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    assert torch.isfinite(pred.grad).all() and torch.isfinite(scale.grad).all()
    assert pred.grad[:, :, 0].abs().sum() == 0


def test_nonfinite_valid_targets_and_wrong_uncertainty_shape_rejected():
    pred = torch.ones((1, 1, 4, 4))
    mask = pred.bool()
    with pytest.raises(ValueError, match="reference"):
        combined_loss(dict(depth=pred), pred * float("nan"), mask)
    with pytest.raises(ValueError, match="log_scale"):
        combined_loss(dict(depth=pred, log_scale=torch.zeros(1)), pred, mask)


def test_official_test_relabelled_as_validation_is_rejected(tmp_path):
    rows = [dict(id=str(i),scene_id=str(i),depth=f"{i}.npy",depth_scale_to_m=1,
                 split=split, official_split=official)
            for i,(split,official) in enumerate((("train","train"),("validation","test")))]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="official test"):
        audit_training_manifest(path)


def test_validation_metrics_keep_reference_mask_and_units():
    prediction = torch.tensor([1.,3.,100.])
    target = torch.tensor([2.,2.,float("nan")])
    assert validation_errors(prediction,target,torch.tensor([True,True,False])) == (2.,2.,2)


def test_nearest_decoder_checkpoint_preserves_architecture(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ResidualCNN(upsampling="nearest")
        path = tmp_path / "nearest_untrained.pt"
        save_checkpoint(path, model, "residual_cnn", {"trained":False})
        loaded,_ = load_checkpoint(path, require_trained=False)
        assert loaded.upsampling == "nearest"
        d = torch.ones((1,1,9,11))
        output = loaded(None,d,d)
        loss = output["depth"].mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in loaded.parameters() if p.grad is not None)
    finally:
        torch.set_num_threads(old)


def test_inference_uses_checkpoint_initializer():
    from trident_depth.inference import InferenceEngine
    from trident_depth.classical.interpolation import interpolate
    engine = InferenceEngine()
    class EchoInitial:
        def __call__(self, rgb, initial, validity, scale, protocol):
            return dict(depth=initial, log_scale=torch.zeros_like(initial))
    engine.model = EchoInitial()  # Synthetic interface spy, never a trained checkpoint.
    engine.metadata = dict(trained=False, config=dict(experiment=dict(initializer="bicubic")))
    observed = np.arange(1,17,dtype=np.float32).reshape(4,4)
    expected,_ = interpolate(observed, observed>0, (8,8), "bicubic")
    actual = engine.run(observed, observed>0, (8,8), method="deep")
    np.testing.assert_array_equal(actual["depth"],expected)
