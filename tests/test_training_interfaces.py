import json
import numpy as np
import pytest
from PIL import Image
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader
from trident_depth.config import PipelineConfig
from trident_depth.models.baseline import LightweightCNN
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.training.dataset import TorchDepthDataset
from trident_depth.training.train import model_loss, main
from trident_depth.inference import InferenceEngine


def test_training_requires_deliberate_flag():
    with pytest.raises(SystemExit) as result:
        main(["--experiment", "does-not-exist.json"])
    assert result.value.code == 2


@pytest.mark.parametrize("protocol", ["dense_sr", "sparse_completion"])
def test_dataset_to_training_loss_without_optimizer(tmp_path, protocol):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        np.save(tmp_path / "depth.npy", np.full((16, 16), 2, np.float32))
        Image.fromarray(np.full((16, 16, 3), 50, np.uint8)).save(tmp_path / "rgb.png")
        row = dict(id="synthetic", scene_id="synthetic-scene", split="train",
                   depth="depth.npy", rgb="rgb.png", depth_scale_to_m=1)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([row]), encoding="utf-8")
        config = PipelineConfig(protocol=protocol)
        batch = next(iter(DataLoader(TorchDepthDataset(manifest, config, "train"), batch_size=1)))
        model = LightweightCNN()
        loss, terms = model_loss(model, batch, config, "cpu")
        assert torch.isfinite(loss) and torch.isfinite(terms["measurement"])
        loss.backward()  # Interface/gradient check only; no optimizer step or training claim.
    finally:
        torch.set_num_threads(old)


def test_hybrid_structure_and_ablation_paths_without_trained_claim():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        engine = InferenceEngine()
        # Inject a random model solely to exercise tensor integration in a unit test.
        # No checkpoint is marked trained or exported; application startup still enforces provenance.
        engine.model = ResidualCNN().eval()
        engine.metadata = {"trained": False, "purpose": "synthetic integration test"}
        depth = np.full((16, 16), 2, np.float32)
        rgb = np.zeros((16, 16, 3), np.uint8)
        for component in (None, "depth_confidence", "geometry", "residual", "uncertainty", "adaptive_fusion", "measurement"):
            result = engine.run(depth, depth > 0, (16, 16), rgb, method="hybrid",
                                ablations=() if component is None else (component,))
            assert result["depth"].shape == depth.shape and np.isfinite(result["depth"]).all()
            assert result["checkpoint"]["trained"] is False
    finally:
        torch.set_num_threads(old)
