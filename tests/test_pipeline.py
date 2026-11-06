"""Synthetic fixtures test contracts and arithmetic, not reconstruction accuracy."""
from dataclasses import replace
import json
import numpy as np
import pytest
from PIL import Image
from trident_depth.config import PipelineConfig
from trident_depth.data.sample import DepthSample, clean_depth
from trident_depth.data.io import load_sample
from trident_depth.data.preprocessing import resize_sample
from trident_depth.data.degradation import degrade
from trident_depth.data.dataset import DepthDataset
from trident_depth.data.splits import validate_records, assign_scene_splits
from trident_depth.cli import main


def sample(depth=None, valid=None):
    depth = np.full((16, 24), 2, np.float32) if depth is None else np.asarray(depth, np.float32)
    valid = np.ones(depth.shape, bool) if valid is None else valid
    return DepthSample("synthetic-1", "synthetic-scene", depth, valid,
                       np.full((*depth.shape, 3), 100, np.uint8))


@pytest.fixture
def manifest(tmp_path):
    Image.fromarray(np.full((16, 24), 2000, np.uint16)).save(tmp_path / "depth.png")
    Image.fromarray(np.full((16, 24, 3), 100, np.uint8)).save(tmp_path / "rgb.png")
    record = dict(id="fixture", scene_id="scene-a", split="train", depth="depth.png",
                  rgb="rgb.png", depth_scale_to_m=0.001)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    return path, record


def test_invalid_depth_and_external_mask():
    d, m = clean_depth([[2, 0, -1], [np.nan, np.inf, 101]])
    np.testing.assert_array_equal(m, [[True, False, False], [False, False, False]])
    assert d.dtype == np.float32 and np.isfinite(d).all()
    d, m = clean_depth([[2, 3]], [[255, 0]])
    np.testing.assert_array_equal(d, [[2, 0]])
    with pytest.raises(ValueError):
        clean_depth([[1]], [[np.nan]])
    with pytest.raises(ValueError):
        clean_depth(np.zeros((2, 2, 3)))


def test_all_invalid_is_supported_for_completion():
    s = sample(np.zeros((8, 8)), np.zeros((8, 8), bool))
    p = degrade(s, PipelineConfig(protocol="sparse_completion"))
    assert not p.input_valid.any() and np.isfinite(p.input_depth).all()
    with pytest.raises(ValueError, match="fully valid"):
        degrade(s, PipelineConfig())


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_dense_scale_and_area_values(scale):
    d = np.arange(1, 385, dtype=np.float32).reshape(16, 24)
    p = degrade(sample(d), PipelineConfig(scale=scale, max_depth=400))
    assert p.input_depth.shape == (16 // scale, 24 // scale)
    assert p.target.rgb.shape == (16, 24, 3)
    assert p.input_valid.all()
    assert p.input_depth[0, 0] == pytest.approx(d[:scale, :scale].mean())


def test_dense_rejects_nondivisible_and_holes():
    with pytest.raises(ValueError, match="divisible"):
        degrade(sample(np.ones((9, 8))), PipelineConfig(scale=2))
    d = np.ones((8, 8), np.float32)
    d[0, 0] = 0
    with pytest.raises(ValueError, match="fully valid"):
        degrade(sample(d, d > 0), PipelineConfig())


def test_resize_does_not_dilute_valid_depth():
    d = np.array([[4, 0], [4, 0]], np.float32)
    s = resize_sample(sample(d, d > 0), (8, 8))
    np.testing.assert_allclose(s.depth[s.valid], 4)
    assert np.all(s.depth[~s.valid] == 0)
    assert s.rgb.shape == (8, 8, 3)
    s = resize_sample(sample(np.zeros((2, 2)), np.zeros((2, 2), bool)), (4, 4))
    assert not s.valid.any() and not s.depth.any()


@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("pattern", ["random", "grid"])
def test_sparse_reproducible_and_separate(scale, pattern):
    cfg = PipelineConfig(protocol="sparse_completion", scale=scale, sampling=pattern, density=0.5)
    a, b = degrade(sample(), cfg), degrade(sample(), cfg)
    np.testing.assert_array_equal(a.input_depth, b.input_depth)
    np.testing.assert_array_equal(a.input_valid, b.input_valid)
    assert a.input_depth.shape == a.target.depth.shape
    assert not a.input_valid[8, 12]
    assert a.input_valid.any()
    assert np.all(a.input_depth[~a.input_valid] == 0)
    np.testing.assert_array_equal(a.target.depth, 2)
    changed = degrade(sample(), replace(cfg, seed=70))
    assert not np.array_equal(a.input_depth, changed.input_depth)


@pytest.mark.parametrize("density", [0, 1])
def test_sparse_sampling_limits(density):
    p = degrade(sample(), PipelineConfig(protocol="sparse_completion", density=density,
                                         hole_fraction=0, noise_std_m=0))
    assert p.input_valid.mean() == density


def test_noise_out_of_range_becomes_invalid():
    p = degrade(sample(), PipelineConfig(protocol="sparse_completion", density=1,
                                         hole_fraction=0, noise_std_m=100))
    assert not p.input_valid.all()
    assert np.all(p.input_depth[p.input_valid] > 0)
    assert np.all(p.input_depth[p.input_valid] <= 100)


@pytest.mark.parametrize("kwargs", [dict(scale=3), dict(scale=True), dict(seed=-1),
    dict(density=-0.1), dict(hole_fraction=2), dict(noise_std_m=float("nan")),
    dict(min_depth=2, max_depth=1), dict(target_size=(0, 10)), dict(protocol="unknown"),
    dict(sampling="lidar"), dict(noise_std_m=-1)])
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)


def test_loader_units_masks_and_alignment(manifest):
    path, row = manifest
    s = load_sample(row, path.parent)
    np.testing.assert_array_equal(s.depth, 2)
    assert s.valid.all()
    row = dict(row, rgb=None)
    assert load_sample(row, path.parent).rgb is None
    np.save(path.parent / "depth.npy", np.full((16, 24), 3, np.float32))
    assert load_sample(dict(row, depth="depth.npy", depth_scale_to_m=1), path.parent).depth[0, 0] == 3
    Image.fromarray(np.zeros((16, 24), np.uint8)).save(path.parent / "mask.png")
    assert not load_sample(dict(row, mask="mask.png"), path.parent).valid.any()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path.parent / "wrong.png")
    with pytest.raises(ValueError, match="aligned"):
        load_sample(dict(row, rgb="wrong.png"), path.parent)
    with pytest.raises(ValueError):
        load_sample(dict(row, depth_scale_to_m=0), path.parent)
    with pytest.raises(ValueError):
        load_sample(dict(row, depth="rgb.png"), path.parent)


def test_manifest_leakage(manifest):
    path, row = manifest
    with pytest.raises(ValueError, match="scene leakage"):
        validate_records([row, dict(row, id="other", split="test")], path.parent)
    with pytest.raises(ValueError, match="file leakage"):
        validate_records([row, dict(row, id="other", scene_id="other", split="test")], path.parent)
    with pytest.raises(ValueError, match="duplicate"):
        validate_records([row, row])
    with pytest.raises(ValueError):
        validate_records([])


def test_scene_assignment_deterministic():
    rows = [dict(id=f"{scene}-{frame}", scene_id=str(scene), depth=f"{scene}-{frame}.npy", depth_scale_to_m=1)
            for scene in range(20) for frame in range(2)]
    a = assign_scene_splits(rows)
    b = assign_scene_splits(list(reversed(rows)))
    assert {r["id"]: r["split"] for r in a} == {r["id"]: r["split"] for r in b}
    assert {r["split"] for r in a} == {"train", "validation", "test"}
    assert all(len({r["split"] for r in a if r["scene_id"] == str(s)}) == 1 for s in range(20))


def test_dataset_and_cli_roundtrip(manifest, tmp_path):
    path, row = manifest
    dataset = DepthDataset(path, PipelineConfig(scale=4))
    assert len(dataset) == 1
    assert len(DepthDataset(path, PipelineConfig(), "test")) == 0
    assert dataset[0].input_depth.shape == (4, 6)
    cfg = tmp_path / "config.json"
    cfg.write_text('{"protocol":"dense_sr", "scale":4}', encoding="utf-8")
    output = tmp_path / "prepared"
    args = ["--manifest", str(path), "--config", str(cfg), "--output", str(output)]
    main(args)
    with np.load(output / "sample.npz", allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["input_depth"], 2)
        assert exported["input_valid"].dtype == np.bool_
        assert exported["rgb"].shape == (16, 24, 3)
    meta = json.loads((output / "metadata.json").read_text())
    assert meta["units"] == "metres" and meta["sample_id"] == row["id"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_sample_rejects_bad_contract():
    with pytest.raises(ValueError):
        DepthSample("a", "s", np.ones((2, 2), np.float64), np.ones((2, 2), bool))
    with pytest.raises(ValueError):
        sample(np.zeros((2, 2)))
