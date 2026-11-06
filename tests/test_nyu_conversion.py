"""Small labelled synthetic HDF5 fixtures exercise MATLAB layout and metadata."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
h5py = pytest.importorskip("h5py", reason="install requirements-nyu.txt for native NYU conversion tests")
from scipy.io import savemat
from trident_depth.data.nyu import NYULabelledReader, convert_nyu
from trident_depth.data.nyu_integrity import inspect_exports
from trident_depth.data.dataset import DepthDataset
from trident_depth.config import PipelineConfig
from trident_depth.evaluation.nyu_preparation import registration_panels, summarize


@pytest.fixture
def nyu_fixture(tmp_path):
    mat = tmp_path / "synthetic_nyu.mat"
    n, height, width = 10, 8, 16
    y, x = np.indices((height, width))
    images = np.stack([np.stack([x, y, np.full_like(x, i)], axis=0).transpose(0, 2, 1) for i in range(n)]).astype(np.uint8)
    depths = np.full((n, width, height), 2.5, np.float32)
    raw = depths.copy()
    raw[:, 0, 0] = 0
    raw[:, 1, 0] = np.nan
    scenes = ["office_0001"] * 2 + ["kitchen_0001"] * 2 + ["office_0002"] * 2 + ["kitchen_0002"] * 2 + ["office_0003"] * 2
    with h5py.File(mat, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("depths", data=depths)
        f.create_dataset("rawDepths", data=raw)
        values = dict(scenes=scenes, sceneTypes=[s.split('_')[0] for s in scenes],
                      rawRgbFilenames=[f"{s}/frame{i}.ppm" for i,s in enumerate(scenes)],
                      rawDepthFilenames=[f"{s}/frame{i}.pgm" for i,s in enumerate(scenes)])
        refs = f.create_group("#refs#")
        for key, strings in values.items():
            cell = f.create_dataset(key, shape=(1, n), dtype=h5py.ref_dtype)
            for i, value in enumerate(strings):
                char = refs.create_dataset(f"{key}_{i}", data=np.array([ord(c) for c in value], np.uint16)[:, None])
                cell[0, i] = char.ref
    split = tmp_path / "splits.mat"
    savemat(split, dict(trainNdxs=np.arange(1, 9, dtype=np.uint16)[:, None], testNdxs=np.array([[9], [10]], np.uint16)))
    return mat, split


def test_actual_axis_mapping_metadata_and_resource_cleanup(nyu_fixture):
    mat, _ = nyu_fixture
    with NYULabelledReader(mat) as reader:
        rgb, depth, raw = reader.frame(0)
        assert rgb.shape == (8, 16, 3) and depth.shape == raw.shape == (8, 16)
        np.testing.assert_array_equal(rgb[:, :, 0], np.indices((8, 16))[1])
        np.testing.assert_array_equal(rgb[:, :, 1], np.indices((8, 16))[0])
        assert reader.metadata["scenes"][0] == "office_0001"
        assert np.isnan(raw[0, 1]) and raw[0, 0] == 0
    assert not reader.handle.id.valid


def test_conversion_masks_official_split_and_protocols(nyu_fixture, tmp_path):
    mat, split = nyu_fixture
    destination = tmp_path / "converted"
    report = convert_nyu(mat, split, destination, expected_counts=(8, 2))
    assert report["complete"] and report["total_samples"] == 10
    assert report["official_split_counts"] == {"train":8, "test":2}
    assert report["development_split_counts"] == {"validation":4, "train":4, "test":2}
    assert report["exported_files"]["missing_files"] == report["exported_files"]["errors"] == []
    assert report["raw_depth"]["nonfinite"] == 10 and report["raw_depth"]["zero"] == 10
    assert report["roundtrip"]["valid_metric_depth_max_abs_error_m"] == 0
    panels = tmp_path / "panels"
    panels.mkdir()
    names = registration_panels(destination / "manifest_validation_subset.json", panels)
    assert names == ["registration_1.png"]
    with Image.open(panels / names[0]) as image:
        assert image.size == (1280, 552)
    for protocol in ("dense_sr", "sparse_completion"):
        for scale in (2, 4, 8):
            sample = DepthDataset(destination / "manifest_development.json", PipelineConfig(protocol=protocol, scale=scale), "validation")[0]
            assert sample.target.depth.shape == (8, 16)
            assert sample.input_depth.shape == ((8 // scale, 16 // scale) if protocol == "dense_sr" else (8, 16))
    with pytest.raises(FileExistsError):
        convert_nyu(mat, split, destination, expected_counts=(8, 2))
    rows = json.loads((destination / "manifest_official.json").read_text())
    (destination / rows[0]["raw_mask"]).unlink()
    assert len(inspect_exports(rows, destination)["missing_files"]) == 1


def test_malformed_hdf5_rejected(tmp_path):
    path = tmp_path / "bad.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("depths", data=[1])
    with pytest.raises(ValueError, match="missing fields"):
        NYULabelledReader(path)


def test_benchmark_summary_uses_explicit_macro_aggregation():
    metric_names = ("rmse_m", "mae_m", "psnr_db", "ssim", "boundary_precision", "boundary_recall", "boundary_f1", "edge_preservation")
    rows = [dict(protocol="dense_sr", scale=4, method="nearest", metrics={k:value for k in metric_names},
                 timing=dict(latency_median_ms=value, python_traced_peak_bytes=value * 100, process_lifetime_peak_rss_bytes=value * 1000))
            for value in (1, 3)]
    result = summarize([dict(results=rows)])[0]
    assert result["samples"] == 2 and result["macro_mean_per_image_metrics"]["rmse_m"] == 2
    assert result["median_of_per_image_latency_medians_ms"] == 2
    assert result["max_python_traced_peak_bytes"] == 300
