import numpy as np
import pytest
from trident_depth.classical.interpolation import interpolate
from trident_depth.classical.joint_bilateral import joint_bilateral
from trident_depth.classical.guided_filter import guided_filter
from trident_depth.classical.hole_filling import fill_holes
from trident_depth.classical.denoising import denoise
from trident_depth.classical.discontinuity import depth_edges
from trident_depth.classical.edge_detection import rgb_edges, boundary_refine
from trident_depth.classical.pipeline import reconstruct


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"])
@pytest.mark.parametrize("scale", [2, 4, 8])
def test_constant_reconstruction(method, scale):
    depth = np.full((4, 4), 3, np.float32)
    rgb = np.zeros((4 * scale, 4 * scale, 3), np.uint8)
    out, mask, _ = reconstruct(depth, depth > 0, rgb.shape[:2], rgb, method)
    assert out.shape == mask.shape == rgb.shape[:2]
    assert mask.all()
    np.testing.assert_allclose(out, 3, atol=1e-5)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"])
def test_empty_mask(method):
    out, mask, _ = reconstruct(np.zeros((4, 4)), np.zeros((4, 4), bool), (8, 8), method=method)
    assert not mask.any() and not out.any()


def test_invalid_zeros_excluded_from_filters():
    d = np.full((8, 8), 4, np.float32)
    d[3:5, 3:5] = 0
    guide = np.zeros((8, 8), np.float32)
    for fn in [joint_bilateral, guided_filter]:
        out, valid = fn(d, d > 0, guide)
        np.testing.assert_allclose(out[valid], 4, atol=1e-5)


def test_bilateral_preserves_aligned_step():
    d = np.ones((16, 16), np.float32)
    d[:, 8:] = 5
    out, _ = joint_bilateral(d, d > 0, (d - 1) / 4, sigma_range=0.01)
    np.testing.assert_allclose(out, d, atol=1e-5)


def test_discontinuities_ignore_missing_values():
    d = np.ones((8, 8), np.float32)
    d[:, 4:] = 5
    e = depth_edges(d, d > 0)
    assert e[:, 3:5].all() and e.sum() == 16
    d[:, 4:] = 0
    assert not depth_edges(d, d > 0).any()


def test_denoising_and_distance_capped_filling():
    d = np.full((8, 8), 2, np.float32)
    d[4, 4] = 99
    out, _ = denoise(d, d > 0)
    assert out[4, 4] == 2
    d[:] = 0
    d[0, 0] = 2
    out, m = fill_holes(d, d > 0, max_distance=1)
    assert m.sum() == 3 and out[0, 1] == 2 and out[7, 7] == 0


def test_rgb_only_texture_cannot_change_depth():
    d = np.full((16, 16), 2, np.float32)
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:, 8:] = 255
    assert rgb_edges(rgb).max() > 0
    out, _, confidence = boundary_refine(d, d > 0, rgb)
    np.testing.assert_array_equal(out, d)
    assert not confidence.any()


def test_measurement_consistency():
    d = np.zeros((16, 16), np.float32)
    d[::4, ::4] = 3
    out, _, _ = reconstruct(d, d > 0, (16, 16), preserve_measurements=True)
    np.testing.assert_array_equal(out[d > 0], d[d > 0])


@pytest.mark.parametrize("fn", [joint_bilateral, guided_filter])
def test_guidance_validation(fn):
    with pytest.raises(ValueError):
        fn(np.ones((8, 8)), np.ones((8, 8), bool), np.ones((2, 2)))
    with pytest.raises(ValueError):
        fn(np.ones((8, 8)), np.ones((8, 8), bool), np.ones((8, 8)), radius=0)


def test_interpolation_validation():
    with pytest.raises(ValueError):
        interpolate([[2]], [[1]], (8, 8), "unknown")
