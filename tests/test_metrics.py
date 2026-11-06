import numpy as np
import pytest
from trident_depth.evaluation.metrics import evaluate
from trident_depth.evaluation.benchmark import measure


def test_metrics_exact_and_constant_offset():
    gt = np.full((16, 16), 2.0)
    mask = np.ones_like(gt, bool)
    result = evaluate(gt, gt, mask)
    assert result["rmse_m"] == result["mae_m"] == 0
    assert result["psnr_infinite"] and result["psnr_db"] is None
    assert result["ssim"] == pytest.approx(1)
    assert result["edge_preservation"] == result["boundary_f1"] == 1
    result = evaluate(gt + 1, gt, mask, data_range_m=10)
    assert result["rmse_m"] == result["mae_m"] == 1
    assert result["psnr_db"] == pytest.approx(20)
    # Population-window SSIM for constant images has only the luminance factor.
    assert result["ssim"] == pytest.approx((2 * 3 * 2 + 0.01) / (3 ** 2 + 2 ** 2 + 0.01))


def test_mask_excludes_bad_targets_but_not_zero_predictions():
    gt = np.full((16, 16), 2.0)
    gt[0, 0] = np.nan
    mask = np.isfinite(gt)
    result = evaluate(np.zeros_like(gt), gt, mask)
    assert result["rmse_m"] == result["mae_m"] == 2
    assert result["valid_pixels"] == 255
    with pytest.raises(ValueError):
        evaluate(gt, gt, np.zeros_like(mask))


def test_boundary_tolerance_and_false_texture_edges():
    gt = np.ones((16, 16))
    gt[:, 8:] = 3
    mask = np.ones_like(gt, bool)
    assert evaluate(gt, gt, mask)["boundary_f1"] == 1
    flat = np.ones_like(gt)
    assert evaluate(flat, gt, mask)["boundary_f1"] == 0
    assert evaluate(gt, flat, mask)["boundary_precision"] == 0
    shifted = np.ones_like(gt)
    shifted[:, 9:] = 3
    assert evaluate(shifted, gt, mask, boundary_tolerance_px=1)["boundary_f1"] == 1


def test_ssim_insufficient_support_and_invalid_prediction():
    gt = np.ones((4, 4))
    mask = np.ones_like(gt, bool)
    assert evaluate(gt, gt, mask)["ssim"] is None
    with pytest.raises(ValueError):
        evaluate(gt * np.nan, gt, mask)


def test_timing_executes_warmup_and_repeats():
    calls = []
    result, timing = measure(lambda: calls.append(1) or 42, warmup=2, repeats=3)
    assert result == 42 and len(calls) == 5
    assert len(timing["latency_samples_ms"]) == 3
    assert timing["python_traced_peak_bytes"] >= 0
