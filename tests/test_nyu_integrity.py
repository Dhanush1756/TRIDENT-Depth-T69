import numpy as np
import pytest
from trident_depth.data.nyu_integrity import depth_statistics, summarize_statistics


def test_invalid_categories_and_ranges_are_reported_without_clipping():
    stats = depth_statistics(np.array([0, -1, np.nan, np.inf, 2, 12]))
    assert stats["nonfinite"] == 2 and stats["zero"] == stats["negative"] == 1
    assert stats["valid_positive"] == 2 and stats["above_10m"] == 1
    assert stats["invalid_fraction"] == pytest.approx(4 / 6)
    assert stats["min_positive_m"] == 2 and stats["max_positive_m"] == 12
    combined = summarize_statistics([stats, stats])
    assert combined["pixels"] == 12 and combined["valid_positive"] == 4
    assert combined["invalid_fraction"] == pytest.approx(4 / 6)
