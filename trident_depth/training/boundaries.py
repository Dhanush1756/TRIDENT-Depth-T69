"""Validation boundary scores using the existing evaluation implementation."""
import numpy as np
from ..evaluation.metrics import evaluate

BOUNDARY_EVALUATION = dict(edge_threshold_m=0.05, boundary_tolerance_px=1,
                           aggregation="arithmetic mean of per-frame boundary F1")


def boundary_f1_batch(prediction, target, mask):
    """Return one F1 per frame; use ground-truth masks, never predicted support."""
    p, t, m = (value.detach().cpu().numpy() for value in (prediction, target, mask))
    return [evaluate(p[i, 0], t[i, 0], m[i, 0].astype(np.bool_),
                     edge_threshold_m=BOUNDARY_EVALUATION["edge_threshold_m"],
                     boundary_tolerance_px=BOUNDARY_EVALUATION["boundary_tolerance_px"])["boundary_f1"]
            for i in range(len(p))]
