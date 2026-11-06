"""Reopen saved NYU predictions and independently verify their RMSE/MAE records."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=Path("artifacts/nyu/validation_baselines"))
    parser.add_argument("--converted", type=Path, default=Path("data/nyu/converted"))
    args = parser.parse_args()
    output = args.benchmarks / "prediction_integrity.json"
    if output.exists():
        parser.error("audit output exists")
    manifest = {r["id"]:r for r in json.loads((args.converted / "manifest_validation_subset.json").read_text(encoding="utf-8"))}
    checked = 0
    max_difference = 0.0
    for protocol in ("dense_sr", "sparse_completion"):
        report = json.loads((args.benchmarks / protocol / "results.json").read_text(encoding="utf-8"))
        if len(report["results"]) != len(manifest) * 3 * 6:
            raise ValueError("incomplete method/scale/sample comparison")
        combinations = set()
        for row in report["results"]:
            key = row["sample_id"], row["scale"], row["method"]
            if key in combinations or row["protocol"] != protocol:
                raise ValueError("duplicate/mislabelled benchmark record")
            combinations.add(key)
            source = manifest[row["sample_id"]]
            if source["split"] != "validation" or source["official_split"] != "train":
                raise ValueError("test-set leakage in validation results")
            with np.load(args.benchmarks / protocol / row["output"], allow_pickle=False) as data:
                pred, gt, valid = data["prediction"], data["ground_truth"], data["ground_truth_valid"]
                if pred.shape != gt.shape or valid.shape != gt.shape or not np.isfinite(pred).all() or valid.dtype != np.bool_ or pred.dtype != np.float32 or gt.dtype != np.float32:
                    raise ValueError("invalid saved prediction shape/dtype/values")
                original = np.load(args.converted / source["depth"], allow_pickle=False)
                if not np.array_equal(gt, original):
                    raise ValueError("saved reference does not match converted source")
                input_depth, input_valid = data["input_depth"], data["input_valid"]
                if input_depth.shape != input_valid.shape or input_valid.dtype != np.bool_ or np.any(input_depth[~input_valid] != 0):
                    raise ValueError("saved input and validity mask disagree")
                errors = pred.astype(np.float64)[valid] - gt.astype(np.float64)[valid]
                actual = dict(rmse_m=float(np.sqrt(np.mean(errors**2))), mae_m=float(np.mean(np.abs(errors))))
                for metric, value in actual.items():
                    delta = abs(value - row["metrics"][metric])
                    max_difference = max(max_difference, delta)
                    if delta > 1e-12:
                        raise ValueError(f"saved prediction metric mismatch: {metric}")
            checked += 1
        expected = {(sample_id, scale, method) for sample_id in manifest for scale in (2, 4, 8)
                    for method in ("nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical")}
        if combinations != expected:
            raise ValueError("missing or unexpected sample/scale/method combination")
        print(f"Audited {protocol}: {len(report['results'])} predictions", flush=True)
    result = dict(checked_predictions=checked, all_prediction_files_present=True,
                  shape_and_numeric_checks_passed=True, reference_arrays_match_source=True,
                  input_masks_consistent=True, test_samples_in_validation=0,
                  independent_rmse_and_mae_recomputation=True, max_absolute_metric_difference=max_difference)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
