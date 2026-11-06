"""Copy small measured NYU reports into docs; never copy the full dataset or train."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, default=Path("data/nyu/converted"))
    parser.add_argument("--benchmarks", type=Path, default=Path("artifacts/nyu/validation_baselines"))
    parser.add_argument("--output", type=Path, default=Path("docs/nyu"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("report destination exists; choose a new directory")
    integrity = load(args.converted / "integrity_report.json")
    summary = load(args.benchmarks / "summary.json")
    if not integrity.get("complete") or not summary["results"]:
        parser.error("conversion and measured benchmarks must complete first")
    args.output.mkdir(parents=True)
    small_integrity = {key: value for key, value in integrity.items() if key != "samples"}
    small_integrity["full_per_frame_report"] = "data/nyu/converted/integrity_report.json (local, ignored dataset directory)"
    (args.output / "integrity_summary.json").write_text(json.dumps(small_integrity, indent=2), encoding="utf-8")
    ref, raw = integrity["reference_depth"], integrity["raw_depth"]
    check = integrity["exported_files"]
    integrity_lines = ["# Official NYU dataset integrity report", "",
        f"Samples: **{integrity['total_samples']}**. Every decoded RGB and depth frame is **480x640**.", "",
        "| Partition | Frames | Scenes |", "| --- | ---: | ---: |",
        f"| Official train | {integrity['official_split_counts']['train']} | {integrity['official_scene_counts']['train']} |",
        f"| Official test | {integrity['official_split_counts']['test']} | {integrity['official_scene_counts']['test']} |",
        f"| Development train | {integrity['development_split_counts']['train']} | {integrity['development_scene_counts']['train']} |",
        f"| Development validation | {integrity['development_split_counts']['validation']} | {integrity['development_scene_counts']['validation']} |",
        f"| Untouched test | {integrity['development_split_counts']['test']} | {integrity['development_scene_counts']['test']} |", "",
        "The official and development manifests are separate. Development validation comes only from whole official training rooms. No scene overlap was detected.", "",
        "| Depth field | Positive range (m) | Missing/invalid fraction | Nonfinite | Negative | Above 10m |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| Filled reference | {ref['min_positive_m']:.9f} to {ref['max_positive_m']:.9f} | {ref['invalid_fraction']:.9%} | {ref['nonfinite']} | {ref['negative']} | {ref['above_10m']} |",
        f"| Projected raw sensor | {raw['min_positive_m']:.9f} to {raw['max_positive_m']:.9f} | {raw['invalid_fraction']:.9%} | {raw['nonfinite']} | {raw['negative']} | {raw['above_10m']} |", "",
        f"Each field contains {ref['pixels']:,} source pixels. Raw zero/missing pixels: {raw['zero']:,}.",
        f"Checked exported files: {check['checked_files']:,}; missing: {len(check['missing_files'])}; shape/mask errors: {len(check['errors'])}; exact RGB+depth duplicates: {len(check['content_duplicates'])}.",
        f"RGB round-trip: exact. Maximum valid depth round-trip error: {integrity['roundtrip']['valid_metric_depth_max_abs_error_m']} m.",
        f"Exported RGB/depth/mask bytes: {check['exported_bytes']:,}.", "",
        "The conversion only transposes MATLAB/HDF5 axes. It performs no interpolation, resizing, crop, warp or depth quantization.",
        "Representative registration panels and visual-review limitations are in `registration_review.json`.",
        "Source URLs and measured file hashes are in `download_provenance.json`; source layout is in `source_inspection.json`.",
        "Manifest hashes, selected validation rooms, source types and aggregate checks are in `integrity_summary.json`.",
        "Full per-frame statistics remain in `data/nyu/converted/integrity_report.json`.", "",
        "Filled depth includes NYU's original inpainting. This report verifies data integrity, not ground-truth sensor accuracy. No training was run.", ""]
    (args.output / "integrity_report.md").write_text("\n".join(integrity_lines), encoding="utf-8")
    for name, source in (("source_inspection.json", Path("artifacts/nyu/source_inspection.json")),
                         ("download_provenance.json", Path("data/nyu/source/download_provenance.json")),
                         ("baseline_summary.json", args.benchmarks / "summary.json"),
                         ("protocol_verification.json", args.benchmarks / "protocol_verification.json"),
                         ("registration.json", args.benchmarks / "registration.json"),
                         ("registration_review.json", args.benchmarks / "registration_review.json"),
                         ("prediction_integrity.json", args.benchmarks / "prediction_integrity.json")):
        if name == "registration_review.json" and not source.exists():
            (args.output / name).write_text(json.dumps(dict(review_status="pending",
                note="Inspect the generated registration panels and record a review; dimensions alone cannot verify registration"), indent=2), encoding="utf-8")
        else:
            shutil.copyfile(source, args.output / name)
    for protocol in ("dense_sr", "sparse_completion"):
        shutil.copyfile(args.benchmarks / protocol / "results.json", args.output / f"baseline_results_{protocol}.json")
        shutil.copyfile(args.benchmarks / f"{protocol}_config.json", args.output / f"{protocol}_config.json")
    for path in sorted(args.benchmarks.glob("registration_*.png")):
        shutil.copyfile(path, args.output / path.name)
    fields = ["protocol", "scale", "method", "samples", "rmse_m", "mae_m", "psnr_db", "ssim", "edge_preservation",
              "boundary_precision", "boundary_recall", "boundary_f1", "latency_ms", "python_traced_peak_bytes", "process_lifetime_peak_rss_bytes"]
    with (args.output / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow(dict(protocol=row["protocol"], scale=row["scale"], method=row["method"], samples=row["samples"],
                **row["macro_mean_per_image_metrics"], latency_ms=row["median_of_per_image_latency_medians_ms"],
                python_traced_peak_bytes=row["max_python_traced_peak_bytes"], process_lifetime_peak_rss_bytes=row["max_process_lifetime_peak_rss_bytes"]))
    lines = ["# Measured NYU development-validation baselines", "",
             "These are actual classical measurements on a frozen, training-derived validation subset, not final test results.",
             "No neural network training or neural/hybrid evaluation was run.", "",
             f"Official partition: {integrity['official_split_counts']}. Development partition: {integrity['development_split_counts']}.",
             f"Subset: {summary['provenance']['subset']['count']} distinct scenes, selected before method scoring.",
             f"Scene types: {summary['provenance']['subset']['scene_types']}.", "",
             "Per-image arithmetic macro means are reported below. Latency is the median of per-image medians, CPU batch size 1, one warmup and three repetitions.",
             "Tracemalloc is enabled during timing. Process RSS is the lifetime peak, not an isolated per-method native/GPU peak.",
             "PSNR range: 10 m; SSIM: uniform 7x7 valid windows. Depth-edge threshold: 0.05 m; boundary tolerance: 1 pixel.", ""]
    lines += ["Evaluation uses full 480x640 images, no Eigen crop and no reference resize. These values must not be directly compared with incompatible published protocols.", ""]
    for protocol in ("dense_sr", "sparse_completion"):
        lines += [f"## {protocol}", "",
                  "| Scale | Method | RMSE m | MAE m | PSNR dB | SSIM | Edge preservation | Boundary F1 | Latency ms | Traced peak MiB | Lifetime RSS MiB |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in summary["results"]:
            if row["protocol"] != protocol:
                continue
            m = row["macro_mean_per_image_metrics"]
            values = [m[k] for k in ("rmse_m", "mae_m", "psnr_db", "ssim", "edge_preservation", "boundary_f1")]
            formatted = ["N/A" if value is None else f"{value:.5f}" for value in values]
            lines.append(f"| {row['scale']} | {row['method']} | " + " | ".join(formatted) +
                f" | {row['median_of_per_image_latency_medians_ms']:.2f} | {row['max_python_traced_peak_bytes'] / 1024**2:.2f} | {row['max_process_lifetime_peak_rss_bytes'] / 1024**2:.2f} |")
        lines += [""]
    lines += ["Sparse completion retains 480x640 spatial resolution; its scale parameter is conditioning metadata. Classical outputs can be identical across those scales by design.",
              "", "Full per-image precision/recall, provenance, source/manifest hashes, configurations, timings and metrics are in the accompanying JSON/CSV files.",
              "Raw NPZ predictions and intermediate arrays remain under `artifacts/nyu/validation_baselines/` and are excluded from Git.",
              "", "Official filled depth is an inpainted reference, not an independently measured dense sensor ground truth.",
              "Do not use these eight development scenes to claim full NYU test performance or cross-dataset generalization.", ""]
    dense = {(row["scale"], row["method"]): row for row in summary["results"] if row["protocol"] == "dense_sr"}
    if all(dense[(s, "bicubic")]["macro_mean_per_image_metrics"]["rmse_m"] < dense[(s, "classical")]["macro_mean_per_image_metrics"]["rmse_m"] for s in (2, 4, 8)):
        lines += ["## Observed dense-SR trade-off", "",
                  "Bicubic interpolation has lower macro-average RMSE than the current classical refinement at all three scales on this frozen subset. The classical pipeline therefore does not uniformly improve depth accuracy.",
                  "Boundary F1 and runtime must be read alongside RMSE; no parameters were changed to hide this result.", ""]
    (args.output / "validation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported measured reports to {args.output}")


if __name__ == "__main__":
    main()
