"""Verify NYU protocols, render registration panels, and benchmark a frozen validation subset."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import importlib.metadata
import os
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ..config import PipelineConfig
from ..data.dataset import DepthDataset
from ..data.io import load_sample, load_array
from ..data.nyu_integrity import file_sha256
from ..data.nyu import write_json
from ..classical.discontinuity import depth_edges
from .benchmark import main as benchmark


def depth_preview(depth, valid, maximum=10.0):
    gray = np.clip(depth / maximum * 255, 0, 255).astype(np.uint8)
    colors = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return np.where(valid[..., None], colors, 0).astype(np.uint8)


def registration_panels(manifest, output):
    manifest, output = Path(manifest), Path(output)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    sheets = []
    for start in range(0, len(records), 4):
        chunk = records[start:start + 4]
        canvas = Image.new("RGB", (1280, 276 * len(chunk)), "#101817")
        draw = ImageDraw.Draw(canvas)
        for row_number, row in enumerate(chunk):
            sample = load_sample(row, manifest.parent, max_depth=float(np.finfo(np.float32).max))
            raw = load_array(manifest.parent / row["raw_depth"])
            raw_valid = load_array(manifest.parent / row["raw_mask"])
            edges = depth_edges(raw, raw_valid)
            overlay = sample.rgb.copy()
            overlay[edges] = (30, 255, 80)
            images = [sample.rgb, depth_preview(sample.depth, sample.valid),
                      depth_preview(raw, raw_valid), overlay]
            titles = ["RGB", "Official filled depth", "Projected raw depth (black=missing)", "Raw depth jumps on RGB (green)"]
            y = row_number * 276
            draw.text((8, y + 3), f"{row['id']} | {row['source_scene']} | {row['scene_type']} | validation, official=train", fill="white")
            for col, (array, title) in enumerate(zip(images, titles)):
                x = col * 320
                draw.text((x + 5, y + 18), title, fill="white")
                canvas.paste(Image.fromarray(array).resize((320, 240)), (x, y + 34))
        path = output / f"registration_{start // 4 + 1}.png"
        canvas.save(path)
        sheets.append(path.name)
    return sheets


def summarize(reports):
    groups = defaultdict(list)
    for report in reports:
        for row in report["results"]:
            groups[(row["protocol"], row["scale"], row["method"])].append(row)
    output = []
    names = ("rmse_m", "mae_m", "psnr_db", "ssim", "boundary_precision", "boundary_recall", "boundary_f1", "edge_preservation")
    for (protocol, scale, method), rows in sorted(groups.items()):
        metrics = {}
        for name in names:
            values = [r["metrics"][name] for r in rows if r["metrics"][name] is not None]
            metrics[name] = statistics.mean(values) if values else None
        output.append(dict(protocol=protocol, scale=scale, method=method, samples=len(rows),
            macro_mean_per_image_metrics=metrics,
            median_of_per_image_latency_medians_ms=statistics.median(r["timing"]["latency_median_ms"] for r in rows),
            max_python_traced_peak_bytes=max(r["timing"]["python_traced_peak_bytes"] for r in rows),
            max_process_lifetime_peak_rss_bytes=max((r["timing"]["process_lifetime_peak_rss_bytes"] or 0) for r in rows)))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, default=Path("data/nyu/converted"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nyu/validation_baselines"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new run directory")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    integrity = json.loads((args.converted / "integrity_report.json").read_text(encoding="utf-8"))
    if not integrity.get("complete"):
        parser.error("NYU conversion has not passed integrity checks")
    manifest = args.converted / "manifest_validation_subset.json"
    if file_sha256(manifest) != integrity["manifests_sha256"][manifest.name]:
        parser.error("validation subset differs from audited, preselected manifest")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if any(r["split"] != "validation" or r["official_split"] != "train" for r in rows):
        parser.error("subset contains samples outside the training-derived validation holdout")
    args.output.mkdir(parents=True)
    provenance = dict(source=integrity["source"], manifests_sha256=integrity["manifests_sha256"],
                      validation_selection=integrity["validation_selection"],
                      subset=integrity["benchmark_subset"], repeats=args.repeats,
                      label="NYU official labelled release; training-derived development validation; no test-set tuning",
                      protocols_note="dense SR and synthetic sparse/noisy completion reported separately; sparse protocol retains target resolution",
                      package_versions={name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "scipy", "h5py", "opencv-python-headless")},
                      cpu_logical_cores=os.cpu_count(), opencv_threads=cv2.getNumThreads(),
                      training_started=False)
    write_json(args.output / "provenance.json", provenance)
    configs = {}
    checks = []
    for protocol in ("dense_sr", "sparse_completion"):
        cfg = PipelineConfig(protocol=protocol, scale=4, seed=69, min_depth=0, max_depth=10,
                             density=.05, noise_std_m=.01, hole_fraction=.1)
        config_path = args.output / f"{protocol}_config.json"
        write_json(config_path, asdict(cfg))
        configs[protocol] = config_path
        for scale in (2, 4, 8):
            dataset = DepthDataset(manifest, replace(cfg, scale=scale), "validation")
            for index in range(len(dataset)):
                sample = dataset[index]
                if not np.isfinite(sample.input_depth).all() or np.any(sample.input_depth[~sample.input_valid] != 0):
                    raise ValueError("protocol produced invalid numeric values")
                checks.append(dict(id=sample.target.sample_id, protocol=protocol, scale=scale,
                                   target_shape=list(sample.target.depth.shape), rgb_shape=list(sample.target.rgb.shape),
                                   input_shape=list(sample.input_depth.shape), input_valid_fraction=float(sample.input_valid.mean()),
                                   target_valid_fraction=float(sample.target.valid.mean()), input_units="metres"))
    write_json(args.output / "protocol_verification.json", checks)
    sheets = registration_panels(manifest, args.output)
    write_json(args.output / "registration.json", dict(panels=sheets,
        source_registration_preserved=True, all_exports_roundtrip_exact=True,
        review="Representative panels require visual inspection; no automatic subpixel registration guarantee"))
    reports = []
    for protocol, config_path in configs.items():
        start = time.perf_counter()
        print(f"Benchmarking {protocol}: {len(rows)} frames, 3 scales, 6 methods, {args.repeats} repeats", flush=True)
        target = args.output / protocol
        benchmark(["--manifest", str(manifest), "--config", str(config_path), "--split", "validation",
                   "--output", str(target), "--repeats", str(args.repeats), "--label", "NYU_training_derived_validation"])
        report = json.loads((target / "results.json").read_text(encoding="utf-8"))
        report["dataset_provenance"] = provenance
        report["configuration_sha256"] = file_sha256(config_path)
        write_json(target / "results.json", report)
        reports.append(report)
        print(f"Completed {protocol} in {time.perf_counter() - start:.1f} seconds", flush=True)
    write_json(args.output / "summary.json", dict(provenance=provenance,
        aggregation="Macro arithmetic mean of each per-image metric; timing is median of per-image latency medians",
        results=summarize(reports)))
    print(f"Saved verified NYU validation baseline results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
