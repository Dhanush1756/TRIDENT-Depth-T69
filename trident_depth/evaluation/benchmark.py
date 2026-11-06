"""Evaluate real local manifest samples; every output records its protocol."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
import numpy as np
from ..config import PipelineConfig
from ..data.dataset import DepthDataset
from ..classical.pipeline import reconstruct
from .metrics import evaluate


def process_peak_rss_bytes():
    """Lifetime process peak, explicitly not an isolated per-inference peak."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "quota_peak_paged",
                    "quota_paged", "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.peak_working_set
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if platform.system() == "Darwin" else peak * 1024)
    except (ImportError, OSError):
        return None


def measure(call, warmup=1, repeats=3):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive; warmup nonnegative")
    for _ in range(warmup):
        call()
    times = []
    result = None
    tracemalloc.start()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            result = call()
            times.append(1000 * (time.perf_counter() - start))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, dict(latency_median_ms=statistics.median(times), latency_samples_ms=times,
                        python_traced_peak_bytes=peak, warmup=warmup, repeats=repeats, batch_size=1,
                        process_lifetime_peak_rss_bytes=process_peak_rss_bytes(),
                        memory_scope="tracemalloc allocation peak plus lifetime process peak RSS; not isolated per-inference native/GPU peak")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True, help="dataset/protocol provenance; e.g. smoke_only_redwood")
    parser.add_argument("--checkpoint", type=Path, help="trained checkpoint; enables deep/hybrid comparisons")
    parser.add_argument("--ablations", action="store_true", help="also evaluate hybrid with each component disabled")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    config = PipelineConfig.load(args.config)
    from ..inference import InferenceEngine
    engine = InferenceEngine(args.checkpoint)
    if args.ablations and args.checkpoint is None:
        parser.error("ablations require a trained checkpoint")
    records = []
    args.output.mkdir(parents=True)
    for scale in (2, 4, 8):
        cfg = replace(config, scale=scale)
        dataset = DepthDataset(args.manifest, cfg, args.split)
        if not len(dataset):
            parser.error("selected split is empty")
        for index in range(len(dataset)):
            sample = dataset[index]
            methods = [(m, ()) for m in ("nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical")]
            if args.checkpoint:
                methods += [("deep", ()), ("hybrid", ())]
            if args.ablations:
                methods += [("hybrid", (a,)) for a in ("depth_confidence", "geometry", "residual", "uncertainty", "adaptive_fusion", "measurement")]
            for method, ablations in methods:
                result, timing = measure(lambda: engine.run(sample.input_depth, sample.input_valid,
                    sample.target.depth.shape, sample.target.rgb, method=method, scale=scale,
                    protocol=cfg.protocol, noisy=cfg.noise_std_m > 0 and cfg.protocol == "sparse_completion",
                    ablations=ablations), repeats=args.repeats)
                depth, valid, intermediates = result["depth"], result["valid"], result["maps"]
                metrics = evaluate(depth, sample.target.depth, sample.target.valid,
                                   data_range_m=cfg.max_depth - cfg.min_depth)
                suffix = "_without_" + ablations[0] if ablations else ""
                name = f"scale{scale}_sample{index}_{method}{suffix}.npz"
                np.savez_compressed(args.output / name, prediction=depth, prediction_valid=valid,
                                    input_depth=sample.input_depth, input_valid=sample.input_valid,
                                    ground_truth=sample.target.depth, ground_truth_valid=sample.target.valid,
                                    **({"rgb": sample.target.rgb} if sample.target.rgb is not None else {}),
                                    **{f"intermediate_{k}": v for k, v in intermediates.items()})
                records.append(dict(sample_id=sample.target.sample_id, scene_id=sample.target.scene_id,
                                    protocol=cfg.protocol, scale=scale, method=method, config=asdict(cfg),
                                    ablations=list(ablations), checkpoint=result["checkpoint"],
                                    metrics=metrics, timing=timing, output=name))
    report = dict(label=args.label, split=args.split, source_manifest=str(args.manifest.resolve()),
                  hardware=dict(platform=platform.platform(), processor=platform.processor(), device="CPU"),
                  numpy_version=np.__version__, results=records)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(records)} measured records to {args.output}; label={args.label}")


if __name__ == "__main__":
    main()
