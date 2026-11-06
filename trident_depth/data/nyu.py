"""Streaming conversion of the inspected official NYU labelled MAT release."""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
from .nyu_integrity import depth_statistics, summarize_statistics, file_sha256, inspect_exports
from .nyu_splits import official_membership, room_group, development_holdout, select_validation_subset
from .sample import clean_depth
from .splits import validate_records


def matlab_string(handle, reference):
    codes = np.asarray(handle[reference])
    if codes.dtype.kind not in "ui":
        raise ValueError("NYU metadata is not a MATLAB character reference")
    return "".join(chr(int(v)) for v in codes.ravel(order="F") if v)


class NYULabelledReader:
    def __init__(self, path):
        self.handle = h5py.File(path, "r")
        try:
            h = self.handle
            required = {"images", "depths", "rawDepths", "scenes", "sceneTypes", "rawRgbFilenames", "rawDepthFilenames"}
            if not required <= h.keys():
                raise ValueError(f"NYU source missing fields: {sorted(required - h.keys())}")
            n, w, height = h["depths"].shape
            if (h["images"].shape != (n, 3, w, height) or h["rawDepths"].shape != (n, w, height)
                    or h["images"].dtype != np.uint8 or h["depths"].dtype != np.float32 or h["rawDepths"].dtype != np.float32):
                raise ValueError("source layout/dtypes differ from inspected NYU release")
            self.count, self.shape = n, (height, w)
            self.metadata = {}
            for key in ("scenes", "sceneTypes", "rawRgbFilenames", "rawDepthFilenames"):
                refs = np.asarray(h[key]).ravel()
                if len(refs) != n:
                    raise ValueError(f"metadata length mismatch: {key}")
                self.metadata[key] = [matlab_string(h, ref) for ref in refs]
                if not all(self.metadata[key]):
                    raise ValueError(f"empty NYU metadata value: {key}")
        except Exception:
            self.handle.close()
            raise

    def frame(self, index):
        # MATLAB/HDF5 storage order is reversed: N,C,W,H -> H,W,C and N,W,H -> H,W.
        rgb = np.asarray(self.handle["images"][index]).transpose(2, 1, 0).copy()
        depth = np.asarray(self.handle["depths"][index]).T.copy()
        raw = np.asarray(self.handle["rawDepths"][index]).T.copy()
        return rgb, depth, raw

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def convert_nyu(mat_path, splits_path, output, *, expected_counts=(795, 654), subset_size=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing conversion directory: {output}")
    membership = official_membership(loadmat(splits_path), sum(expected_counts))
    counts = Counter(membership.values())
    if (counts["train"], counts["test"]) != tuple(expected_counts):
        raise ValueError("official split counts differ from expected release")
    with NYULabelledReader(mat_path) as reader:
        if reader.count != len(membership):
            raise ValueError("source frame count disagrees with official split")
        records = []
        for i in range(reader.count):
            sample_id = f"nyu_{i + 1:05d}"
            scene = reader.metadata["scenes"][i]
            records.append(dict(id=sample_id, source_index_1based=i + 1,
                source_scene=scene, scene_id=room_group(scene), scene_type=reader.metadata["sceneTypes"][i],
                official_split=membership[i + 1], split=membership[i + 1],
                rgb=f"rgb/{sample_id}.png", depth=f"depth/{sample_id}.npy", mask=f"mask/{sample_id}.npy",
                raw_depth=f"raw_depth/{sample_id}.npy", raw_mask=f"raw_mask/{sample_id}.npy",
                depth_scale_to_m=1.0, reference_kind="official_inpainted_depths",
                source_rgb_filename=reader.metadata["rawRgbFilenames"][i],
                source_depth_filename=reader.metadata["rawDepthFilenames"][i]))
        validate_records(records, output)
        development, selection = development_holdout(records)
        validate_records(development, output)
        subset = select_validation_subset(development, subset_size)
        output.mkdir(parents=True)
        for directory in ("rgb", "depth", "mask", "raw_depth", "raw_mask"):
            (output / directory).mkdir()
        per_sample = []
        max_pixel_error = 0.0
        for i, row in enumerate(records):
            rgb, original, raw_original = reader.frame(i)
            stat, raw_stat = depth_statistics(original), depth_statistics(raw_original)
            # Preserve positive metric values, including >10m; range policies belong to configs.
            depth, mask = clean_depth(original, max_depth=float(np.finfo(np.float32).max))
            raw, raw_mask = clean_depth(raw_original, max_depth=float(np.finfo(np.float32).max))
            Image.fromarray(rgb).save(output / row["rgb"])
            for key, array in (("depth", depth), ("mask", mask), ("raw_depth", raw), ("raw_mask", raw_mask)):
                np.save(output / row[key], array, allow_pickle=False)
            # Exact orientation and serialization checks against the actual source, every sample.
            with Image.open(output / row["rgb"]) as saved:
                if not np.array_equal(np.array(saved), rgb):
                    raise ValueError(f"RGB round-trip mismatch: {row['id']}")
            for key, source, valid in (("depth", original, mask), ("raw_depth", raw_original, raw_mask)):
                saved = np.load(output / row[key], allow_pickle=False)
                if not np.array_equal(saved[valid], source[valid]):
                    raise ValueError(f"metric-depth round-trip mismatch: {row['id']}")
                if valid.any():
                    max_pixel_error = max(max_pixel_error, float(np.abs(saved[valid] - source[valid]).max()))
            per_sample.append(dict(id=row["id"], official_split=row["official_split"],
                                   scene_id=row["scene_id"], scene_type=row["scene_type"],
                                   dimensions=list(depth.shape), reference=stat, raw=raw_stat))
            if (i + 1) % 100 == 0 or i + 1 == reader.count:
                print(f"Converted and round-trip checked {i + 1}/{reader.count}", flush=True)
        file_checks = inspect_exports(records, output)
        if file_checks["missing_files"] or file_checks["errors"] or any(d["cross_official_split"] for d in file_checks["content_duplicates"]):
            raise ValueError("export integrity failed; incomplete conversion retained for inspection")
        source_layout = {k: dict(shape=list(reader.handle[k].shape), dtype=str(reader.handle[k].dtype))
                         for k in ("images", "depths", "rawDepths")}
    source = dict(dataset="Official NYU Depth V2 labelled release", units="metres",
                  mat_file=Path(mat_path).name, mat_sha256=file_sha256(mat_path),
                  split_file=Path(splits_path).name, splits_sha256=file_sha256(splits_path),
                  official_page="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
                  source_layout=source_layout)
    official_path = output / "manifest_official.json"
    development_path = output / "manifest_development.json"
    subset_path = output / "manifest_validation_subset.json"
    write_json(official_path, records)
    write_json(development_path, development)
    write_json(subset_path, subset)
    manifest_hashes = {p.name: file_sha256(p) for p in (official_path, development_path, subset_path)}
    report = dict(source=source, manifests_sha256=manifest_hashes,
        total_samples=len(records), official_split_counts=dict(counts),
        official_scene_counts={s: len({r["scene_id"] for r in records if r["official_split"] == s}) for s in ("train", "test")},
        development_split_counts=dict(Counter(r["split"] for r in development)),
        development_scene_counts={s: len({r["scene_id"] for r in development if r["split"] == s}) for s in ("train", "validation", "test")},
        scene_overlap=[], validation_selection=selection,
        benchmark_subset=dict(rule="Before scoring: round-robin across validation scene types ranked by prevalence; median-index frame per room",
                              count=len(subset), ids=[r["id"] for r in subset],
                              scene_types=dict(Counter(r["scene_type"] for r in subset))),
        reference_depth=summarize_statistics([r["reference"] for r in per_sample]),
        raw_depth=summarize_statistics([r["raw"] for r in per_sample]),
        roundtrip=dict(samples_checked=len(records), rgb_exact=True, valid_metric_depth_max_abs_error_m=max_pixel_error,
                       spatial_operation="axis transpose only; no resize, crop, warp or depth quantization"),
        registration=dict(source="Official projected raw depth and aligned filled depth in RGB coordinates",
                          structural_checks="All RGB/depth/mask dimensions and source axis mapping checked",
                          visual_verification="Representative RGB/depth overlays must be inspected separately; dimensions alone are insufficient"),
        exported_files=file_checks, samples=per_sample,
        complete=True, training_started=False)
    write_json(output / "integrity_report.json", report)
    write_json(output / "provenance.json", source)
    print(json.dumps({k: report[k] for k in ("total_samples", "official_split_counts", "development_split_counts", "benchmark_subset")}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=Path("data/nyu/source/nyu_depth_v2_labeled.mat"))
    parser.add_argument("--splits", type=Path, default=Path("data/nyu/source/splits.mat"))
    parser.add_argument("--output", type=Path, default=Path("data/nyu/converted"))
    parser.add_argument("--subset-size", type=int, default=8)
    args = parser.parse_args(argv)
    convert_nyu(args.mat, args.splits, args.output, subset_size=args.subset_size)


if __name__ == "__main__":
    main()
