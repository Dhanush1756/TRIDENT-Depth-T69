"""Streaming statistics and exported-file checks for NYU preparation."""
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_statistics(array, limit_m=10.0):
    array = np.asarray(array)
    finite = np.isfinite(array)
    positive = finite & (array > 0)
    return dict(pixels=int(array.size), nonfinite=int((~finite).sum()),
                zero=int((finite & (array == 0)).sum()), negative=int((finite & (array < 0)).sum()),
                above_10m=int((finite & (array > limit_m)).sum()),
                valid_positive=int(positive.sum()), invalid_fraction=float((~positive).mean()),
                min_positive_m=float(array[positive].min()) if positive.any() else None,
                max_positive_m=float(array[positive].max()) if positive.any() else None)


def summarize_statistics(stats):
    keys = ("pixels", "nonfinite", "zero", "negative", "above_10m", "valid_positive")
    total = {key: sum(row[key] for row in stats) for key in keys}
    total["invalid_fraction"] = 1 - total["valid_positive"] / total["pixels"] if total["pixels"] else None
    minimums = [r["min_positive_m"] for r in stats if r["min_positive_m"] is not None]
    maximums = [r["max_positive_m"] for r in stats if r["max_positive_m"] is not None]
    total.update(min_positive_m=min(minimums) if minimums else None,
                 max_positive_m=max(maximums) if maximums else None)
    return total


def inspect_exports(records, root):
    """Check all referenced files, array shape/dtype/mask agreement, and content duplicates."""
    root = Path(root)
    missing, errors, seen_content, duplicates = [], [], {}, []
    dimensions = Counter()
    total_bytes = 0
    for row in records:
        paths = {k: root / row[k] for k in ("rgb", "depth", "mask", "raw_depth", "raw_mask")}
        absent = [str(p) for p in paths.values() if not p.is_file()]
        missing.extend(absent)
        if absent:
            continue
        with Image.open(paths["rgb"]) as image:
            rgb = np.array(image)
        depth = np.load(paths["depth"], allow_pickle=False)
        raw = np.load(paths["raw_depth"], allow_pickle=False)
        mask = np.load(paths["mask"], allow_pickle=False)
        raw_mask = np.load(paths["raw_mask"], allow_pickle=False)
        dimensions[str(list(depth.shape))] += 1
        if (rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8 or depth.dtype != np.float32
                or raw.dtype != np.float32 or mask.dtype != np.bool_ or raw_mask.dtype != np.bool_
                or raw.shape != depth.shape or mask.shape != depth.shape or raw_mask.shape != depth.shape):
            errors.append(dict(id=row["id"], error="shape/dtype mismatch"))
        elif (not np.array_equal(mask, np.isfinite(depth) & (depth > 0))
              or not np.array_equal(raw_mask, np.isfinite(raw) & (raw > 0))
              or np.any(depth[~mask] != 0) or np.any(raw[~raw_mask] != 0)):
            errors.append(dict(id=row["id"], error="mask/value inconsistency"))
        fingerprint = hashlib.sha256(rgb.tobytes() + depth.tobytes()).hexdigest()
        if fingerprint in seen_content:
            prior = seen_content[fingerprint]
            duplicates.append(dict(first=prior["id"], second=row["id"], cross_official_split=prior["official_split"] != row["official_split"]))
        else:
            seen_content[fingerprint] = row
        total_bytes += sum(p.stat().st_size for p in paths.values())
    return dict(checked_samples=len(records), checked_files=len(records) * 5,
                missing_files=missing, errors=errors, content_duplicates=duplicates,
                dimensions=dict(dimensions), exported_bytes=total_bytes,
                limitation="Exact RGB+depth content duplicates checked; near-duplicates require scene-aware/manual review")
