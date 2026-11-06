"""Strict training-partition checks and reference-masked validation aggregates."""
from collections import Counter
import json
from pathlib import Path
import torch
from ..data.nyu_integrity import file_sha256
from ..data.splits import validate_records


def audit_training_manifest(manifest):
    manifest = Path(manifest)
    rows = validate_records(json.loads(manifest.read_text(encoding="utf-8")), manifest.parent)
    for row in rows:
        if "official_split" in row:
            expected = "test" if row["split"] == "test" else "train"
            if row["official_split"] != expected:
                raise ValueError("official test data cannot enter training or validation")
    counts = Counter(r["split"] for r in rows)
    if not counts["train"] or not counts["validation"]:
        raise ValueError("nonempty scene-disjoint train and validation splits are required")
    fingerprint = file_sha256(manifest)
    integrity_path = manifest.parent / "integrity_report.json"
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        expected = integrity.get("manifests_sha256", {}).get(manifest.name)
        if expected is not None and expected != fingerprint:
            raise ValueError("manifest differs from the verified NYU conversion")
    return dict(manifest_sha256=fingerprint, sample_counts=dict(counts),
                scene_counts={s:len({r['scene_id'] for r in rows if r['split']==s}) for s in counts},
                test_used_for_training_or_selection=False)


def validation_errors(prediction, target, mask):
    selected = mask.bool()
    error = prediction[selected].double() - target[selected].double()
    if not torch.isfinite(error).all():
        raise ValueError("nonfinite validation prediction or target")
    return float(error.square().sum()), float(error.abs().sum()), int(selected.sum())
