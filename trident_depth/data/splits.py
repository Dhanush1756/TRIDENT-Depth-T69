"""Scene-disjoint manifests and deterministic scene-level splitting."""
import hashlib
from pathlib import Path

SPLITS = {"train", "validation", "test"}


def validate_records(records, root="."):
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a nonempty JSON list")
    ids, scenes, files = set(), {}, {}
    for row in records:
        required = {"id", "scene_id", "split", "depth", "depth_scale_to_m"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"each record requires {sorted(required)}")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("id", "scene_id", "depth", "split")):
            raise ValueError("id, scene_id, depth, split must be nonempty strings")
        if row["id"] in ids or row["split"] not in SPLITS:
            raise ValueError("duplicate id or invalid split (use train/validation/test)")
        ids.add(row["id"])
        scene, split = row["scene_id"], row["split"]
        if scene in scenes and scenes[scene] != split:
            raise ValueError(f"scene leakage: {scene}")
        scenes[scene] = split
        for key in ("depth", "rgb", "mask"):
            if row.get(key):
                path = str((Path(root) / row[key]).resolve()).casefold()
                if path in files and files[path] != split:
                    raise ValueError(f"file leakage: {row[key]}")
                files[path] = split
    return records


def assign_scene_splits(records, seed=69, train_fraction=0.8, validation_fraction=0.1):
    """For custom development only; official published splits take precedence."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("split fractions must leave a nonzero test fraction")
    scenes = sorted({r["scene_id"] for r in records}, key=lambda s: (hashlib.sha256(f"{seed}:{s}".encode()).digest(), s))
    n_train = int(len(scenes) * train_fraction)
    n_val = int(len(scenes) * validation_fraction)
    assignments = {s: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
                   for i, s in enumerate(scenes)}
    result = [dict(row, split=assignments[row["scene_id"]]) for row in records]
    return validate_records(result)
