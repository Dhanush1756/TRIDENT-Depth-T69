"""NYU official index validation and deterministic, whole-room development holdout."""
from collections import Counter, defaultdict
import re
import numpy as np


def official_membership(split_data, sample_count):
    membership = {}
    for key, label in (("trainNdxs", "train"), ("testNdxs", "test")):
        if key not in split_data:
            raise ValueError(f"official split missing {key}")
        values = np.asarray(split_data[key])
        if values.dtype.kind not in "ui" or values.size == 0:
            raise ValueError("official split indices must be nonempty integer arrays")
        for original_index in values.ravel():
            original_index = int(original_index)
            if not 1 <= original_index <= sample_count or original_index in membership:
                raise ValueError("duplicate, overlapping or out-of-range official index")
            membership[original_index] = label
    if len(membership) != sample_count:
        raise ValueError("official splits do not cover every source sample")
    return membership


def room_group(scene):
    """Conservatively group recordings like bedroom_0001a/b as the same room."""
    return re.sub(r"(?<=\d)[a-z]$", "", scene)


def development_holdout(records, fraction=0.1):
    """Preserve source split; only official training rooms may become validation."""
    if not 0 < fraction < 0.5:
        raise ValueError("validation scene fraction must lie in (0, .5)")
    room_partitions, room_types = {}, {}
    by_type = defaultdict(set)
    for row in records:
        room = row["scene_id"]
        official = row["official_split"]
        if official not in {"train", "test"}:
            raise ValueError("unknown official partition")
        if room in room_partitions and room_partitions[room] != official:
            raise ValueError(f"official train/test room overlap: {room}")
        room_partitions[room] = official
        kind = row["scene_type"]
        if room in room_types and room_types[room] != kind:
            raise ValueError(f"inconsistent room type: {room}")
        room_types[room] = kind
        if official == "train":
            by_type[kind].add(room)
    held = set()
    for kind, rooms in sorted(by_type.items()):
        ordered = sorted(rooms)
        if len(ordered) < 2:
            continue  # Never remove the only training room of a scene type.
        count = min(len(ordered) - 1, max(1, int(len(ordered) * fraction + .5)))
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        held.update(ordered[i] for i in positions)
    if not held:
        raise ValueError("not enough official training rooms for a scene-disjoint validation holdout")
    output = [dict(row, split="validation" if row["scene_id"] in held else row["official_split"]) for row in records]
    return output, dict(rule="Within each scene type, sorted training room IDs; evenly spaced positions; never hold out the only room",
                        requested_scene_fraction=fraction, validation_rooms=sorted(held),
                        split_samples=dict(Counter(r["split"] for r in output)),
                        official_split_unchanged=True, note="Derived development validation; not an official NYU validation split")


def select_validation_subset(records, count=8):
    if count < 1:
        raise ValueError("subset size must be positive")
    groups = defaultdict(list)
    type_samples = Counter()
    for row in records:
        if row["split"] == "validation":
            groups[row["scene_id"]].append(row)
            type_samples[row["scene_type"]] += 1
    by_type = defaultdict(list)
    for scene, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["source_index_1based"])
        by_type[rows[0]["scene_type"]].append(rows[len(rows) // 2])
    types = sorted(by_type, key=lambda t: (-type_samples[t], t))
    selected = []
    while len(selected) < count:
        advanced = False
        for kind in types:
            if by_type[kind] and len(selected) < count:
                selected.append(by_type[kind].pop(0))
                advanced = True
        if not advanced:
            break
    if not selected:
        raise ValueError("no validation samples")
    return selected
