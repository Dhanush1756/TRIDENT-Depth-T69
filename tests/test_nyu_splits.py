import numpy as np
import pytest
from trident_depth.data.nyu_splits import official_membership, room_group, development_holdout, select_validation_subset


def test_official_matlab_indices():
    assert official_membership(dict(trainNdxs=np.array([[1], [3]]), testNdxs=np.array([[2], [4]])), 4) == {1:"train", 3:"train", 2:"test", 4:"test"}


@pytest.mark.parametrize("train,test", [([1, 1], [2, 3]), ([0, 1], [2, 3]), ([1], [3]), ([1, 2], [2, 3])])
def test_bad_official_indices_rejected(train, test):
    with pytest.raises(ValueError):
        official_membership(dict(trainNdxs=np.array(train), testNdxs=np.array(test)), 3)


def records():
    return [dict(id=f"{kind}-{scene}-{frame}", scene_id=f"{kind}_{scene:04d}",
                 source_scene=f"{kind}_{scene:04d}a", scene_type=kind, source_index_1based=i,
                 official_split="test" if scene == 5 else "train", split="test" if scene == 5 else "train")
            for i, (kind, scene, frame) in enumerate(( (k,s,f) for k in ("bedroom", "kitchen") for s in range(6) for f in range(2)), 1)]


def test_whole_room_holdout_deterministic_and_official_test_untouched():
    rows = records()
    first, metadata = development_holdout(rows)
    second, _ = development_holdout(list(reversed(rows)))
    assert {r["id"]:r["split"] for r in first} == {r["id"]:r["split"] for r in second}
    assert metadata["validation_rooms"] == ["bedroom_0000", "kitchen_0000"]
    assert all(r["split"] == "test" for r in first if r["official_split"] == "test")
    for scene in {r["scene_id"] for r in first}:
        assert len({r["split"] for r in first if r["scene_id"] == scene}) == 1
    selected = select_validation_subset(first)
    assert len(selected) == 2 and len({r["scene_type"] for r in selected}) == 2
    assert all(r["official_split"] == "train" and r["split"] == "validation" for r in selected)


def test_recording_suffixes_are_grouped_conservatively():
    assert room_group("bedroom_0001a") == room_group("bedroom_0001b") == "bedroom_0001"
    assert room_group("office") == "office"


def test_official_room_overlap_fails():
    rows = records()
    rows[1]["official_split"] = "test"
    with pytest.raises(ValueError, match="room overlap"):
        development_holdout(rows)
