"""Bind future A1/B0 validation to the immutable 123-frame pilot reference."""
import json
from pathlib import Path
from .colab import sha256

REFERENCE_SHA256 = "bba902eb2f3e9cc5bd66393b3768ffd5777a2d063989e75d2ff0cc762e83b138"


def audit_validation_reference(manifest):
    manifest=Path(manifest)
    reference=Path(__file__).resolve().parents[2]/"docs/pilot_dense4/per_frame.json"
    if sha256(reference)!=REFERENCE_SHA256:
        raise ValueError("historical validation reference SHA256 mismatch")
    frames=json.loads(reference.read_text())
    rows=[r for r in json.loads(manifest.read_text()) if r["split"]=="validation"]
    if len(rows)!=123 or [r["id"] for r in rows]!=[r["id"] for r in frames]:
        raise ValueError("validation IDs/order differ from the original pilot")
    for row,frame in zip(rows,frames):
        if row["official_split"]!="train" or row["scene_id"]!=frame["scene_id"]:
            raise ValueError("validation scene/official split mismatch")
        if any(sha256(manifest.parent/row[key])!=frame["source_files_sha256"][key]
               for key in ("rgb","depth","mask")):
            raise ValueError("validation file differs from original pilot")
    return frames
