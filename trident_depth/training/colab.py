"""Dataset archive staging and CPU inference export, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import torch
from ..models.checkpoint import load_checkpoint, save_checkpoint
from .validation import audit_training_manifest

NYU_MANIFEST_SHA256 = "d1cdbcf0ff3ee96f2e0c173c3c4701ed3f5c526ab517ce3b988268c3c6a391f5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_archive(archive, destination, expected_sha256):
    """Extract to a new local cache; reject traversal, duplicates and symlinks."""
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != expected_sha256:
        raise ValueError("dataset archive SHA256 mismatch")
    if destination.exists():
        raise FileExistsError("cache exists; audit it or choose a new cache directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if sum(m.file_size for m in members) > shutil.disk_usage(destination.parent).free:
            raise ValueError("insufficient space for extracted dataset")
        seen = set()
        for member in members:
            name = member.filename
            target = (destination / name).resolve()
            if ("\\" in name or not target.is_relative_to(destination.resolve())
                    or name in seen or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("unsafe dataset archive member")
            seen.add(name)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            extracted = Path(temporary) / "dataset"
            extracted.mkdir()
            source.extractall(extracted)
            (extracted / ".archive_sha256").write_text(expected_sha256, encoding="ascii")
            extracted.rename(destination)
    return destination


def audit_pilot_dataset(manifest):
    """Check original split bytes and all referenced files without loading test pixels."""
    manifest = Path(manifest)
    if sha256(manifest) != NYU_MANIFEST_SHA256:
        raise ValueError("pilot requires the approved NYU development manifest")
    audit = audit_training_manifest(manifest)
    if audit["sample_counts"] != {"train": 672, "validation": 123, "test": 654}:
        raise ValueError("NYU split counts changed")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    missing = []
    for row in rows:
        for key in ("rgb", "depth", "mask"):
            if row.get(key) and not (manifest.parent / row[key]).is_file():
                missing.append((row["id"], key))
    if missing:
        raise ValueError(f"missing dataset files: {missing[:10]}")
    return audit


def export_checkpoint(source, destination):
    """Export existing trained weights in the application's native checkpoint format."""
    model, metadata = load_checkpoint(source, device="cpu", require_trained=True)
    experiment = metadata["config"]["experiment"]
    exported = dict(metadata, export_source_sha256=sha256(source), export_device="cpu")
    save_checkpoint(destination, model, experiment["model"], exported)
    # Exercise the same guarded loader used by local inference.
    load_checkpoint(destination, device="cpu", require_trained=True)
    return sha256(destination)


def best_checkpoint_path(directory):
    """Recover selection from a completed checkpoint rather than a separate pointer."""
    directory = Path(directory)
    complete = sorted(directory.glob("epoch_*.pt"))
    if not complete:
        raise ValueError("no completed training checkpoint exists")
    state = torch.load(complete[-1], map_location="cpu", weights_only=True)
    best_epoch = state["training_state"]["best_epoch"]
    selected = directory / f"epoch_{best_epoch:03d}.pt"
    if not selected.is_file():
        raise ValueError("best epoch checkpoint is missing")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(dict(checkpoint=str(args.destination), sha256=export_checkpoint(args.source, args.destination))))


if __name__ == "__main__":
    main()
