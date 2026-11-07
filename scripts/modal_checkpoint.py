"""Upload only the approved checkpoint to an immutable private Volume path."""
import argparse
import hashlib
from pathlib import Path

SHA256 = "09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd"
VOLUME_NAME = "trident-t69-private-models"
REMOTE = f"/extended_v1/{SHA256}/epoch_012.pt"


def verify(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("checkpoint file is required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise ValueError("checkpoint SHA256 mismatch; upload refused")
    return path


def upload(path):
    path = verify(path)  # No SDK lookup or network access before verification.
    import modal
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    # force=False refuses overwriting an existing file, even on repeated runs.
    with volume.batch_upload(force=False) as batch:
        batch.put_file(str(path), REMOTE)
    return dict(volume=VOLUME_NAME, path=REMOTE, sha256=SHA256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(upload(args.checkpoint))
