"""Fetch a bounded, verified Open3D/Redwood sample for I/O checks only.

Not NYU/KITTI training data or benchmark ground truth. No Open3D install needed.
"""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from PIL import Image

URL = "https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/SampleRedwoodRGBDImages.zip"
MD5 = "43971c5f690c9cfc52dda8c96a0140ee"  # Upstream Open3D dataset descriptor.
LIMIT = 10 * 1024 * 1024


def main():
    root = Path("data/smoke_redwood")
    if root.exists():
        raise SystemExit(f"Refusing to overwrite {root}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read(LIMIT + 1)
    if len(payload) > LIMIT:
        raise ValueError("sample archive exceeds the 10 MiB safety limit")
    if hashlib.md5(payload).hexdigest() != MD5:
        raise ValueError("sample archive differs from upstream checksum")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        rgb_bytes = archive.read("color/00000.jpg")
        depth_bytes = archive.read("depth/00000.png")
    rgb = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
    depth = np.array(Image.open(io.BytesIO(depth_bytes)))
    if rgb.shape[:2] != depth.shape:
        raise ValueError("upstream pair is not aligned")
    # Explicit deterministic crop for strict dense SR. Never fill missing targets.
    size = 64
    crop = None
    for y in range(0, depth.shape[0] - size + 1, 8):
        for x in range(0, depth.shape[1] - size + 1, 8):
            tile = depth[y:y + size, x:x + size]
            if ((tile > 0) & (tile <= 10000)).all():
                crop = (y, x)
                break
        if crop is not None:
            break
    if crop is None:
        raise ValueError("no fully valid 64x64 crop found")
    root.mkdir(parents=True)
    (root / "rgb.jpg").write_bytes(rgb_bytes)
    (root / "depth.png").write_bytes(depth_bytes)
    y, x = crop
    Image.fromarray(rgb[y:y + size, x:x + size]).save(root / "rgb_crop.png")
    Image.fromarray(depth[y:y + size, x:x + size]).save(root / "depth_crop.png")
    rows = [dict(id="redwood-smoke-crop", scene_id="redwood-livingroom1", split="test",
                 rgb="rgb_crop.png", depth="depth_crop.png", depth_scale_to_m=0.001)]
    (root / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    provenance = dict(url=URL, archive_md5=MD5, archive_sha256=hashlib.sha256(payload).hexdigest(),
                      downloaded_bytes=len(payload), crop_yxhw=[y, x, size, size],
                      source="Open3D SampleRedwoodRGBDImages / Redwood living-room1",
                      use="I/O smoke check only; sensor reference, not benchmark ground truth",
                      depth_scale_to_m=0.001,
                      documentation="https://www.open3d.org/html/tutorial/geometry/rgbd_image.html")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
