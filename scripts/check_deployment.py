"""HTTP smoke checks using synthetic inputs only; no scientific evaluation."""
import argparse
import base64
import io
import json
from pathlib import Path
from urllib.parse import urlsplit
import httpx
import numpy as np
from PIL import Image

SHA256 = "09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd"


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def check(client, require_learned=False):
    for route in ("/", "/static/app.js", "/static/style.css"):
        response = client.get(route)
        response.raise_for_status()
        assert response.content, route
    response = client.get("/api/health")
    response.raise_for_status()
    health = response.json()
    assert health["status"] == "ok"
    if require_learned:
        assert health["trained_checkpoint_loaded"], "learned checkpoint not loaded"
        assert health["checkpoint"]["checkpoint_sha256"] == SHA256
    depth = png(np.full((8, 8), 2000, np.uint16))
    rgb = png(np.full((32, 32, 3), 90, np.uint8))
    methods = ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"]
    if require_learned:
        methods += ["deep", "hybrid"]
    records = []
    for method in methods:
        r = client.post("/api/reconstruct", files={"depth": ("depth.png", depth, "image/png"),
                        "rgb": ("rgb.png", rgb, "image/png")},
                        data={"method": method, "scale": "4", "protocol": "dense_sr"})
        r.raise_for_status()
        result = r.json()
        assert result["units"] == "metres" and result["output_shape"] == [32, 32]
        assert result["metrics"] is None, "synthetic smoke must not claim accuracy"
        with np.load(io.BytesIO(base64.b64decode(result["output_npz"])), allow_pickle=False) as arrays:
            values = arrays["depth_m"]
            assert values.shape == (32, 32) and np.isfinite(values).all()
            assert arrays["valid"].shape == values.shape
        for key in ("input_preview", "output_preview", "output_png16"):
            assert result[key], key
            with Image.open(io.BytesIO(base64.b64decode(result[key]))) as image:
                image.verify()
        if method in ("deep", "hybrid"):
            assert result["checkpoint"]["checkpoint_sha256"] == SHA256
        records.append(dict(method=method, status="PASS", latency_ms=result["latency_ms"],
                            numeric_download="finite 32x32 metres NPZ"))
    assert client.post("/api/reconstruct").status_code == 422
    for fields in ({"scale": "3"}, {"method": "unsupported"}, {"depth_scale_to_m": "-1"}):
        r = client.post("/api/reconstruct", files={"depth": ("depth.png", depth)}, data=fields)
        assert r.status_code == 400, r.text
    r = client.post("/api/reconstruct", files={"depth": ("depth.png", b"not an image")})
    assert r.status_code == 400
    return dict(status="PASS", health=health, synthetic_only=True, datasets_accessed=False,
                checks=records, invalid_inputs="PASS", frontend_assets="PASS",
                browser_interaction="separate browser check required on public URL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--require-learned", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new report filename")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.username or url.password or url.query or url.fragment:
        parser.error("use a plain application URL without credentials/query/fragment")
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost")):
        parser.error("public deployments require HTTPS")
    if args.output.exists():
        parser.error("report already exists; choose a new filename")
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=300) as client:
        report = check(client, args.require_learned)
    report["url"] = args.url
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("PASS: synthetic HTTP smoke; not a scientific benchmark")


if __name__ == "__main__":
    main()
