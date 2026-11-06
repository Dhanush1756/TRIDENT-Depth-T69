"""Real-file HTTP smoke verification against a running local server."""
import argparse
import base64
import io
import json
from pathlib import Path
import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--sample", type=Path, default=Path("artifacts/phase1/dense4/sample.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/http_smoke.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    with np.load(args.sample) as sample:
        depth = sample["input_depth"]
        from PIL import Image
        rgb_stream = io.BytesIO()
        Image.fromarray(sample["rgb"]).save(rgb_stream, format="PNG")
        depth_stream, gt_stream = io.BytesIO(), io.BytesIO()
        np.save(depth_stream, depth)
        np.save(gt_stream, sample["ground_truth"])
    with httpx.Client(base_url=args.url, timeout=60) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        page = client.get("/")
        page.raise_for_status()
        response = client.post("/api/reconstruct", files={
            "depth": ("depth.npy", depth_stream.getvalue()),
            "rgb": ("rgb.png", rgb_stream.getvalue()),
            "ground_truth": ("reference.npy", gt_stream.getvalue())},
            data={"scale": 4, "method": "classical", "depth_scale_to_m": 1, "gt_scale_to_m": 1})
        response.raise_for_status()
    result = response.json()
    with np.load(io.BytesIO(base64.b64decode(result["output_npz"]))) as output:
        assert output["depth_m"].shape == (64, 64)
        assert np.isfinite(output["depth_m"]).all()
        assert output["valid"].all()
    summary = dict(label="real_redwood_HTTP_smoke_only", http_status=response.status_code,
                   frontend_status=page.status_code, health=health.json(), method=result["method"],
                   output_shape=result["output_shape"], latency_ms=result["latency_ms"], metrics=result["metrics"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("label", "http_status", "frontend_status", "output_shape", "latency_ms")}))


if __name__ == "__main__":
    main()
