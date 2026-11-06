"""Render selected saved classical predictions for training-preparation review."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trident_depth.evaluation.nyu_preparation import depth_preview

root = Path(__file__).resolve().parent.parent
out = root / "docs/training_preparation"
for protocol, scale in (("dense_sr",8), ("sparse_completion",4)):
    destination = out / f"prediction_review_{protocol}.png"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    report = json.loads((root / f"docs/nyu/baseline_results_{protocol}.json").read_text())
    rows = [r for r in report["results"] if r["scale"]==scale and r["sample_id"]=="nyu_00178"]
    panels = []
    with np.load(root / "artifacts/nyu/validation_baselines" / protocol / rows[0]["output"]) as data:
        panels += [("RGB",data["rgb"]),("Filled reference",depth_preview(data["ground_truth"],data["ground_truth_valid"])),
                   (f"Input / {protocol} / scale {scale}",depth_preview(data["input_depth"],data["input_valid"]))]
    for row in rows:
        with np.load(root / "artifacts/nyu/validation_baselines" / protocol / row["output"]) as data:
            panels.append((f"{row['method']} / RMSE {row['metrics']['rmse_m']:.5f} m",
                           depth_preview(data["prediction"], data["prediction_valid"])))
    canvas = Image.new("RGB",(960,810),"#101817")
    draw = ImageDraw.Draw(canvas)
    for i,(title,array) in enumerate(panels):
        x,y=(i%3)*320,(i//3)*270
        draw.text((x+5,y+7),title,fill="white")
        canvas.paste(Image.fromarray(array).resize((320,240)),(x,y+25))
    canvas.save(destination)
    print(destination.name)
