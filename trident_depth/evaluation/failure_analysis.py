"""Produce panels of measured worst cases without inventing explanations."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("count must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    # One record per unique sample/protocol/scale; prevents relabeling repeated methods as separate scenes.
    groups = {}
    for row in report["results"]:
        key = (row["sample_id"], row["protocol"], row["scale"])
        groups.setdefault(key, []).append(row)
    selected = sorted(groups.values(), key=lambda rows: max(r["metrics"]["rmse_m"] for r in rows), reverse=True)[:args.count]
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, rows in enumerate(selected):
        first = rows[0]
        with np.load(args.report.parent / first["output"]) as source:
            maps = [("Input", source["input_depth"]), ("Reference", source["ground_truth"])]
            rgb = source["rgb"] if "rgb" in source else None
        for row in rows:
            with np.load(args.report.parent / row["output"]) as source:
                maps.append((row["method"] + (" ablated" if row.get("ablations") else ""), source["prediction"]))
        panels = []
        if rgb is not None:
            panels.append(("RGB", Image.fromarray(rgb)))
        maximum = float(first["metrics"]["data_range_m"])
        for title, depth in maps:
            norm = np.clip(depth / maximum, 0, 1)
            panels.append((title, Image.fromarray(np.uint8(np.stack([norm, np.sqrt(norm), 1 - norm], axis=-1) * 255))))
        canvas = Image.new("RGB", (256 * 3, 220 * ((len(panels) + 2) // 3)), "#101817")
        draw = ImageDraw.Draw(canvas)
        for i, (title, image) in enumerate(panels):
            x, y = (i % 3) * 256, (i // 3) * 220
            draw.text((x + 8, y + 8), title, fill="white")
            image.thumbnail((240, 180))
            canvas.paste(image, (x + 8, y + 30))
        canvas.save(args.output / f"case_{index + 1}.png")
        summaries.append(dict(sample_id=first["sample_id"], protocol=first["protocol"], scale=first["scale"],
                              methods=rows, panel=f"case_{index + 1}.png",
                              explanation="Requires human inspection; no cause inferred automatically"))
    result = dict(source_label=report["label"], distinct_scene_count=len({r[0]["scene_id"] for r in selected}), cases=summaries,
                  limitation="Multiple scales of one scene are not independent real-world failure cases")
    (args.output / "cases.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} diagnostic panels; not a completed failure-case study")


if __name__ == "__main__":
    main()
