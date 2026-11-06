"""Lazy map-style dataset; works without PyTorch and with its later DataLoader."""
import json
from pathlib import Path
from .io import load_sample
from .degradation import degrade
from .splits import SPLITS, validate_records


class DepthDataset:
    def __init__(self, manifest, config, split="train"):
        self.manifest = Path(manifest)
        self.config = config
        if split not in SPLITS:
            raise ValueError("unknown split")
        with self.manifest.open(encoding="utf-8") as stream:
            records = validate_records(json.load(stream), self.manifest.parent)
        self.records = [row for row in records if row["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = load_sample(self.records[index], self.manifest.parent,
                             min_depth=self.config.min_depth, max_depth=self.config.max_depth)
        return degrade(sample, self.config)
