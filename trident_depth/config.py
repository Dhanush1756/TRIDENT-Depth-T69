"""Strict JSON configuration shared by the CLI and dataset wrapper."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    protocol: str = "dense_sr"
    scale: int = 4
    seed: int = 69
    min_depth: float = 0.0
    max_depth: float = 100.0
    target_size: tuple[int, int] | None = None
    density: float = 0.05
    sampling: str = "random"
    noise_std_m: float = 0.01
    hole_fraction: float = 0.1

    def __post_init__(self):
        if self.protocol not in {"dense_sr", "sparse_completion"}:
            raise ValueError("protocol must be dense_sr or sparse_completion")
        if type(self.scale) is not int or self.scale not in {2, 4, 8}:
            raise ValueError("scale must be 2, 4, or 8")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        values = (self.min_depth, self.max_depth, self.density,
                  self.noise_std_m, self.hole_fraction)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("configuration numbers must be finite")
        if not 0 <= self.min_depth < self.max_depth:
            raise ValueError("require 0 <= min_depth < max_depth (metres)")
        if not 0 <= self.density <= 1 or not 0 <= self.hole_fraction <= 1:
            raise ValueError("density and hole_fraction must lie in [0, 1]")
        if self.noise_std_m < 0 or self.sampling not in {"random", "grid"}:
            raise ValueError("invalid noise_std_m or sampling pattern")
        if self.target_size is not None:
            if len(self.target_size) != 2 or any(type(v) is not int or v <= 0 for v in self.target_size):
                raise ValueError("target_size must be [height, width], positive integers")

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if "target_size" in values and values["target_size"] is not None:
            values["target_size"] = tuple(values["target_size"])
        return cls(**values)
