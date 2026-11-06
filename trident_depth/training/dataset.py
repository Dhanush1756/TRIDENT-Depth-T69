"""NumPy pipeline to Torch tensors, retaining observation and target masks."""
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data.dataset import DepthDataset
from ..classical.pipeline import reconstruct
from ..classical.interpolation import interpolate


class TorchDepthDataset(Dataset):
    def __init__(self, manifest, config, split, initializer="classical"):
        self.source = DepthDataset(manifest, config, split)
        if split not in {"train", "validation"}:
            raise ValueError("training datasets only expose train or validation")
        if initializer not in {"classical", "bicubic", "guided"}:
            raise ValueError("unsupported training initializer")
        self.initializer = initializer

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        initial, _, _ = reconstruct(sample.input_depth, sample.input_valid,
                                     sample.target.depth.shape, sample.target.rgb, method=self.initializer)
        _, support = interpolate(sample.input_depth, sample.input_valid, sample.target.depth.shape, "nearest")
        channel = lambda x: torch.from_numpy(np.ascontiguousarray(x))[None]
        rgb = (np.zeros((*sample.target.depth.shape, 3), np.uint8) if sample.target.rgb is None else sample.target.rgb)
        return dict(initial=channel(initial), observed_support=channel(support),
                    target=channel(sample.target.depth), target_valid=channel(sample.target.valid),
                    observations=channel(sample.input_depth), input_valid=channel(sample.input_valid),
                    rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255,
                    rgb_present=sample.target.rgb is not None, sample_id=sample.target.sample_id)
