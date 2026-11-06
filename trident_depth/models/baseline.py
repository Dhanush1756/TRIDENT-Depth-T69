"""Explicit lightweight CNN baseline; not a reproduction of DKN or FDKN."""
from torch import nn
import torch
from .inputs import pack_features


class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, rgb, initial, validity, scale=4, protocol="dense_sr"):
        features = pack_features(rgb, initial, validity, scale, protocol)
        residual = self.layers(features)
        return {"depth": torch.clamp(initial + residual, min=1e-6), "residual": residual}
