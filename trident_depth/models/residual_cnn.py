"""Lightweight multiscale residual encoder-decoder with uncertainty heads."""
import torch
from torch import nn
from torch.nn import functional as F
from .inputs import pack_features


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())


class ResidualCNN(nn.Module):
    def __init__(self, upsampling="bilinear"):
        super().__init__()
        if upsampling not in {"bilinear", "nearest"}:
            raise ValueError("decoder upsampling must be bilinear or nearest")
        self.upsampling = upsampling
        self.enc1 = block(8, 32)
        self.enc2 = block(32, 64)
        self.bottleneck = block(64, 128)
        self.dec2 = block(192, 64)
        self.dec1 = block(96, 32)
        self.residual_head = nn.Conv2d(32, 1, 1)
        self.log_scale_head = nn.Conv2d(32, 1, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, rgb, initial, validity, scale=4, protocol="dense_sr"):
        features = pack_features(rgb, initial, validity, scale, protocol)
        if min(initial.shape[2:]) < 4:
            raise ValueError("model requires target height and width >=4")
        e1 = self.enc1(features)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        middle = self.bottleneck(F.avg_pool2d(e2, 2))
        options = dict(mode=self.upsampling)
        if self.upsampling == "bilinear":
            options["align_corners"] = False
        d2 = self.dec2(torch.cat([F.interpolate(middle, size=e2.shape[2:], **options), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], **options), e1], dim=1))
        residual = self.residual_head(d1)
        return dict(depth=torch.clamp(initial + residual, min=1e-6), residual=residual,
                    log_scale=self.log_scale_head(d1).clamp(-6, 6),
                    boundary_confidence=torch.sigmoid(self.boundary_head(d1)))
