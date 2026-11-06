"""Versioned exact-zero residual controls, exclusively A1 and B0."""
import torch
from torch.nn import functional as F
from .inputs import pack_features
from .residual_cnn import ResidualCNN


class ResidualCNNV2(ResidualCNN):
    def __init__(self, upsampling="nearest", alpha=1.0, uncertainty_enabled=True,
                 initialization="exact_zero_v1"):
        if upsampling != "nearest" or alpha != 1 or initialization != "exact_zero_v1":
            raise ValueError("V2 supports only approved exact-zero, alpha=1, nearest controls")
        if type(uncertainty_enabled) is not bool:
            raise ValueError("uncertainty_enabled must be boolean")
        # Consume every legacy constructor draw before changing the residual head.
        super().__init__(upsampling=upsampling)
        self.alpha = float(alpha)
        self.uncertainty_enabled = uncertainty_enabled
        self.initialization = initialization
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def configuration(self):
        return dict(upsampling=self.upsampling, alpha=self.alpha,
                    uncertainty_enabled=self.uncertainty_enabled, initialization=self.initialization)

    def forward(self, rgb, initial, validity, scale=4, protocol="dense_sr"):
        if scale != 4 or protocol != "dense_sr":
            raise ValueError("V2 controls require dense 4x")
        features = pack_features(rgb, initial, validity, scale, protocol)
        if min(initial.shape[2:]) < 4:
            raise ValueError("model requires target height and width >=4")
        e1 = self.enc1(features)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        middle = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(middle, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        raw = self.residual_head(d1)
        correction = self.alpha * raw
        output = dict(depth=(initial + correction).clamp_min(1e-6), residual=correction,
                      raw_residual=raw, boundary_confidence=torch.sigmoid(self.boundary_head(d1)))
        # B0 does not execute this head: no graph, no zero gradient/Adam moments.
        if self.uncertainty_enabled:
            output["log_scale"] = self.log_scale_head(d1).clamp(-6, 6)
        return output
