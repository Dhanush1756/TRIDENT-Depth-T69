"""B0-compatible backbone with multi-protocol dispatch and supervised boundary logits."""
import torch
from torch.nn import functional as F
from ..models.residual_v2 import ResidualCNNV2
from ..models.inputs import pack_features
from ..hybrid_v1_dense4 import load_b0


class ExtendedTRIDENT(ResidualCNNV2):
    identity = "trident_extended_v1"

    def __init__(self):
        super().__init__(uncertainty_enabled=False)
        # Retain the tensor for exact B0 transfer, but exclude this unused head
        # from optimization. There is no uncertainty forward graph or objective.
        self.log_scale_head.requires_grad_(False)

    def forward(self, rgb, initial, validity, scale=4, protocol="dense_sr"):
        features = pack_features(rgb, initial, validity, scale, protocol)
        if min(initial.shape[-2:]) < 4:
            raise ValueError("target dimensions must be at least four")
        e1 = self.enc1(features)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        middle = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(middle, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        # The depth addition and all losses remain FP32, including under autocast.
        residual = self.residual_head(d1).float()
        logits = self.boundary_head(d1).float()
        return dict(depth=(initial.float()+residual).clamp_min(1e-6), residual=residual,
                    boundary_logits=logits, boundary_confidence=logits.sigmoid())

    def warm_start(self, checkpoint):
        original, _ = load_b0(checkpoint, "cpu")  # SHA256 + original strict loader
        self.load_state_dict(original.state_dict(), strict=True)
        return self
