"""Application dispatch only; frozen scientific/training implementations stay intact."""
from pathlib import Path
import numpy as np
from trident_depth.inference import InferenceEngine

EPOCH12_SHA256 = "09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd"
CLASSICAL = ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"]


class ApplicationEngine:
    def __init__(self, checkpoint=None, device="cpu", model_format="legacy"):
        if model_format not in {"legacy", "extended_v1"}:
            raise ValueError("TRIDENT_MODEL_FORMAT must be legacy or extended_v1")
        self.device, self.model, self.metadata = device, None, None
        self.extended = model_format == "extended_v1"
        if not self.extended or checkpoint is None:
            self.legacy = InferenceEngine(checkpoint, device)
            self.model, self.metadata = self.legacy.model, self.legacy.metadata
            return
        from trident_depth.training.colab import sha256
        if sha256(Path(checkpoint)) != EPOCH12_SHA256:
            raise ValueError("extended serving requires the exact approved epoch-12 checkpoint SHA256")
        from trident_depth.extended_v1.checkpoint import load
        self.model, payload = load(checkpoint, device)
        if payload.get("training_state", {}).get("epoch") != 12:
            raise ValueError("extended checkpoint must record epoch 12")
        self.model.requires_grad_(False)
        self.metadata = dict(model="trident_extended_v1", epoch=12,
                             checkpoint_sha256=EPOCH12_SHA256, precision="FP32")

    def capabilities(self):
        caps = {m:dict(protocols=["dense_sr", "sparse_completion"], scales=[2,4,8],
                       rgb_required=False) for m in CLASSICAL}
        if self.model is not None:
            b0 = not self.extended and getattr(self.model, "uncertainty_enabled", True) is False
            spec = dict(protocols=["dense_sr"] if b0 else ["dense_sr","sparse_completion"],
                        scales=[4] if b0 else [2,4,8],rgb_required=self.extended or b0,
                        max_depth_m=10 if self.extended else None)
            if self.extended:
                spec["protocol_scales"]={"dense_sr":[2,4,8],"sparse_completion":[4]}
            elif not b0:
                recorded=(self.metadata or {}).get("config",{}).get("pipeline",{})
                if recorded.get("protocol") in spec["protocols"] and recorded.get("scale") in spec["scales"]:
                    spec["protocols"]=[recorded["protocol"]]
                    spec["scales"]=[recorded["scale"]]
            caps["deep"] = spec
            if not b0: caps["hybrid"] = dict(spec)
        return caps

    def run(self, depth, mask, size, rgb=None, *, method="classical", scale=4,
            protocol="dense_sr", noisy=False):
        caps=self.capabilities()
        if method not in caps:
            raise ValueError("selected method is unavailable for the configured checkpoint")
        spec=caps[method]
        if protocol not in spec["protocols"] or scale not in spec["scales"]:
            raise ValueError("checkpoint does not support the selected protocol/scale")
        if spec["rgb_required"] and rgb is None:
            raise ValueError("this learned model requires registered RGB guidance")
        if method not in {"deep","hybrid"} or not self.extended:
            engine = self.legacy if hasattr(self,"legacy") else InferenceEngine()
            return engine.run(depth,mask,size,rgb,method=method,scale=scale,protocol=protocol,noisy=noisy)
        if np.any(depth[mask] > 10) or min(size) < 4:
            raise ValueError("extended model requires target dimensions >=4 and depth <=10 metres")
        if protocol=="sparse_completion" and scale!=4:
            raise ValueError("extended completion uses conditioning scale 4 and keeps input resolution")
        if protocol=="dense_sr" and (noisy or not mask.all()):
            raise ValueError("learned dense SR requires clean, fully valid LR depth; use target-grid completion for noisy/incomplete inputs")
        from trident_depth.extended_v1.inference import ExtendedInference
        if method=="hybrid":
            result=ExtendedInference(self.model,self.device).run(depth,mask,rgb,scale=scale,protocol=protocol,noisy=noisy)
            maps={k:result[k] for k in ("initial","neural","classical","geometry")}
            out=result["depth"]
        else:
            # Same initializer and feature tensors as original tensor_sample,
            # without constructing or supplying any ground-truth tensor.
            import torch
            from trident_depth.classical.interpolation import interpolate
            from trident_depth.classical.hole_filling import fill_holes
            initial,valid=interpolate(depth,mask,size,"bicubic")
            initial,_=fill_holes(initial,valid)
            _,support=interpolate(depth,mask,size,"nearest")
            channel=lambda a:torch.from_numpy(np.ascontiguousarray(a))[None,None].to(self.device)
            guide=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))[None].to(self.device).float()/255
            with torch.inference_mode():
                result=self.model(guide,channel(initial),channel(support),scale,protocol)
            out=result["depth"].float().cpu().numpy()[0,0]
            maps=dict(initial=initial,neural=out)
        return dict(depth=out,valid=np.isfinite(out)&(out>0),maps=maps,checkpoint=self.metadata)
