"""Multi-scale/sparse hybrid adapter, independent of frozen hybrid_v1_dense4."""
import time
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from ..classical.interpolation import interpolate
from ..classical.hole_filling import fill_holes
from ..classical.pipeline import reconstruct
from ..classical.edge_detection import boundary_refine
from ..classical.discontinuity import depth_edges


def project_dense(candidate,observations,scale):
    """Bounded area-mean projection, never an HR point-observation substitution."""
    low=np.asarray(observations,dtype=np.float64)
    if scale not in (2,4,8) or candidate.shape!=tuple(s*scale for s in low.shape):
        raise ValueError("dense projection shape/scale mismatch")
    if not np.isfinite(candidate).all() or not np.isfinite(low).all() or np.any((low<=0)|(low>10)):
        raise ValueError("dense observations require finite positive metres <=10")
    lower=float(np.nextafter(np.float32(1e-6),np.float32(np.inf)))
    h,w=low.shape
    blocks=np.clip(candidate,lower,10).astype(np.float64).reshape(h,scale,w,scale).transpose(0,2,1,3).reshape(h,w,scale*scale)
    left=lower-blocks.max(-1)
    right=10-blocks.min(-1)
    for _ in range(48):
        middle=(left+right)/2
        below=np.clip(blocks+middle[...,None],lower,10).mean(-1)<low
        left=np.where(below,middle,left)
        right=np.where(below,right,middle)
    values=np.clip(blocks+((left+right)/2)[...,None],lower,10)
    return values.reshape(h,w,scale,scale).transpose(0,2,1,3).reshape(candidate.shape).astype(np.float32)


def hybrid_refine(neural,boundary_probability,observations,mask,rgb,*,scale,protocol,noisy=False,method="joint_bilateral"):
    if method not in {"joint_bilateral","guided","classical"}:
        raise ValueError("unsupported existing classical method")
    n=np.clip(np.asarray(neural,dtype=np.float32),1.000001e-6,10)
    probability=np.asarray(boundary_probability)
    if probability.shape!=n.shape or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
        raise ValueError("boundary probabilities must align and lie in [0,1]")
    c,cm,_=reconstruct(observations,mask,n.shape,rgb,method)
    c,filled=fill_holes(c,cm)
    if not filled.all():
        raise ValueError("no observations to fill")
    g,_,_=boundary_refine(n,np.ones(n.shape,bool),rgb,.05)
    geometric=maximum_filter(depth_edges(n,np.ones(n.shape,bool),.05).astype(np.float32),size=3)
    # RGB guidance and learned head cannot authorize RGB-only depth edges.
    gate=geometric*probability
    fused=n+gate*(.15*(c-n)+.15*(g-n))
    fused=np.clip(fused,1.000001e-6,10).astype(np.float32)
    if protocol=="dense_sr":
        if noisy or not np.asarray(mask).all():
            raise ValueError("dense SR requires clean, fully observed LR block means")
        fused=project_dense(fused,observations,scale)
    elif protocol=="sparse_completion":
        if observations.shape!=n.shape or mask.shape!=n.shape:
            raise ValueError("completion measurements must be aligned at target resolution")
        # Hard retention only for explicitly clean observed points. Noisy points
        # have a soft consistency objective during training, no hard inference copy.
        if not noisy:
            fused=np.where(mask,observations,fused).astype(np.float32)
    else:
        raise ValueError("unknown protocol")
    return dict(depth=fused,classical=c,geometry=g,geometry_gate=gate,
                boundary_probability=probability,confidence_interpretation="supervised boundary probability; not calibrated uncertainty")


class ExtendedInference:
    def __init__(self,model,device="cpu"):
        self.model=model.to(device).eval()
        self.device=torch.device(device)

    def run(self,depth,mask,rgb,*,scale=4,protocol="dense_sr",noisy=False):
        started=time.perf_counter()
        depth,mask=np.asarray(depth),np.asarray(mask)
        if (depth.ndim!=2 or mask.shape!=depth.shape or mask.dtype!=np.bool_ or not mask.any()
                or not np.isfinite(depth[mask]).all() or np.any((depth[mask]<=0)|(depth[mask]>10))):
            raise ValueError("requires nonempty finite positive measured depth <=10m and bool mask")
        if scale not in (2,4,8) or protocol not in {"dense_sr","sparse_completion"}:
            raise ValueError("unsupported scale/protocol")
        size=tuple(x*scale for x in depth.shape) if protocol=="dense_sr" else depth.shape
        if rgb is not None and (rgb.shape!=(*size,3) or rgb.dtype!=np.uint8):
            raise ValueError("RGB must be registered target-resolution uint8")
        initial,valid=interpolate(depth,mask,size,"bicubic")
        initial,_=fill_holes(initial,valid)
        _,support=interpolate(depth,mask,size,"nearest")
        channel=lambda a:torch.from_numpy(np.ascontiguousarray(a))[None,None].to(self.device)
        guide=None if rgb is None else torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))[None].to(self.device).float()/255
        with torch.inference_mode():
            out=self.model(guide,channel(initial),channel(support),scale,protocol)
        neural=out["depth"].cpu().numpy()[0,0]
        boundary=out["boundary_confidence"].cpu().numpy()[0,0]
        result=hybrid_refine(neural,boundary,depth,mask,rgb,scale=scale,protocol=protocol,noisy=noisy)
        result.update(neural=neural,initial=initial,total_seconds=time.perf_counter()-started)
        return result
