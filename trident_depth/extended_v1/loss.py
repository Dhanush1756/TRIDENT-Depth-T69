"""Metric-depth losses; supervised depth boundaries and false-edge penalty."""
import torch
from torch.nn import functional as F
from ..training.losses import combined_loss, masked_l1, masked_mean, depth_edge_weights


def boundary_targets(target,mask):
    # Both endpoints of valid depth differences >= 5 cm; no RGB edge labels.
    labels=(depth_edge_weights(target,mask,.05)>1).float()
    # Require a fully valid local neighbourhood so invalid holes cannot be edges.
    invalid=F.max_pool2d((~mask).float(),3,1,1)>0
    return labels, mask & ~invalid


def false_edge_loss(pred,target,mask):
    terms=[]
    for axis in (-1,-2):
        pair=mask.narrow(axis,1,mask.shape[axis]-1)&mask.narrow(axis,0,mask.shape[axis]-1)
        safe=torch.where(mask,target,torch.zeros_like(target))
        true= torch.diff(safe,dim=axis).abs()
        # Flat/slowly varying GT; penalize excess predicted gradient, not RGB.
        flat=pair & (true<.01)
        terms.append(masked_mean((torch.diff(pred,dim=axis).abs()-true).clamp_min(0),flat))
    return sum(terms)


def objective(output,batch,task,weights):
    # Explicit FP32 reductions and no uncertainty head/objective.
    out={k:v.float() for k,v in output.items()}
    target,mask=batch["target"].float(),batch["target_valid"]
    legacy={k:weights[k] for k in ("depth","gradient","edge","measurement")}
    legacy["uncertainty"]=0.0
    loss,terms=combined_loss(out,target,mask,weights=legacy,use_uncertainty=False)
    cfg=task["pipeline"]
    aligned=F.avg_pool2d(out["depth"],cfg["scale"],cfg["scale"]) if cfg["protocol"]=="dense_sr" else out["depth"]
    terms["measurement"]=masked_l1(aligned,batch["observations"].float(),batch["input_valid"])
    if cfg.get("noise_std_m",0)>0 and cfg["protocol"]!="dense_sr":
        terms["measurement"]=.25*terms["measurement"]
    labels,valid=boundary_targets(target,mask)
    terms["boundary"]=masked_mean(F.binary_cross_entropy_with_logits(out["boundary_logits"],labels,reduction="none"),valid)
    terms["false_edge"]=false_edge_loss(out["depth"],target,mask)
    loss=loss+sum(weights[k]*terms[k] for k in ("measurement","boundary","false_edge"))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite extended objective")
    return loss,terms
