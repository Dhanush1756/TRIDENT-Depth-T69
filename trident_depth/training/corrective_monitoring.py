"""Fixed training-only gradient/calibration probes; no RNG draws or parameter updates."""
import torch
from .corrective import corrective_loss
from .losses import depth_edge_weights


def fixed_training_probe(model, batch, pipeline, cfg, device):
    loss, terms, prediction = corrective_loss(model,batch,pipeline,device,
        weights=cfg["loss_weights"],return_prediction=True)
    shared = model.enc1[0].weight
    names = ["depth","gradient","edge","measurement"]
    if model.uncertainty_enabled:
        names.append("uncertainty")
    reconstruction = sum(cfg["loss_weights"][k]*terms[k] for k in names if k!="uncertainty")
    result = dict(sample_ids=batch["sample_id"], optimizer_steps=0,
                  objective=float(loss.detach()), per_term={})
    for label, tensor in (("depth_output",prediction),("shared_enc1_weight",shared)):
        reference = torch.autograd.grad(reconstruction,tensor,retain_graph=True,allow_unused=True)[0]
        for name in names:
            gradient = torch.autograd.grad(cfg["loss_weights"][name]*terms[name],tensor,
                                           retain_graph=True,allow_unused=True)[0]
            norm = 0.0 if gradient is None else float(gradient.norm())
            refnorm = 0.0 if reference is None else float(reference.norm())
            cosine = (float((gradient.flatten()*reference.flatten()).sum()/(gradient.norm()*reference.norm()))
                      if norm>0 and refnorm>0 else None)
            result["per_term"].setdefault(name,{})[label]=dict(norm=norm,cosine_to_reconstruction=cosine)
    with torch.no_grad():
        if model.uncertainty_enabled:
            # A second deterministic forward supplies scale without retaining its graph.
            initial=batch["initial"].to(device)
            out=model(batch["rgb"].to(device),initial,batch["observed_support"].to(device))
            target=batch["target"].to(device);mask=batch["target_valid"].to(device)
            error=(prediction-target).abs()
            scale=out["log_scale"].exp()
            edges=depth_edge_weights(target,mask)>1
            groups={"all":mask,"edge":mask&edges,"flat":mask&~edges}
            # Deciles describe this fixed training sample, not independent calibration.
            if mask.any():
                thresholds=torch.quantile(scale[mask],torch.linspace(0,1,11,device=device))
                for i in range(10):
                    groups[f"scale_decile_{i+1}"]=mask&(scale>=thresholds[i])&(
                        (scale<=thresholds[i+1]) if i==9 else (scale<thresholds[i+1]))
            result["uncertainty_training_probe"]={}
            for name,selected in groups.items():
                count=int(selected.sum())
                result["uncertainty_training_probe"][name]=None if count==0 else dict(
                    count=count,mean_scale_m=float(scale[selected].mean()),mean_abs_error_m=float(error[selected].mean()),
                    coverage50=float((error[selected]<=scale[selected]*.69314718056).float().mean()),
                    coverage90=float((error[selected]<=scale[selected]*2.30258509299).float().mean()),
                    coverage95=float((error[selected]<=scale[selected]*2.99573227355).float().mean()))
        else:
            result["uncertainty_training_probe"]=None
    return result
