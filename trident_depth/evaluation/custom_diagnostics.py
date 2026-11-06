"""Read-only custom pilot diagnostics; no optimizer, training, or test pixels."""
import argparse
import io
import json
import math
from pathlib import Path
import platform
import zipfile
import numpy as np
import PIL
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from ..config import PipelineConfig
from ..models.residual_cnn import ResidualCNN
from ..training.colab import audit_pilot_dataset, sha256
from ..training.dataset import TorchDepthDataset
from ..training.losses import combined_loss, masked_l1, depth_edge_weights
from ..training.state import seed_everything
from ..training.boundaries import boundary_f1_batch
from ..training.validation import validation_errors
from ..training.train import model_loss
from .pilot_audit import require, tensor_hash


def independent_losses(pred, target, mask, log_scale, observations, input_mask, scale=4):
    """Float64 NumPy reference using explicit eligible pixels/neighbour pairs."""
    p, t, s = (np.asarray(v, np.float64) for v in (pred, target, log_scale))
    m = np.asarray(mask, bool)
    error = np.abs(p-t)
    edge_weights = np.ones_like(t)
    gradient = 0.
    for axis in (0, 1):
        left = (slice(None,-1),slice(None)) if axis == 0 else (slice(None),slice(None,-1))
        right = (slice(1,None),slice(None)) if axis == 0 else (slice(None),slice(1,None))
        pairs = m[left] & m[right]
        difference = np.abs((p[right]-p[left]) - (t[right]-t[left]))
        gradient += float(difference[pairs].mean()) if pairs.any() else 0.
        hit = (np.abs(t[right]-t[left]) >= .05) & pairs
        edge_weights[left] += hit
        edge_weights[right] += hit
    h,w = p.shape
    measurement = p.reshape(h//scale,scale,w//scale,scale).mean(axis=(1,3))
    clipped = np.clip(s, -6, 6)
    return dict(depth=float(error[m].mean()), gradient=gradient, edge=float((error*edge_weights)[m].mean()),
        uncertainty=float((np.exp(-clipped)*error+clipped)[m].mean()),
        measurement=float(np.abs(measurement-observations)[input_mask].mean()))


def calibration_summary(sample):
    """Approximate ranks/quantiles on a deterministic spatial phase-mixed sample."""
    error, b, residual, initial_error = sample.T
    order = np.argsort(b, kind="stable")
    bins = []
    for indices in np.array_split(order,10):
        bins.append(dict(count=len(indices), predicted_b_m=float(b[indices].mean()),
            observed_mae_m=float(error[indices].mean()), error_over_b=float((error[indices]/b[indices]).mean())))
    rank_correlation = float(spearmanr(b,error).statistic) if np.ptp(b)>0 and np.ptp(error)>0 else None
    correction_correlation = float(np.corrcoef(residual,-initial_error)[0,1]) if np.ptp(residual)>0 and np.ptp(initial_error)>0 else None
    return dict(sample_pixels=len(sample), sampling="every 67th flattened valid pixel; start=frame index modulo 67",
        scale_quantiles_m={str(q):float(np.quantile(b,q)) for q in (0,.01,.1,.5,.9,.99,1)},
        correction_abs_quantiles_m={str(q):float(np.quantile(np.abs(residual),q)) for q in (.1,.5,.9,.99,1)},
        spearman_b_vs_abs_error=rank_correlation,
        residual_vs_ideal_correction_correlation=correction_correlation,
        decile_bins=bins)


def output_gradient_probe(output, batch, weights):
    """Differentiate loss outputs only, without changing model state."""
    with torch.inference_mode(False), torch.enable_grad():
        p = output["depth"].detach().clone().requires_grad_(True)
        s = output["log_scale"].detach().clone().requires_grad_(True)
        target, mask = batch["target"].clone(), batch["target_valid"].clone()
        _, terms = combined_loss(dict(depth=p,log_scale=s),target,mask,weights=weights)
        terms["measurement"] = masked_l1(F.avg_pool2d(p,4,4),batch["observations"].clone(),batch["input_valid"].clone())
        grads = {k:torch.autograd.grad(weights[k]*value,p,retain_graph=True)[0] for k,value in terms.items()}
        non_uncertainty = sum(v for k,v in grads.items() if k != "uncertainty")
        return dict(weighted_depth_gradient_l2={k:float(v.norm()) for k,v in grads.items()},
            uncertainty_to_other_gradient_norm_ratio=float(grads["uncertainty"].norm()/non_uncertainty.norm().clamp_min(1e-30)),
            uncertainty_other_gradient_cosine=float(F.cosine_similarity(grads["uncertainty"].flatten(),non_uncertainty.flatten(),dim=0)),
            log_scale_gradient_mean=float(torch.autograd.grad(weights["uncertainty"]*terms["uncertainty"],s)[0].mean()))


def model_gradient_probe(model, batch, pipeline, weights):
    with torch.enable_grad():
        loss, _ = model_loss(model,batch,pipeline,torch.device("cpu"),weights=weights)
        named = list(model.named_parameters())
        grads = torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    return {name:dict(connected=g is not None, norm=None if g is None else float(g.norm()),
                      finite=None if g is None else bool(torch.isfinite(g).all())) for (name,_),g in zip(named,grads)}


def bicubic_loss_reference(manifest, pipeline, weights):
    """Same validation loss operators applied to the unchanged bicubic depth."""
    dataset=TorchDepthDataset(manifest,pipeline,"validation",initializer="bicubic")
    records=[]
    roundoff=0.
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0):
            _,terms=combined_loss(dict(depth=batch["initial"]),batch["target"],batch["target_valid"],weights=weights)
            terms["measurement"]=masked_l1(F.avg_pool2d(batch["initial"],4,4),batch["observations"],batch["input_valid"])
            records.append({k:float(v) for k,v in terms.items()})
            roundoff=max(roundoff,float((F.avg_pool2d(batch["target"],4,4)-batch["observations"]).abs().max()))
    means={key:float(np.mean([row[key] for row in records])) for key in weights}
    return dict(frames=len(records),terms=means,weighted_terms={k:weights[k]*v for k,v in means.items()},
                positive_reconstruction_loss=sum(weights[k]*v for k,v in means.items() if k!="uncertainty"),
                target_area_pool_vs_observation_max_float32_roundoff_m=roundoff)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",required=True,type=Path)
    parser.add_argument("--environment",required=True,type=Path)
    parser.add_argument("--manifest",type=Path,default=Path("data/nyu/converted/manifest_development.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/custom_diagnostics"))
    parser.add_argument("--report-dir",type=Path,default=Path("docs/custom_diagnostics"))
    args=parser.parse_args(argv)
    require(not args.output.exists() and not args.report_dir.exists(),"diagnostic destinations must be new")
    torch.set_num_threads(2)
    torch.set_grad_enabled(True)
    seed_everything(69)
    initialized=ResidualCNN(upsampling="nearest").eval()
    initial_state={k:v.detach().clone() for k,v in initialized.state_dict().items()}
    protection_path=Path("artifacts/custom_diagnostics_protected_before.json")
    if protection_path.exists():
        protected=json.loads(protection_path.read_text())
    else:
        protected_paths=list(Path("trident_depth").rglob("*.py"))+list(Path("configs").rglob("*.json"))
        protected_paths+=list(Path("artifacts/nyu/validation_baselines").rglob("results.json"))+[args.archive,args.environment]
        protected={str(p):sha256(p) for p in protected_paths if p.is_file()}
    environment=json.loads(args.environment.read_text())
    require(sha256("artifacts/colab/trident_custom_pilot_source.zip")==environment["source_id"],"training source archive mismatch")
    require(sha256("artifacts/colab/nyu_converted.zip")==environment["dataset_archive_sha256"],"training dataset archive mismatch")
    with zipfile.ZipFile("artifacts/colab/trident_custom_pilot_source.zip") as source:
        require(all(Path(n).read_bytes()==source.read(n) for n in source.namelist() if n.startswith("trident_depth/") and n.endswith(".py")),"current code differs from pilot source")
    prefix="pilot_dense4_residual_seed69/"
    with zipfile.ZipFile(args.archive) as archive:
        require(archive.testzip() is None,"run archive CRC failed")
        run=json.loads(archive.read(prefix+"run.json"))
        history=[json.loads(line) for line in archive.read(prefix+"history.jsonl").decode().splitlines() if line]
        pointer=json.loads(archive.read(prefix+"best_checkpoint.json"))
        comparison=json.loads(archive.read(prefix+"validation_comparison/comparison.json"))
        comparison_frames=json.loads(archive.read(prefix+"validation_comparison/per_frame.json"))
        states={e:torch.load(io.BytesIO(archive.read(prefix+f"epoch_{e:03d}.pt")),map_location="cpu",weights_only=True) for e in (1,2)}
        checkpoint_hashes={e:__import__('hashlib').sha256(archive.read(prefix+f"epoch_{e:03d}.pt")).hexdigest() for e in (1,2)}
    require(run["environment_record"]==environment and sha256(args.environment)==run["environment_record_sha256"],"environment record mismatch")
    audit=audit_pilot_dataset(args.manifest)
    require(audit==run["split_audit"],"development split mismatch")
    require(sha256("configs/phase5/pilot_dense4_residual.json")==environment["experiment_sha256"] and sha256("configs/phase5/dense_sr_4x_pipeline.json")==environment["pipeline_sha256"],"config hashes differ")
    require([r["epoch"] for r in history]==[1,2],"unexpected pilot history")
    best=min(history,key=lambda r:r["validation_rmse_m"])
    require(pointer==dict(epoch=best["epoch"],checkpoint=f"epoch_{best['epoch']:03d}.pt",validation_rmse_m=best["validation_rmse_m"]),"selection pointer mismatch")
    require(comparison["best_custom_epoch"]==best["epoch"] and comparison["custom_checkpoint_sha256"]==checkpoint_hashes[best["epoch"]],"comparison selected wrong checkpoint")
    pipeline=PipelineConfig(**run["config"]["pipeline"])
    weights=run["config"]["experiment"]["loss_weights"]
    dataset=TorchDepthDataset(args.manifest,pipeline,"validation",initializer="bicubic")
    require(len(dataset)==len(comparison_frames)==123,"validation count mismatch")
    models, wiring={},{}
    for epoch,state in states.items():
        require(state["metadata"]==dict(run,trained=True,epochs=epoch) and state["training_state"]["history"]==history[:epoch],"checkpoint metadata/history mismatch")
        require(state["model_name"]=="residual_cnn" and state["model_configuration"]==dict(upsampling="nearest"),"architecture mismatch")
        model=ResidualCNN(upsampling="nearest").eval()
        model.load_state_dict(state["model_state"],strict=True)
        require(sum(p.numel() for p in model.parameters())==473059,"parameter count mismatch")
        names=[n for n,_ in model.named_parameters()]
        group=state["optimizer_state"]["param_groups"][0]
        inactive=[names[i] for i,param_id in enumerate(group["params"]) if param_id not in state["optimizer_state"]["state"]]
        require(all(float(v["step"])==672*epoch for v in state["optimizer_state"]["state"].values()),"optimizer steps mismatch")
        wiring[epoch]=dict(missing_adam_state=inactive,unchanged_from_seed69=[n for n,v in state["model_state"].items() if torch.equal(v,initial_state[n])],
            dense_protocol_channel_unchanged=torch.equal(state["model_state"]["enc1.0.weight"][:,6],initial_state["enc1.0.weight"][:,6]))
        wiring[epoch]["boundary_head_unchanged_between_epochs"]=all(torch.equal(states[1]["model_state"][n],states[2]["model_state"][n]) for n in ("boundary_head.weight","boundary_head.bias"))
        wiring[epoch]["boundary_head_seed69_max_abs_difference"]={n:float((state["model_state"][n]-initial_state[n]).abs().max()) for n in ("boundary_head.weight","boundary_head.bias")}
        protocol_state=state["optimizer_state"]["state"][group["params"][names.index("enc1.0.weight")]]
        wiring[epoch]["dense_protocol_channel_moment_max"]={k:float(protocol_state[k][:,6].abs().max()) for k in ("exp_avg","exp_avg_sq")}
        models[epoch]=model
    sample_ids=set(r["id"] for r in json.loads(Path("data/nyu/converted/manifest_validation_subset.json").read_text()))
    summaries={e:dict(count=0,terms=[],records=[],samples=[],gradients=[],
        sums={k:0. for k in ("s","b","raw_s","abs_error","ratio","residual","abs_residual","residual2","initial_error2","e0_r","error2","improved","worse","nll","oracle_gap","low_clamp","high_clamp","depth_clamp","within50","within90","within95")},
        regions={k:dict(count=0,abs_error=0.,initial_abs_error=0.,b=0.,residual2=0.,error2_delta=0.) for k in ("edge","nonedge")},
        extrema=dict(raw_min=float("inf"),raw_max=-float("inf")),initialization=[]) for e in (1,2)}
    per_frame=[]
    module_probes={}
    independent_max={key:0. for key in weights}
    nll_distribution_delta=0.
    for index,batch in enumerate(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)):
        row=dataset.source.records[index]
        require(row["official_split"]=="train" and row["id"]==comparison_frames[index]["id"] and tensor_hash(batch["target_valid"])==comparison_frames[index]["mask_sha256"],"frame/mask identity mismatch")
        baseline_reference=json.loads(Path("docs/pilot_dense4/per_frame.json").read_text())[index]
        require(all(sha256(args.manifest.parent/row[k])==baseline_reference["source_files_sha256"][k] for k in ("rgb","depth","mask")),"validation data changed")
        gt,mask,initial=(batch[k][0,0].numpy() for k in ("target","target_valid","initial"))
        frame=dict(id=row["id"],scene_id=row["scene_id"],valid_pixels=int(mask.sum()),epochs={})
        with torch.inference_mode():
            initial_diagnostic=None
            if row["id"] in sample_ids:
                o=initialized(batch["rgb"],batch["initial"],batch["observed_support"],4,"dense_sr")
                residual=o["residual"][0,0].numpy()[mask]
                initial_diagnostic=dict(id=row["id"],mean_residual_m=float(residual.mean()),abs_residual_m=float(np.abs(residual).mean()),
                    initial_residual_rmse_m=float(np.sqrt(np.mean(residual.astype(np.float64)**2))),bicubic_mae_m=float(np.abs(initial[mask]-gt[mask]).mean()))
            for epoch,model in models.items():
                raw=[]
                hook=model.log_scale_head.register_forward_hook(lambda _,__,value:raw.append(value.detach()))
                output=model(batch["rgb"],batch["initial"],batch["observed_support"],4,"dense_sr")
                hook.remove()
                loss,terms=combined_loss(output,batch["target"],batch["target_valid"],weights=weights)
                terms["measurement"]=masked_l1(F.avg_pool2d(output["depth"],4,4),batch["observations"],batch["input_valid"])
                loss=loss+weights["measurement"]*terms["measurement"]
                reference_terms=independent_losses(output["depth"][0,0].numpy(),gt,mask,output["log_scale"][0,0].numpy(),batch["observations"][0,0].numpy(),batch["input_valid"][0,0].numpy())
                for key in weights:
                    independent_max[key]=max(independent_max[key],abs(reference_terms[key]-float(terms[key])))
                distribution=-torch.distributions.Laplace(output["depth"],output["log_scale"].exp()).log_prob(batch["target"])+0.
                distribution=distribution[batch["target_valid"]].mean()-math.log(2)
                nll_distribution_delta=max(nll_distribution_delta,abs(float(distribution-terms["uncertainty"])))
                p=output["depth"][0,0].numpy().astype(np.float64)
                s=output["log_scale"][0,0].numpy().astype(np.float64)
                raw_s=raw[0][0,0].numpy().astype(np.float64)
                e=p-gt
                e0=initial.astype(np.float64)-gt
                r=p-initial
                b=np.exp(s)
                ae=np.abs(e)
                nll=ae/b+s
                oracle_s=np.log(np.clip(ae,np.exp(-6),np.exp(6)))
                edge=depth_edge_weights(batch["target"],batch["target_valid"])[0,0].numpy()>1
                summary=summaries[epoch]
                summary["count"]+=int(mask.sum())
                arrays=dict(s=s,b=b,raw_s=raw_s,abs_error=ae,ratio=ae/b,residual=r,abs_residual=np.abs(r),residual2=r*r,initial_error2=e0*e0,e0_r=e0*r,error2=e*e,
                    improved=ae<np.abs(e0),worse=ae>np.abs(e0),nll=nll,oracle_gap=nll-(ae/np.exp(oracle_s)+oracle_s),
                    low_clamp=raw_s<=-6,high_clamp=raw_s>=6,depth_clamp=(initial+output["residual"][0,0].numpy())<=1e-6,
                    within50=ae<=b*math.log(2),within90=ae<=b*math.log(10),within95=ae<=b*math.log(20))
                for key,value in arrays.items():summary["sums"][key]+=float(value[mask].sum())
                summary["extrema"]["raw_min"]=min(summary["extrema"]["raw_min"],float(raw_s[mask].min()))
                summary["extrema"]["raw_max"]=max(summary["extrema"]["raw_max"],float(raw_s[mask].max()))
                for region,active in (("edge",mask&edge),("nonedge",mask&~edge)):
                    totals=summary["regions"][region]
                    totals["count"]+=int(active.sum())
                    for key,array in dict(abs_error=ae,initial_abs_error=np.abs(e0),b=b,residual2=r*r,error2_delta=e*e-e0*e0).items():totals[key]+=float(array[active].sum())
                values=np.stack([ae[mask],b[mask],r[mask],e0[mask]],axis=1)[index%67::67].astype(np.float32)
                summary["samples"].append(values)
                squared,absolute,count=validation_errors(output["depth"],batch["target"],batch["target_valid"])
                boundary=boundary_f1_batch(output["depth"],batch["target"],batch["target_valid"])[0]
                record=dict(rmse_m=math.sqrt(squared/count),mae_m=absolute/count,boundary_f1=boundary,loss=float(loss),
                    terms={k:float(v) for k,v in terms.items()},mean_b_m=float(b[mask].mean()),mean_log_b=float(s[mask].mean()),
                    mean_abs_correction_m=float(np.abs(r[mask]).mean()),correction_rmse_m=float(np.sqrt(np.mean(r[mask]**2))),
                    bicubic_rmse_m=float(np.sqrt(np.mean(e0[mask]**2))),bicubic_mae_m=float(np.abs(e0[mask]).mean()),
                    improved_pixel_fraction=float((ae[mask]<np.abs(e0[mask])).mean()))
                summary["terms"].append(record["terms"])
                summary["records"].append(record)
                frame["epochs"][epoch]=record
                if initial_diagnostic is not None:
                    summary["initialization"].append(initial_diagnostic)
                    summary["gradients"].append(dict(id=row["id"],**output_gradient_probe(output,batch,weights)))
        if index==0:
            module_probes={epoch:model_gradient_probe(model,batch,pipeline,weights) for epoch,model in models.items()}
        per_frame.append(frame)
        if (index+1)%10==0:print(f"Diagnosed {index+1}/123 validation frames, both checkpoints",flush=True)
    results={}
    args.output.mkdir(parents=True)
    for epoch,summary in summaries.items():
        count=summary["count"]
        means={k:v/count for k,v in summary["sums"].items()}
        sample=np.concatenate(summary.pop("samples"))
        np.savez_compressed(args.output/f"epoch_{epoch:03d}_sample.npz",values=sample,columns=np.array(["abs_error_m","laplace_b_m","correction_m","bicubic_signed_error_m"]))
        records=summary["records"]
        result=dict(pixels=count,means=means,extrema=summary["extrema"],calibration=calibration_summary(sample),
            terms={k:float(np.mean([v[k] for v in summary["terms"]])) for k in weights},
            weighted_terms={k:weights[k]*float(np.mean([v[k] for v in summary["terms"]])) for k in weights},
            rmse_m=math.sqrt(means["error2"]),mae_m=means["abs_error"],boundary_f1=float(np.mean([v["boundary_f1"] for v in records])),
            validation_loss=float(np.mean([v["loss"] for v in records])),residual_energy_identity=dict(
                bicubic_mse=means["initial_error2"],correction_mse=means["residual2"],twice_error_correction_cross_term=2*means["e0_r"],
                resulting_mse=means["error2"]),regions={k:{name:value/v["count"] for name,value in v.items() if name!="count"}|dict(count=v["count"]) for k,v in summary["regions"].items()},
            output_gradient_probes=summary["gradients"],module_gradient_probe=module_probes[epoch],wiring=wiring[epoch],
            untrained_seed69_reference=summary["initialization"])
        for metric in ("rmse_m","mae_m","boundary_f1"):
            require(abs(result[metric]-history[epoch-1]["validation_"+metric])<1e-6,"CPU replay differs from recorded "+metric)
        require(abs(result["validation_loss"]-history[epoch-1]["validation_loss"])<1e-6,"loss replay mismatch")
        result["cpu_minus_recorded"]={k:result[k]-history[epoch-1]["validation_"+k] for k in ("rmse_m","mae_m","boundary_f1")}
        results[epoch]=result
    require(max(independent_max.values())<2e-6 and nll_distribution_delta<2e-6,"independent loss verification failed")
    for path,digest in protected.items():require(sha256(path)==digest,"protected file changed: "+path)
    report=dict(status="diagnostics only; zero optimizer steps; no checkpoint writes; official test pixels excluded",
        archive_sha256=sha256(args.archive),environment=environment,run=run,history=history,checkpoint_sha256=checkpoint_hashes,
        selection_verified=True,selected_epoch=best["epoch"],original_comparison=comparison["aggregate"],
        local_environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pillow=PIL.__version__,device="cpu"),
        epochs=results,independent_loss_max_abs_delta=independent_max,torch_laplace_max_abs_delta=nll_distribution_delta,
        protected_files_unchanged=True,protected_files_sha256=protected,
        bicubic_loss_reference=bicubic_loss_reference(args.manifest,pipeline,weights),
        limitations=["No exact CUDA resume was attempted on CPU.","Rank/quantile/decile statistics use explicitly labelled deterministic samples; means, coverage, clamp rates and regional sums use all valid pixels.","Seed-69 initialization reconstructed under local PyTorch; equality of untouched heads/conditioning slices provides a cross-version check, not proof of identical initialization for every parameter.","Gradient probes use eight fixed validation frames at the output and one full validation frame for module connectivity. No optimizer is created or stepped."])
    args.report_dir.mkdir(parents=True)
    (args.report_dir/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    (args.report_dir/"per_frame.json").write_text(json.dumps(per_frame,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({e:dict(rmse_m=r["rmse_m"],mae_m=r["mae_m"],means=r["means"],weighted_terms=r["weighted_terms"],calibration=r["calibration"],wiring=r["wiring"]) for e,r in results.items()},indent=2))


if __name__=="__main__":
    main()
