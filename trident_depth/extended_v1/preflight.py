"""Real training-sample backward checks; never constructs or steps an optimizer."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import default_collate
from .data import DevelopmentData,ProtocolDataset
from .model import ExtendedTRIDENT
from .loss import objective
from .contract import read_config,source_hashes,fingerprint,environment_identity
from ..training.colab import sha256
from ..training.state import seed_everything
from ..training.portable import select_device


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--crop",type=int,help="CPU checks only; GPU qualification must use full resolution")
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("use a fresh preflight report path")
    cfg=read_config(args.config)
    device=select_device(args.device)
    if device.type=="cuda" and (args.crop or torch.cuda.get_device_name(device)!="Tesla T4"):
        p.error("GPU qualification requires full-resolution Tesla T4")
    torch.set_num_threads(2)
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=DevelopmentData(args.manifest,Path(__file__).resolve().parents[2]/"docs/pilot_dense4/per_frame.json")
    model=ExtendedTRIDENT().warm_start(args.checkpoint).to(device)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    results=[]
    with data.isolated():
        payload_hashes=data.verify_files()
        # Each task uses the SAME training row; no validation inference or tuning.
        for task in cfg["tasks"]:
            dataset=ProtocolDataset(data,[task],"train")
            batch=default_collate([dataset[0]])
            if args.crop:
                size=args.crop
                if size<8 or size%8:
                    p.error("CPU crop must be >=8 and divisible by 8")
                for k,v in batch.items():
                    if isinstance(v,torch.Tensor) and v.ndim==4:
                        s=task["pipeline"]["scale"] if k in {"observations","input_valid"} and task["pipeline"]["protocol"]=="dense_sr" else 1
                        batch[k]=v[...,:size//s,:size//s]
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            model.zero_grad(set_to_none=True)
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            started=time.perf_counter()
            out=model(batch["rgb"],batch["initial"],batch["observed_support"],task["pipeline"]["scale"],task["pipeline"]["protocol"])
            loss,terms=objective(out,batch,task,cfg["loss_weights"])
            loss.backward()
            for name,param in model.named_parameters():
                if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                    raise ValueError("missing/nonfinite gradient: "+name)
                if name.startswith("log_scale_head") and param.grad is not None:
                    raise ValueError("uncertainty graph must remain disabled")
            if device.type=="cuda":
                torch.cuda.synchronize()
            results.append(dict(task=task["name"],sample_id=batch["sample_id"][0],
                input_shape=list(batch["observations"].shape),target_shape=list(batch["target"].shape),
                loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                finite_gradients=True,boundary_head_gradient_norm=float(model.boundary_head.weight.grad.norm())))
        if not all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()):
            raise ValueError("preflight changed model weights")
        audit=data.audit()
    report=dict(status="PASS",version=cfg["version"],optimizer_steps=0,
                device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                torch_version=str(torch.__version__),environment=environment_identity(),full_resolution=args.crop is None,
                config_sha256=sha256(args.config),checkpoint_sha256=sha256(args.checkpoint),
                source_hashes=source_hashes(),dataset_fingerprint=fingerprint(payload_hashes),
                split_audit=audit,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
