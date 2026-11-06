"""Paired A1/B0 CPU/CUDA diagnostics. Never performs an optimizer step."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import random
import numpy as np
from pathlib import Path
import tempfile
import time
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
from ..models.residual_cnn import ResidualCNN
from ..models.checkpoint import save_checkpoint, load_checkpoint, resume_checkpoint
from .corrective import verify_configuration, make_model, corrective_loss
from .colab import audit_pilot_dataset, sha256
from .dataset import TorchDepthDataset
from .portable import select_device
from .state import seed_everything, capture_state
from .boundaries import BOUNDARY_EVALUATION
from .corrective_monitoring import fixed_training_probe
from .corrective_data import audit_validation_reference, REFERENCE_SHA256


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def synthetic_batch():
    # No random draws; asymmetric residual guarantees a useful head derivative.
    target = torch.linspace(1, 3, 32*40).reshape(1, 1, 32, 40)
    initial = target + .03
    return dict(rgb=torch.full((1,3,32,40), .5), rgb_present=torch.tensor([True]),
        initial=initial, observed_support=torch.ones_like(target, dtype=torch.bool),
        target=target, target_valid=torch.ones_like(target, dtype=torch.bool),
        observations=F.avg_pool2d(target,4,4),
        input_valid=torch.ones((1,1,8,10), dtype=torch.bool), sample_id=["synthetic_diagnostic"])


def probe(model, batch, pipeline, cfg, device):
    before = fingerprint(model)
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, terms, prediction = corrective_loss(model, batch, pipeline, device,
        weights=cfg["loss_weights"], return_prediction=True)
    assert torch.equal(prediction, batch["initial"].to(device).clamp_min(1e-6)), "identity failed"
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    loss.backward()
    norms = {name:None if p.grad is None else float(p.grad.norm()) for name,p in model.named_parameters()}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    head = sum(v or 0 for k,v in norms.items() if k.startswith("residual_head"))
    body = sum(v or 0 for k,v in norms.items() if not k.startswith(("residual_head", "log_scale_head", "boundary_head")))
    uncertainty = [v for k,v in norms.items() if k.startswith("log_scale_head")]
    assert head > 0, "residual head must receive reconstruction gradients"
    assert all(v is None for k,v in norms.items() if k.startswith("boundary_head"))
    if cfg["uncertainty_enabled"]:
        assert body > 0 and sum(v or 0 for v in uncertainty) > 0
    else:
        assert body == 0 and all(v is None for v in uncertainty)
    assert fingerprint(model) == before, "forward/backward changed parameters"
    return dict(sample_ids=batch["sample_id"], shape=list(prediction.shape),
        exact_identity=True, parameters_unchanged=True, finite_loss_and_gradients=True,
        gradient_norms=norms, diagnostic_loss=float(loss.detach()),
        components={k:float(v.detach()) for k,v in terms.items()},
        seconds=time.perf_counter()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)


def roundtrip(model, cfg, pipeline, manifest_hash, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    generator = torch.Generator().manual_seed(69)
    metadata = dict(trained=False, purpose="untrained preflight", torch_version=str(torch.__version__),
        split_manifest_sha256=manifest_hash, config=dict(experiment=cfg,pipeline=asdict(pipeline)),
        uncertainty_available=cfg["uncertainty_enabled"], boundary_head_supervised=False)
    state = capture_state(generator, 0, float("inf"), 0)
    expected_python, expected_numpy = random.random(), np.random.random()
    expected_rng = torch.rand(4)
    expected_cuda = torch.rand(4,device=device) if device.type=="cuda" else None
    expected_order = torch.randperm(672, generator=generator)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"untrained.pt"
        save_checkpoint(path,model,cfg["model"],metadata,optimizer,state,scheduler)
        loaded,_ = load_checkpoint(path, require_trained=False, device=device)
        assert fingerprint(loaded) == fingerprint(model)
        restored = make_model(cfg).to(device)
        opt = torch.optim.Adam(restored.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _:1.)
        result = resume_checkpoint(path,restored,opt,generator,manifest_sha256=manifest_hash,
            experiment=cfg,pipeline=asdict(pipeline),scheduler=sched)
        assert result["epoch"] == 0 and fingerprint(restored) == fingerprint(model)
        assert random.random()==expected_python and np.random.random()==expected_numpy
        assert torch.equal(torch.rand(4), expected_rng)
        if expected_cuda is not None:
            assert torch.equal(torch.rand(4,device=device),expected_cuda)
        assert torch.equal(torch.randperm(672,generator=generator), expected_order)
        assert opt.state_dict() == optimizer.state_dict() and sched.state_dict() == scheduler.state_dict()
    return dict(model_optimizer_scheduler_rng_roundtrip=True, optimizer_moments="empty: zero optimizer steps")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs",type=Path,default=Path("configs/corrective_v1"))
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("preflight report exists; choose a new path")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    device=select_device(args.device)
    audit=audit_pilot_dataset(args.manifest)
    reference=audit_validation_reference(args.manifest)
    configs={row:verify_configuration(args.configs/f"{row}.json") for row in ("A1","B0")}
    seed_everything(69)
    legacy=ResidualCNN(upsampling="nearest")
    legacy_state={k:v.clone() for k,v in legacy.state_dict().items()}
    expected_rng=torch.get_rng_state().clone()
    expected_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    del legacy
    results={}
    paired_hashes=[]
    for row,(cfg,pipeline) in configs.items():
        seed_everything(69)
        model=make_model(cfg)
        assert sum(p.numel() for p in model.parameters())==473059
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert all(torch.equal(v,legacy_state[k]) for k,v in model.state_dict().items() if not k.startswith("residual_head"))
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],expected_cuda))
        paired_hashes.append(fingerprint(model))
        model=model.to(device)
        dataset=TorchDepthDataset(args.manifest,pipeline,"train",initializer="bicubic")
        probes=[probe(model,synthetic_batch(),pipeline,cfg,device)]
        real_provenance=[]
        rows=json.loads(args.manifest.read_text())
        second=next(i for i,r in enumerate(dataset.source.records)
                    if r["scene_id"]!=dataset.source.records[0]["scene_id"])
        for index in (0,second):
            batch=default_collate([dataset[index]])
            probes.append(probe(model,batch,pipeline,cfg,device))
            entry=next(r for r in rows if r["id"]==batch["sample_id"][0])
            assert entry["split"]=="train"
            real_provenance.append(dict(id=entry["id"], scene=entry["scene_id"],
                files={key:dict(path=entry[key],sha256=sha256(args.manifest.parent/entry[key])) for key in ("rgb","depth","mask")}))
        results[row]=dict(configuration=cfg, experiment_sha256=sha256(args.configs/f"{row}.json"),
            pipeline=asdict(pipeline), pipeline_sha256=sha256(args.configs/cfg["pipeline_config"]),
            probes=probes, training_sample_provenance=real_provenance,
            fixed_training_probe=fixed_training_probe(model,default_collate([dataset[0]]),pipeline,cfg,device),
            checkpoint=roundtrip(model,cfg,pipeline,audit["manifest_sha256"],device))
        del model
    assert paired_hashes[0]==paired_hashes[1]
    report=dict(purpose="untrained compatibility checks, not efficacy evidence", optimizer_steps=0,
        device=str(device), torch_version=str(torch.__version__), cuda_verified=device.type=="cuda",
        gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        parameters=473059, paired_initial_tensor_sha256=paired_hashes[0],
        matching_non_head_tensors_and_constructor_rng=True, split_audit=audit,
        validation_reference=dict(sha256=REFERENCE_SHA256, frames=len(reference),
            rgb_depth_mask_file_hashes_verified=True, evaluated=False),
        official_test_pixels_loaded=0,boundary_evaluation=BOUNDARY_EVALUATION,experiments=results)
    root=Path(__file__).resolve().parents[2]
    report["source_files_sha256"]={p.relative_to(root).as_posix():sha256(p)
        for p in sorted((root/"trident_depth").rglob("*.py"))}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
