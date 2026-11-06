"""Verify the approved two-epoch custom pilot without optimizer steps."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import torch
from ..config import PipelineConfig
from ..models.residual_cnn import ResidualCNN
from ..models.checkpoint import save_checkpoint, resume_checkpoint
from .colab import audit_pilot_dataset, sha256
from .dataset import TorchDepthDataset
from .portable import select_device
from .state import seed_everything, capture_state
from .train import model_loss
from .boundaries import BOUNDARY_EVALUATION


def verify_configuration(experiment):
    experiment = Path(experiment)
    cfg = json.loads(experiment.read_text())
    approved = json.loads((experiment.parent / "dense_sr_4x_residual_cnn.json").read_text())
    identity = lambda c: {k:v for k,v in c.items() if k not in {"epochs", "output"}}
    if identity(cfg) != identity(approved) or cfg["epochs"] != 2:
        raise ValueError("custom pilot must preserve the approved configuration and use exactly two epochs")
    if cfg["output"] != "../../artifacts/training/pilot_dense4_residual_seed69":
        raise ValueError("custom pilot must use its separate output directory")
    if (cfg["model"], cfg["decoder_upsampling"], cfg["initializer"], cfg["batch_size"], cfg["seed"], cfg["learning_rate"]) != ("residual_cnn", "nearest", "bicubic", 1, 69, 1e-4):
        raise ValueError("custom pilot architecture or hyperparameters differ")
    pipeline = PipelineConfig.load(experiment.parent / cfg["pipeline_config"])
    if pipeline.protocol != "dense_sr" or pipeline.scale != 4 or pipeline.target_size is not None:
        raise ValueError("custom pilot requires full-frame dense 4x")
    return cfg, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, help="optional diagnostic JSON, never an evaluation result")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("diagnostic output already exists")
    cfg, pipeline = verify_configuration(args.experiment)
    audit = audit_pilot_dataset(args.manifest)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    seed_everything(69)
    device = select_device(args.device)
    model = ResidualCNN(upsampling=cfg["decoder_upsampling"]).to(device)
    if sum(p.numel() for p in model.parameters()) != 473059:
        raise ValueError("custom architecture parameter count changed")
    if not all(p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("custom pilot requires FP32")
    dataset = TorchDepthDataset(args.manifest, pipeline, "train", initializer="bicubic")
    sample = dataset[0]
    batch = {k:v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] if k == "sample_id" else torch.tensor([v]) for k,v in sample.items()}
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    loss, terms = model_loss(model, batch, pipeline, device, weights=cfg["loss_weights"])
    loss.backward()  # Numerical diagnostic only; no optimizer steps.
    if not torch.isfinite(loss) or not all(torch.isfinite(v) for v in terms.values()):
        raise ValueError("nonfinite custom loss")
    active = [name for name, p in model.named_parameters() if p.grad is not None]
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("nonfinite custom gradient")
    if not any(n.startswith("log_scale_head") for n in active):
        raise ValueError("uncertainty head is disconnected")
    if any(n.startswith("boundary_head") for n in active):
        raise ValueError("approved loss unexpectedly supervises the boundary-confidence head")
    if not all(torch.equal(before[k], v) for k,v in model.state_dict().items()):
        raise ValueError("preflight changed model parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    generator = torch.Generator().manual_seed(69)
    metadata = dict(trained=False, purpose="untrained preflight serialization check", torch_version=str(torch.__version__),
                    split_manifest_sha256=audit["manifest_sha256"], config=dict(experiment=cfg, pipeline=asdict(pipeline)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "untrained_preflight.pt"
        save_checkpoint(path, model, "residual_cnn", metadata, optimizer, capture_state(generator, 0, float("inf"), 0), scheduler)
        restored = ResidualCNN(upsampling="nearest").to(device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.)
        state = resume_checkpoint(path, restored, restored_optimizer, generator, manifest_sha256=audit["manifest_sha256"],
            experiment=cfg, pipeline=asdict(pipeline), scheduler=restored_scheduler)
        if state["epoch"] != 0 or restored_scheduler.state_dict() != scheduler.state_dict():
            raise ValueError("preflight checkpoint state mismatch")
        if not all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()):
            raise ValueError("preflight checkpoint model mismatch")
    result = dict(purpose="untrained compatibility diagnostic; not benchmark results", optimizer_steps=0,
        parameters=473059, device=str(device), torch_version=str(torch.__version__),
        experiment_sha256=sha256(args.experiment), config=cfg, pipeline=asdict(pipeline), split_audit=audit,
        diagnostic_training_sample=sample["sample_id"], full_frame_shape=list(sample["target"].shape[-2:]),
        loss_and_active_gradients_finite=True, parameters_unchanged=True, checkpoint_roundtrip_passed=True,
        uncertainty_head_receives_gradients=True, boundary_confidence_head_supervised=False,
        boundary_evaluation=BOUNDARY_EVALUATION,
        cuda_verified=device.type == "cuda")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
