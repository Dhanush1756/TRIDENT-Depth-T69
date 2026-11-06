"""A1/B0 invariants, serialization and gradients: zero optimizer steps."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from trident_depth.models.baseline import LightweightCNN
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.models.residual_v2 import ResidualCNNV2
from trident_depth.models.checkpoint import load_checkpoint, save_checkpoint, resume_checkpoint
from trident_depth.training.corrective import approved_configuration, verify_configuration, make_model, corrective_loss
from trident_depth.training.corrective_preflight import synthetic_batch, probe, fingerprint
from trident_depth.training.state import seed_everything, capture_state
from trident_depth.training.losses import combined_loss, masked_l1
from trident_depth.training.train import model_loss as legacy_loss
from trident_depth.training.corrective_train import main
from trident_depth.config import PipelineConfig


@pytest.fixture(autouse=True)
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("row", ["A1", "B0"])
def test_constructor_draws_and_non_head_tensors_match_legacy(row):
    seed_everything(69)
    legacy = ResidualCNN(upsampling="nearest")
    rng = torch.get_rng_state()
    seed_everything(69)
    model = make_model(approved_configuration(row))
    assert sum(p.numel() for p in model.parameters()) == 473059
    assert torch.equal(rng, torch.get_rng_state())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, torch.zeros_like(tensor) if name.startswith("residual_head") else legacy.state_dict()[name])


@pytest.mark.parametrize("row", ["A1", "B0"])
def test_identity_and_first_step_gradients_without_optimizer(row, monkeypatch):
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a,**kw:pytest.fail("optimizer step prohibited"))
    cfg = approved_configuration(row)
    seed_everything(69)
    model = make_model(cfg)
    report = probe(model, synthetic_batch(), PipelineConfig(), cfg, torch.device("cpu"))
    assert report["exact_identity"] and report["parameters_unchanged"]


def test_b0_never_executes_uncertainty_head():
    model = ResidualCNNV2(uncertainty_enabled=False)
    handle = model.log_scale_head.register_forward_hook(lambda *a:pytest.fail("B0 uncertainty head executed"))
    b = synthetic_batch()
    out = model(b["rgb"],b["initial"],b["observed_support"])
    assert "log_scale" not in out
    assert all(p.requires_grad for p in model.log_scale_head.parameters())
    handle.remove()


def test_paired_initial_tensors_loader_and_rng_match():
    hashes, draws = [], []
    for row in ("A1", "B0"):
        seed_everything(69)
        model = make_model(approved_configuration(row))
        hashes.append(fingerprint(model))
        draws.append((random.random(), np.random.rand(),torch.rand(4),
            torch.randperm(672,generator=torch.Generator().manual_seed(69))))
    assert hashes[0] == hashes[1] and draws[0][:2] == draws[1][:2]
    assert all(torch.equal(a,b) for a,b in zip(draws[0][2:],draws[1][2:]))


@pytest.mark.parametrize("row", ["A1", "B0"])
def test_objective_equals_legacy_reconstruction_with_only_uncertainty_removed(row):
    cfg = approved_configuration(row)
    model = make_model(cfg)
    batch = synthetic_batch()
    actual, terms = corrective_loss(model,batch,PipelineConfig(),"cpu",weights=cfg["loss_weights"])
    out = model(batch["rgb"],batch["initial"],batch["observed_support"])
    expected, _ = combined_loss(out,batch["target"],batch["target_valid"],
        weights=cfg["loss_weights"],use_uncertainty=row=="A1")
    measurement = masked_l1(torch.nn.functional.avg_pool2d(out["depth"],4,4),batch["observations"],batch["input_valid"])
    assert torch.equal(terms["measurement"],measurement)
    torch.testing.assert_close(actual, expected+.2*measurement,rtol=0,atol=0)
    if row == "A1":
        legacy, _ = legacy_loss(model,batch,PipelineConfig(),"cpu",weights=cfg["loss_weights"])
        assert torch.equal(actual,legacy)


@pytest.mark.parametrize("row", ["A1", "B0"])
@pytest.mark.parametrize("empty", [False,True])
def test_masks_nan_targets_and_empty_support_finite(row, empty):
    cfg = approved_configuration(row)
    batch = synthetic_batch()
    batch["target_valid"][...,:8,:] = False
    batch["input_valid"][...,:2,:] = False
    if empty:
        batch["target_valid"].zero_();batch["input_valid"].zero_()
    batch["target"][~batch["target_valid"]] = float("nan")
    batch["observations"][~batch["input_valid"]] = float("nan")
    model = make_model(cfg)
    loss,terms = corrective_loss(model,batch,PipelineConfig(),"cpu",weights=cfg["loss_weights"])
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if empty:
        assert loss == 0


@pytest.mark.parametrize("row", ["A1", "B0"])
def test_serialization_resume_preserves_learned_head_moments_scheduler_rng(tmp_path,row):
    cfg=approved_configuration(row); pipeline=asdict(PipelineConfig())
    model=make_model(cfg)
    # Synthetic nonzero state tests loading, NOT an optimizer update or trained result.
    with torch.no_grad():
        model.residual_head.weight.fill_(.002);model.residual_head.bias.fill_(.003)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    p=next(model.parameters())
    opt.state[p]=dict(step=torch.tensor(1.),exp_avg=torch.full_like(p,.25),exp_avg_sq=torch.full_like(p,.5))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scheduler.last_epoch=1  # synthetic completed-epoch counter; no scheduler/optimizer step
    generator=torch.Generator().manual_seed(69)
    seed_everything(69)
    state=capture_state(generator,1,.2,1)
    state["history"]=[dict(epoch=1,purpose="synthetic unit fixture")]
    metadata=dict(trained=False,split_manifest_sha256="fixture",torch_version=str(torch.__version__),
        config=dict(experiment=cfg,pipeline=pipeline))
    path=tmp_path/"unit.pt"
    save_checkpoint(path,model,cfg["model"],metadata,opt,state,scheduler)
    expected=(random.random(),np.random.rand(),torch.rand(4),torch.randperm(672,generator=generator))
    loaded,_=load_checkpoint(path,require_trained=False)
    assert loaded.configuration()==model.configuration() and fingerprint(loaded)==fingerprint(model)
    other_opt=torch.optim.Adam(loaded.parameters(),lr=.1)
    other_sched=torch.optim.lr_scheduler.LambdaLR(other_opt,lambda _:1.)
    result=resume_checkpoint(path,loaded,other_opt,generator,manifest_sha256="fixture",experiment=cfg,pipeline=pipeline,scheduler=other_sched)
    assert result["epoch"]==1 and result["history"]==state["history"]
    assert fingerprint(loaded)==fingerprint(model)
    assert other_sched.state_dict()==scheduler.state_dict()
    assert other_opt.param_groups[0]["lr"]==1e-4
    assert torch.equal(other_opt.state[next(loaded.parameters())]["exp_avg"],opt.state[p]["exp_avg"])
    actual=(random.random(),np.random.rand(),torch.rand(4),torch.randperm(672,generator=generator))
    assert actual[:2]==expected[:2] and all(torch.equal(a,b) for a,b in zip(actual[2:],expected[2:]))
    with pytest.raises(ValueError,match="decoder"):
        wrong=make_model(approved_configuration("B0" if row=="A1" else "A1"))
        resume_checkpoint(path,wrong,other_opt,generator,manifest_sha256="fixture",experiment=cfg,pipeline=pipeline,scheduler=other_sched)
    with pytest.raises(ValueError,match="untrained"):
        load_checkpoint(path)


@pytest.mark.parametrize("model,name,options", [(ResidualCNN,"residual_cnn",{"upsampling":"nearest"}),
    (ResidualCNN,"residual_cnn",{}),(LightweightCNN,"lightweight_cnn",{})])
def test_legacy_checkpoint_prediction_and_format_unchanged(tmp_path,model,name,options):
    original=model(**options).eval();b=synthetic_batch()
    expected=original(b["rgb"],b["initial"],b["observed_support"])
    path=tmp_path/"legacy.pt";save_checkpoint(path,original,name,dict(trained=False))
    stored=torch.load(path,weights_only=True)
    assert stored["model_configuration"]==({"upsampling":original.upsampling} if name=="residual_cnn" else {})
    loaded,_=load_checkpoint(path,require_trained=False)
    actual=loaded(b["rgb"],b["initial"],b["observed_support"])
    assert set(actual)==set(expected)
    for key in expected:
        assert torch.equal(actual[key],expected[key])


def test_configs_frozen_scope_and_launch_gated(tmp_path):
    for row in ("A1","B0"):
        cfg,_=verify_configuration(f"configs/corrective_v1/{row}.json")
        assert cfg==approved_configuration(row)
        cfg["epochs"]=3
        path=tmp_path/"changed.json";path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError,match="frozen"):
            verify_configuration(path)
    for row in ("A2","A3","B1","B4","C1","S1"):
        with pytest.raises(ValueError):approved_configuration(row)
    with pytest.raises(ValueError):ResidualCNNV2(alpha=.1)
    with pytest.raises(SystemExit):main(["--experiment","missing.json"])


def test_version_identity_and_missing_configuration_rejected(tmp_path):
    model=ResidualCNNV2(uncertainty_enabled=False)
    with pytest.raises(ValueError,match="identity"):
        save_checkpoint(tmp_path/"bad.pt",model,"residual_cnn",dict(trained=False))
    path=tmp_path/"new.pt"
    save_checkpoint(path,model,"residual_cnn_v2",dict(trained=False))
    with pytest.raises(FileExistsError):
        save_checkpoint(path,model,"residual_cnn_v2",dict(trained=False))
    data=torch.load(path,weights_only=True);data.pop("model_configuration")
    corrupt=tmp_path/"fixture_missing.pt";torch.save(data,corrupt)
    with pytest.raises(ValueError,match="complete"):
        load_checkpoint(corrupt,require_trained=False)


def test_disabled_uncertainty_cannot_drive_hybrid():
    from trident_depth.inference import InferenceEngine
    engine=InferenceEngine();engine.model=ResidualCNNV2(uncertainty_enabled=False)
    with pytest.raises(ValueError,match="unavailable"):
        engine.run(np.ones((4,4),np.float32),np.ones((4,4),bool),(16,16),method="hybrid")


@pytest.mark.parametrize("row",["A1","B0"])
def test_fixed_probe_preserves_parameters_rng_and_grads(row):
    from trident_depth.training.corrective_monitoring import fixed_training_probe
    cfg=approved_configuration(row);seed_everything(69);model=make_model(cfg)
    before=fingerprint(model);rng=torch.get_rng_state().clone()
    result=fixed_training_probe(model,synthetic_batch(),PipelineConfig(),cfg,"cpu")
    assert fingerprint(model)==before and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert (result["uncertainty_training_probe"] is None)==(row=="B0")
    assert result["optimizer_steps"]==0
    assert result["per_term"]["depth"]["shared_enc1_weight"]["norm"]==0


def test_new_notebook_compiles_and_disabled_launches_are_inert():
    import ast
    nb=json.loads(Path("notebooks/colab_corrective_A1_B0.ipynb").read_text())
    code=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
    for c in nb["cells"]:
        if c["cell_type"]=="code":
            ast.parse("".join(c["source"]))
            assert c["execution_count"] is None and c["outputs"]==[]
    assert "TRAIN_A1_APPROVED = False" in code[0] and "TRAIN_B0_APPROVED = False" in code[0]
    assert "EXPORT_APPROVED = False" in code[0]
    helper=next(c for c in code if "def launch_control" in c)
    namespace={}
    exec(compile(helper,"disabled-launch-helper","exec"),namespace)
    for row in ("A1","B0"):
        namespace["launch_control"](row,False)
    setup=next(c for c in code if "TORCH_BEFORE =" in c)
    old=json.loads(Path("notebooks/colab_dense4_custom_pilot.ipynb").read_text())
    original=next("".join(c["source"]) for c in old["cells"] if "TORCH_BEFORE =" in "".join(c["source"]))
    assert setup==original
    assert "trident-venv" not in "\n".join(code)
    assert any('"--device", "cuda"' in c and "corrective_preflight" in c for c in code)


def test_contract_rejects_changed_objective_or_dense_measurement_weight():
    cfg=approved_configuration("B0");model=make_model(cfg)
    wrong=dict(cfg["loss_weights"],measurement=0)
    with pytest.raises(ValueError,match="objective weights"):
        corrective_loss(model,synthetic_batch(),PipelineConfig(),"cpu",weights=wrong)


def test_reference_audit_rejects_validation_id_changes_before_pixel_reads(tmp_path):
    from trident_depth.training.corrective_data import audit_validation_reference
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps([dict(id="wrong",split="validation")]))
    with pytest.raises(ValueError,match="IDs/order"):
        audit_validation_reference(manifest)


def test_b0_backbone_reconstruction_path_connected_after_nonzero_head_fixture():
    cfg=approved_configuration("B0");model=make_model(cfg)
    with torch.no_grad():model.residual_head.weight.fill_(.001)
    loss,_=corrective_loss(model,synthetic_batch(),PipelineConfig(),"cpu",weights=cfg["loss_weights"])
    loss.backward()
    assert model.enc1[0].weight.grad.norm()>0
    assert all(p.grad is None for p in model.log_scale_head.parameters())


def test_pipeline_values_are_frozen_independently_of_canonical_file(tmp_path):
    configs=tmp_path/"corrective_v1";configs.mkdir()
    phase5=tmp_path/"phase5";phase5.mkdir()
    experiment=configs/"A1.json"
    experiment.write_text(json.dumps(approved_configuration("A1")))
    pipeline=json.loads(Path("configs/phase5/dense_sr_4x_pipeline.json").read_text())
    pipeline["max_depth"]=5
    (phase5/"dense_sr_4x_pipeline.json").write_text(json.dumps(pipeline))
    with pytest.raises(ValueError,match="pipeline differs"):
        verify_configuration(experiment)
