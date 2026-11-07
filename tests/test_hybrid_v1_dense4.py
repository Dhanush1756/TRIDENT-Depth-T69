"""Hybrid-v1 synthetic tests; optional genuine B0 serialization check, never a benchmark."""
from dataclasses import replace, asdict
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from trident_depth.hybrid_v1_dense4 import (HybridConfig, HybridV1Dense4, B0_MEMBER,
    area_mean, project_area_means, normalized_blend, refine_candidates, load_b0, comparison_outputs)
from trident_depth.inference import InferenceEngine
from trident_depth.models.residual_v2 import ResidualCNNV2
from trident_depth.models.checkpoint import load_checkpoint
from trident_depth.evaluation.hybrid_v1_dense4 import main, frame_metrics, aggregate_frames
from trident_depth.evaluation import hybrid_v1_data as data_module


@pytest.fixture(autouse=True)
def few_threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def sample():
    low=np.linspace(1,3,24,dtype=np.float32).reshape(4,6)
    mask=np.ones_like(low,dtype=bool)
    rgb=np.zeros((16,24,3),np.uint8);rgb[:,12:]=170
    return low,mask,rgb


class DepthOnly(torch.nn.Module):
    def forward(self,rgb,initial,support,scale,protocol):
        class Output(dict):
            def __getitem__(self,key):
                if key!="depth":raise AssertionError("confidence output was accessed")
                return super().__getitem__(key)
        return Output(depth=initial+.017)


def unit_engine():
    # Synthetic interface fixture, never a deployment model or saved trained artifact.
    engine=object.__new__(HybridV1Dense4)
    engine.config=HybridConfig();engine.device=torch.device("cpu")
    engine.model=DepthOnly().eval();engine.metadata={"purpose":"synthetic unit fixture"}
    return engine


def test_engine_shapes_ranges_and_normalized_weights_without_confidence_outputs():
    d,m,rgb=sample();result=unit_engine().run(d,m,rgb)
    assert result["depth"].shape==(16,24) and result["depth"].dtype==np.float32
    assert result["valid"].all() and result["units"]=="metres"
    assert np.isfinite(result["depth"]).all() and result["depth"].min()>=1e-6 and result["depth"].max()<=10
    assert set(result["outputs"])=={"bicubic","b0","hybrid_v1_dense4"}
    w=result["weights"]["hybrid_v1_dense4"]
    np.testing.assert_allclose(w.sum(axis=0),1,atol=2e-7)
    assert (w>=0).all() and (w<=1).all()
    np.testing.assert_allclose(area_mean(result["depth"]),d,atol=2e-6,rtol=0)
    from trident_depth.classical.pipeline import reconstruct
    initial,_,_=reconstruct(d,m,(16,24),rgb,"bicubic")
    np.testing.assert_array_equal(result["outputs"]["b0"],initial+.017)
    np.testing.assert_array_equal(result["outputs"]["bicubic"],initial)


def test_partial_lr_support_and_masked_nonfinite_values_have_defined_outputs():
    d,m,rgb=sample();m[:,2:4]=False;d[~m]=np.nan
    result=unit_engine().run(d,m,rgb)
    assert result["partial_lr_input_unvalidated"] and np.isfinite(result["depth"]).all()
    np.testing.assert_allclose(area_mean(result["depth"])[m],d[m],atol=2e-6,rtol=0)


@pytest.mark.parametrize("change",["zero_valid","bad_mask","bad_rgb","rgb_missing","bad_range","nonfinite","bad_protocol","bad_scale"])
def test_invalid_inputs_and_unsupported_capabilities_rejected(change):
    d,m,rgb=sample();kwargs={}
    if change=="zero_valid":m[:]=False
    if change=="bad_mask":m=m.astype(np.float32)
    if change=="bad_rgb":rgb=rgb[:-1]
    if change=="rgb_missing":rgb=None
    if change=="bad_range":d[0,0]=10.1
    if change=="nonfinite":d[0,0]=np.nan
    if change=="bad_protocol":kwargs["protocol"]="sparse_completion"
    if change=="bad_scale":kwargs["scale"]=2
    with pytest.raises(ValueError):unit_engine().run(d,m,rgb,**kwargs)


@pytest.mark.parametrize("target",[1e-6,0.1,9.999,10.0])
def test_bounded_area_projection_preserves_measurements_at_limits(target):
    cfg=HybridConfig();d=np.full((2,3),target,np.float32);m=d>0
    y=np.linspace(-4,14,96,dtype=np.float32).reshape(8,12)
    result=project_area_means(y,d,m,cfg)
    assert result.dtype==np.float32 and np.isfinite(result).all()
    assert result.min()>=cfg.min_depth_m and result.max()<=cfg.max_depth_m
    np.testing.assert_allclose(area_mean(result),d,atol=2e-6,rtol=0)


def test_unobserved_blocks_are_not_projected_to_invented_measurements():
    cfg=HybridConfig();d=np.full((2,2),2,np.float32);m=d>0;m[0,1]=False;d[0,1]=np.nan
    candidate=np.full((8,8),3,np.float32)
    result=project_area_means(candidate,d,m,cfg)
    np.testing.assert_array_equal(result[:4,4:],candidate[:4,4:])
    np.testing.assert_allclose(area_mean(result)[m],2,atol=1e-6)


def test_rgb_texture_alone_does_not_create_geometry_changes():
    d=np.full((4,6),2,np.float32);m=d>0;rgb=np.zeros((16,24,3),np.uint8);rgb[:,::2]=255
    neural=np.full((16,24),2,np.float32)
    result=refine_candidates(neural,d,m,rgb)
    assert not result["reliability"]["geometry_gate"].any()
    np.testing.assert_array_equal(result["candidates"]["geometry"],neural)
    for v in result["outputs"].values():np.testing.assert_allclose(v,2,atol=1e-6)


def test_comparison_blends_share_projection_and_reproduce_timed_hybrid_bitwise():
    d,m,rgb=sample();result=unit_engine().run(d,m,rgb)
    outputs,weights,_=comparison_outputs(result["candidates"],result["reliability"]["scores"],d,m,HybridConfig())
    assert set(outputs)=={"b0_consistency_only","b0_plus_classical","b0_plus_geometry","hybrid_v1_dense4"}
    np.testing.assert_array_equal(outputs["hybrid_v1_dense4"],result["depth"])
    for name,v in outputs.items():
        np.testing.assert_allclose(area_mean(v),d,atol=2e-6,rtol=0)
        if name in weights:np.testing.assert_allclose(weights[name].sum(0),1,atol=2e-7)
    assert not weights["b0_plus_classical"][2].any()
    assert not weights["b0_plus_geometry"][1].any()


@pytest.mark.parametrize("invalid",["negative","nan","zero","shape"])
def test_invalid_fusion_scores_rejected(invalid):
    candidates=[np.ones((4,4))]*3;scores=[np.ones((4,4)) for _ in range(3)]
    if invalid=="negative":scores[0][0,0]=-1
    if invalid=="nan":scores[1][0,0]=np.nan
    if invalid=="zero":scores=[np.zeros((4,4))]*3
    if invalid=="shape":scores[0]=np.ones((2,4))
    with pytest.raises(ValueError):normalized_blend(candidates,scores,(1,1,1))


@pytest.mark.parametrize("kwargs",[{"scale":8},{"protocol":"sparse_completion"},{"prior_neural":0},
    {"local_tau_m":float("nan")},{"local_window":2},{"projection_iterations":2},{"max_depth_m":11}])
def test_configuration_is_versioned_and_rejects_bad_controls(kwargs):
    with pytest.raises(ValueError):HybridConfig(**kwargs)


def test_config_file_matches_defaults_and_legacy_guard_remains():
    assert HybridConfig.load("configs/hybrid_v1_dense4.json")==HybridConfig()
    engine=InferenceEngine();engine.model=ResidualCNNV2(uncertainty_enabled=False)
    d,m,rgb=sample()
    with pytest.raises(ValueError,match="uncertainty"):
        engine.run(d,m,(16,24),rgb,method="hybrid")


def test_checkpoint_hash_rejected_before_loading(tmp_path,monkeypatch):
    import trident_depth.hybrid_v1_dense4 as module
    path=tmp_path/"epoch_002.pt";path.write_bytes(b"unit fixture, not a checkpoint")
    monkeypatch.setattr(module,"load_checkpoint",lambda *a,**kw:pytest.fail("unverified bytes were loaded"))
    with pytest.raises(ValueError,match="SHA256"):load_b0(path)


def test_original_v2_serialization_remains_compatible(tmp_path):
    from trident_depth.models.checkpoint import save_checkpoint
    model=ResidualCNNV2(uncertainty_enabled=False).eval()
    with torch.no_grad():model.residual_head.bias.fill_(.003)
    path=tmp_path/"untrained_unit.pt"
    save_checkpoint(path,model,"residual_cnn_v2",dict(trained=False,purpose="synthetic serialization fixture"))
    restored,_=load_checkpoint(path,require_trained=False)
    assert restored.configuration()==model.configuration()
    d=torch.ones((1,1,8,12));rgb=torch.zeros((1,3,8,12));m=d.bool()
    with torch.inference_mode():
        assert torch.equal(model(rgb,d,m)["depth"],restored(rgb,d,m)["depth"])


@pytest.mark.skipif(not os.environ.get("TRIDENT_B0_AUDIT_ZIP"),reason="optional actual B0 archive is not distributed with source")
def test_actual_b0_checkpoint_strict_loading_and_prediction_parity_on_synthetic_input(tmp_path):
    path=Path(os.environ["TRIDENT_B0_AUDIT_ZIP"])
    with zipfile.ZipFile(path) as archive:raw=archive.read(B0_MEMBER)
    original,_=load_checkpoint(io.BytesIO(raw),device="cpu")
    model,_=load_b0(path,"cpu")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    d=torch.linspace(1,3,96).reshape(1,1,8,12);rgb=torch.full((1,3,8,12),.5);m=d.bool()
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():
        a=original(rgb,d,m)["depth"];b=model(rgb,d,m)["depth"]
    assert torch.equal(a,b)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    # Exercise the complete new engine against the original dataset preprocessing.
    from PIL import Image
    from torch.utils.data import default_collate
    from trident_depth.config import PipelineConfig
    from trident_depth.training.dataset import TorchDepthDataset
    gt=np.linspace(1,3,16*24,dtype=np.float32).reshape(16,24)
    rgb=np.full((16,24,3),128,np.uint8)
    np.save(tmp_path/"depth.npy",gt);Image.fromarray(rgb).save(tmp_path/"rgb.png")
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps([dict(id="synthetic_fixture",scene_id="unit_scene",split="validation",
        depth="depth.npy",rgb="rgb.png",depth_scale_to_m=1)]))
    ds=TorchDepthDataset(manifest,PipelineConfig(max_depth=10),"validation",initializer="bicubic")
    batch=default_collate([ds[0]])
    with torch.inference_mode():
        expected=original(batch["rgb"],batch["initial"],batch["observed_support"])["depth"][0,0].numpy()
    engine=HybridV1Dense4(path,device="cpu")
    result=engine.run(batch["observations"][0,0].numpy(),batch["input_valid"][0,0].numpy(),rgb,
        prepared=(batch["rgb"],batch["initial"],batch["observed_support"]))
    np.testing.assert_array_equal(result["outputs"]["b0"],expected)


def test_evaluation_requires_opt_in_and_has_no_test_split_option(tmp_path,monkeypatch):
    import trident_depth.evaluation.hybrid_v1_dense4 as module
    monkeypatch.setattr(module,"HybridV1Dense4",lambda *a,**kw:pytest.fail("benchmark must not start"))
    args=["--manifest","missing.json","--checkpoint","missing.pt","--output",str(tmp_path/"hybrid_v1_dense4/unit")]
    with pytest.raises(SystemExit):main(args)
    with pytest.raises(SystemExit):main(args+["--execute-validation","--split","test"])
    assert not (tmp_path/"hybrid_v1_dense4").exists()


def test_metric_aggregation_uses_pixels_and_macro_boundaries():
    from trident_depth.hybrid_v1_dense4 import METHODS
    mask=np.ones((8,8),bool);gt=np.ones((8,8),np.float32)
    a=frame_metrics(gt+1,gt,mask)
    b=frame_metrics(np.ones((16,16),np.float32)*3,np.ones((16,16),np.float32),np.ones((16,16),bool))
    frames=[dict(methods={name:a for name in METHODS}),dict(methods={name:b for name in METHODS})]
    result=aggregate_frames(frames)["b0"]
    assert result["global_rmse_m"]==pytest.approx(np.sqrt((64+256*4)/320))
    assert result["global_mae_m"]==pytest.approx((64+256*2)/320)
    assert result["mean_per_frame"]["boundary_f1"]==1


def test_validation_guard_blocks_nonvalidation_payloads_and_writes(tmp_path):
    # Pure synthetic guard fixture: no real dataset is inspected.
    guard=object.__new__(data_module.ValidationOnly)
    guard.root=tmp_path;guard.manifest=tmp_path/"manifest.json"
    guard.paths={tmp_path/"validation.npy"};guard.opened=[];guard.denied=[];guard.loaded=[];guard.enabled=False
    allowed=tmp_path/"validation.npy";allowed.write_bytes(b"synthetic fixture")
    with guard.isolated():
        assert allowed.read_bytes()==b"synthetic fixture"
        with pytest.raises(PermissionError):(tmp_path/"locked.npy").read_bytes()
        with pytest.raises(PermissionError):allowed.write_bytes(b"overwrite")
        with pytest.raises(PermissionError):list(tmp_path.iterdir())
    assert not guard.enabled and allowed.read_bytes()==b"synthetic fixture"
    assert len(guard.denied)==3


def test_wrong_manifest_identity_stops_before_payload_access(tmp_path):
    manifest=tmp_path/"manifest.json";manifest.write_text("[]")
    reference=tmp_path/"reference.json";reference.write_text("[]")
    with pytest.raises(ValueError,match="immutable"):
        data_module.ValidationOnly(manifest,reference)


def test_validation_projection_preserves_original_preprocessing_and_never_selects_locked_rows(tmp_path,monkeypatch):
    from PIL import Image
    from torch.utils.data import default_collate
    from trident_depth.config import PipelineConfig
    from hashlib import sha256
    root=tmp_path/"data";root.mkdir();output=tmp_path/"hybrid_v1_dense4";output.mkdir()
    gt=np.full((480,640),2,np.float32);mask=np.ones_like(gt,dtype=bool)
    np.save(root/"d0.npy",gt);np.save(root/"m0.npy",mask)
    Image.fromarray(np.zeros((480,640,3),np.uint8)).save(root/"r0.png")
    rows=[dict(id=f"unit_{i}",scene_id=f"room_{i%30}",split="validation",official_split="train",
        depth=f"d{i}.npy",mask=f"m{i}.npy",rgb=f"r{i}.png",depth_scale_to_m=1) for i in range(123)]
    # No files are created for these synthetic excluded split records.
    rows += [dict(id="excluded",scene_id="excluded_room",split="test",official_split="test",
        depth="NEVER_OPEN.npy",depth_scale_to_m=1)]
    ref=[dict(id=r["id"],scene_id=r["scene_id"],mask_sha256=sha256(mask.tobytes()).hexdigest(),
              valid_pixels=gt.size) for r in rows[:123]]
    manifest=root/"manifest_development.json";manifest.write_text(json.dumps(rows))
    reference=tmp_path/"reference.json";reference.write_text(json.dumps(ref))
    before=manifest.read_bytes()
    monkeypatch.setattr(data_module,"NYU_MANIFEST_SHA256",sha256(before).hexdigest())
    monkeypatch.setattr(data_module,"REFERENCE_SHA256",sha256(reference.read_bytes()).hexdigest())
    # Test the new projection/loader bridge; existing real-file verifier is tested separately.
    monkeypatch.setattr(data_module,"audit_validation_reference",lambda _:ref)
    guarded=data_module.ValidationOnly(manifest,reference)
    with guarded.isolated():
        ds,path=guarded.prepare(output,PipelineConfig(max_depth=10))
        assert len(ds)==123 and all(r["split"]=="validation" for r in ds.source.records)
        assert "NEVER_OPEN" not in path.read_text()
        batch=default_collate([ds[0]])
        guarded.verify_batch(batch,0)
        np.testing.assert_array_equal(batch["initial"][0,0].numpy(),gt)
        with pytest.raises(ValueError,match="incomplete"):guarded.audit()
    assert manifest.read_bytes()==before and guarded.loaded==["unit_0"]


def test_reference_or_manifest_sample_paths_rejected_before_hashing(tmp_path,monkeypatch):
    monkeypatch.setattr(data_module,"sha256",lambda _:pytest.fail("sample must not be opened"))
    with pytest.raises(ValueError,match="outside"):
        data_module.ValidationOnly(tmp_path/"manifest_development.json",tmp_path/"locked.npy")
    with pytest.raises(ValueError,match="not a sample"):
        data_module.ValidationOnly(tmp_path/"locked.npy",tmp_path.parent/"reference.json")


def test_validation_symlink_rejected_before_any_payload_hash(tmp_path,monkeypatch):
    guarded=object.__new__(data_module.ValidationOnly)
    guarded.root=tmp_path/"data";payload=guarded.root/"validation.npy"
    guarded.paths={payload};guarded.enabled=True
    original=Path.is_symlink
    monkeypatch.setattr(Path,"is_symlink",lambda p:True if p==payload else original(p))
    monkeypatch.setattr(data_module,"audit_validation_reference",lambda _:pytest.fail("symlink target must not be read"))
    with pytest.raises(ValueError,match="symlink"):
        guarded.prepare(tmp_path/"output",None)
