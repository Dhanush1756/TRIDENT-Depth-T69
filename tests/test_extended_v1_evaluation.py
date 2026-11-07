"""Independent verification of frozen evaluation, masks, units and read-only paths."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate
from trident_depth.extended_v1 import evaluation as ev
from trident_depth.extended_v1 import evaluation_audit as audit
from trident_depth.extended_v1.evaluation_diagnostics import error_sums,summary,regions,correction_diagnostics,texture_diagnostic,panel
from trident_depth.extended_v1.data import pipeline,tensor_sample
from trident_depth.extended_v1.model import ExtendedTRIDENT
from trident_depth.extended_v1.contract import read_config
from trident_depth.data.degradation import degrade
from trident_depth.data.sample import DepthSample
from trident_depth.evaluation.metrics import evaluate

ROOT=Path(__file__).resolve().parents[1]
CFG=read_config(ROOT/'configs/extended_v1/train.json')
PROTO=json.loads((ROOT/'configs/extended_v1/validation_epoch012.json').read_text())


def sample(task):
    depth=np.full((32,32),2,np.float32);depth[:,16:]=3
    rgb=np.zeros((32,32,3),np.uint8);rgb[:,::2]=255
    return degrade(DepthSample('synthetic','scene',depth,np.ones((32,32),bool),rgb),pipeline(task))


def test_protocol_is_pinned_and_b0_is_only_dense4():
    assert hashlib.sha256((ROOT/'configs/extended_v1/validation_epoch012.json').read_bytes()).hexdigest()==ev.EVALUATION_PROTOCOL_SHA256
    for task in CFG['tasks']:
        assert ('b0' in ev.methods_for(task))==(task['name']=='dense4')
    assert PROTO['tasks']==[t['name'] for t in CFG['tasks']]
    assert PROTO['frames_per_task']==123 and PROTO['warmup']==1 and PROTO['repeats']==3


@pytest.mark.parametrize('task',CFG['tasks'],ids=lambda t:t['name'])
def test_evaluation_uses_original_preprocessing_exactly_and_never_updates(task):
    prepared=sample(task)
    b=default_collate([tensor_sample(prepared,task,'synthetic')])
    m=ExtendedTRIDENT().eval()
    before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.inference_mode():
        expected=m(b['rgb'],b['initial'],b['observed_support'],task['pipeline']['scale'],task['pipeline']['protocol'])['depth'].numpy()[0,0]
    actual=ev.neural_prediction(m,prepared,task,'cpu')
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(ev.baseline('bicubic_fill',prepared),b['initial'].numpy()[0,0])
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())


def test_checkpoint_sha_is_checked_before_torch_deserialization(tmp_path,monkeypatch):
    p=tmp_path/'epoch_012.pt';p.write_bytes(b'not-the-checkpoint')
    monkeypatch.setattr(audit,'load',lambda *a,**k:pytest.fail('must not deserialize'))
    with pytest.raises(ValueError,match='SHA256'):
        audit.inspect(p,'unused','unused','unused')


def test_log_audit_requires_twelve_epochs_and_exact_hash(tmp_path,monkeypatch):
    p=tmp_path/'training.log';p.write_text('\n'.join(json.dumps({'epoch':i}) for i in range(1,13)))
    monkeypatch.setattr(audit,'TRAINING_LOG_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
    assert len(audit.log_rows(p))==12
    p.write_text('changed')
    with pytest.raises(ValueError,match='SHA256'):audit.log_rows(p)


def test_full_metadata_audit_accepts_consistent_fixture_and_rejects_changed_metrics(tmp_path,monkeypatch):
    # Synthetic metadata fixture, explicitly NOT the real epoch-12 checkpoint.
    from trident_depth.extended_v1.contract import source_hashes
    model=ExtendedTRIDENT()
    checkpoint=tmp_path/'epoch_012.pt';checkpoint.write_bytes(b'synthetic-test-fixture')
    config=ROOT/'configs/extended_v1/train.json'
    reference=ROOT/'docs/pilot_dense4/per_frame.json'
    ids=[r['id'] for r in json.loads(reference.read_text())]
    history=[]
    for epoch in range(1,13):
        history.append(dict(epoch=epoch,optimizer_steps=672,selection_score_m=.25+(12-epoch)*.01,
            training_task_counts={t['name']:96 for t in CFG['tasks']},
            split_audit=dict(official_test_sample_files_opened=0,forbidden_access_attempts=[]),
            validation={t['name']:dict(rmse_m=.25,mae_m=.0625) for t in CFG['tasks']}))
    log=tmp_path/'console.log';log.write_text('\n'.join(json.dumps(r) for r in history))
    monkeypatch.setattr(audit,'TRAINING_LOG_SHA256',hashlib.sha256(log.read_bytes()).hexdigest())
    real_hash=audit.sha256
    monkeypatch.setattr(audit,'sha256',lambda p:audit.EPOCH12_SHA256 if Path(p)==checkpoint else real_hash(p))
    provenance=dict(config=CFG,config_sha256=real_hash(config),manifest_sha256=audit.NYU_MANIFEST_SHA256,
        initial_checkpoint_sha256=audit.B0_SHA256,model_identity='trident_extended_v1',precision='fp32',amp_enabled=False,
        source_hashes=source_hashes())
    payload=dict(model_state=model.state_dict(),provenance=provenance,
        training_state=dict(epoch=12,best_epoch=12,best_rmse=.25,history=history,
                            **{k:[] for k in ('python_rng','numpy_rng','torch_rng','loader_rng','cuda_rng')}),
        scheduler_state=dict(last_epoch=12),scaler_state={},
        optimizer_state=dict(state={i:dict(step=torch.tensor(8064.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                                   for i,p in enumerate(p for p in model.parameters() if p.requires_grad)}))
    monkeypatch.setattr(audit,'load',lambda *a,**k:(model,payload))
    (tmp_path/'run.json').write_text(json.dumps(provenance))
    metrics=dict(summary=history[-1],validation_frames=[dict(task=t['name'],id=i,valid_pixels=16,squared_error_m2=1,absolute_error_m=1)
                                                     for t in CFG['tasks'] for i in ids])
    path=tmp_path/'epoch_012_metrics.json';path.write_text(json.dumps(metrics))
    _,_,report=audit.inspect(checkpoint,log,config,reference)
    assert report['status']=='PASS' and report['parameters']==473059
    metrics['validation_frames'][0]['squared_error_m2']=100
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match='per-frame error sums'):audit.inspect(checkpoint,log,config,reference)


def test_region_masks_partition_missing_and_distance_without_hr_point_fiction():
    t=CFG['tasks'][5];s=sample(t)
    masks=regions(s.target.valid,s.input_valid,t,PROTO['distance_bins_px'])
    assert masks['central_hole'].sum()==32*32//4
    assert not (masks['observed']&masks['missing']).any()
    assert np.array_equal(masks['observed']|masks['missing'],s.target.valid)
    count=sum(v.astype(int) for k,v in masks.items() if k.startswith('distance_'))
    np.testing.assert_array_equal(count,np.ones((32,32),int))
    assert list(regions(s.target.valid,np.ones((16,16),bool),CFG['tasks'][0],PROTO['distance_bins_px']))==['all']


def test_weighted_aggregation_is_not_mean_frame_rmse_and_psnr_uses_10m():
    rows=[]
    for n,error in [(1,1),(3,3)]:
        target=np.ones((2,2),np.float32)
        pred=target+error
        mask=np.zeros((2,2),bool);mask.flat[:n]=True
        sums=error_sums(pred,target,mask)
        rows.append(dict(errors=sums,regions={'all':sums},metrics=evaluate(pred,target,mask),
            timing=dict(median_seconds=.1,memory=dict(peak_allocated_bytes=None,peak_reserved_bytes=None))))
    r=ev.aggregate(rows)
    assert r['pixels']==4 and r['rmse_m']==pytest.approx(np.sqrt(7))
    assert r['mae_m']==2.5 and r['global_psnr_db']==pytest.approx(10*np.log10(100/7))
    assert r['mean_frame_ssim'] is None and r['ssim_eligible_frames']==0


def test_region_errors_do_not_hide_missing_pixels_and_exclude_invalid_nan():
    target=np.full((4,4),2,np.float32);pred=target.copy();pred[1:3,1:3]+=1
    mask=np.ones((4,4),bool);mask[0,0]=False;target[0,0]=np.nan
    sums=summary(error_sums(pred,target,mask))
    assert sums['pixels']==15 and sums['squared_error_m2']==4
    assert sums['rmse_m']==pytest.approx(np.sqrt(4/15))
    c=correction_diagnostics(pred,np.full((4,4),2,np.float32),target,mask)
    assert c['worsened_pixels']==4 and c['unchanged_pixels']==11
    empty=summary(error_sums(pred,target,np.zeros_like(mask)))
    assert empty['rmse_m'] is None and empty['pixels']==0


def test_measure_uses_repeats_and_rejects_nonreproducible_or_nan_outputs():
    counter=[]
    def call():counter.append(1);return np.ones((8,8),np.float32)
    value,t=ev.measure(call,torch.device('cpu'),1,3)
    assert len(counter)==4 and len(t['seconds'])==3 and np.isfinite(value).all()
    state=[0]
    def changes():state[0]+=1;return np.full((4,4),state[0],np.float32)
    with pytest.raises(ValueError,match='reproducible'):ev.measure(changes,torch.device('cpu'),0,2)
    with pytest.raises(ValueError,match='nonfinite'):ev.measure(lambda:np.full((4,4),np.nan),torch.device('cpu'),0,1)


def test_texture_pair_diagnostic_and_panel_do_not_overwrite(tmp_path):
    target=np.ones((32,32),np.float32)
    rgb=np.zeros((32,32,3),np.uint8);rgb[:,::2]=255
    mask=np.ones_like(target,bool)
    result=texture_diagnostic(target,target,mask,rgb)
    assert result['rgb_edge_flat_gt_pairs']>0 and result['false_depth_edge_fraction']==0
    striped=target.copy();striped[:,::2]+=.1
    assert texture_diagnostic(striped,target,mask,rgb)['false_depth_edge_pairs']>0
    p=tmp_path/'panel.png'
    panel(p,rgb,target,mask,target,mask,{'extended':striped},'Synthetic diagnostic, not experiment evidence')
    assert p.read_bytes().startswith(b'\x89PNG')
    with pytest.raises(FileExistsError):panel(p,rgb,target,mask,target,mask,{},'no overwrite')


def test_evaluation_cli_requires_deliberate_execution_and_gpu_before_data_io():
    args=['--checkpoint','unused','--training-log','unused','--manifest','unused','--b0-checkpoint','unused','--output','extended_v1_epoch012/new']
    with pytest.raises(SystemExit):ev.main(args)
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):ev.main(args+['--execute-validation'])
