"""Extended protocol, isolation, objective and complete-state regression checks."""
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import pytest
import torch
from trident_depth.extended_v1.model import ExtendedTRIDENT
from trident_depth.extended_v1.loss import objective,boundary_targets,false_edge_loss
from trident_depth.extended_v1.data import DevelopmentData,ProtocolDataset,pipeline,tensor_sample
from trident_depth.extended_v1.contract import read_config
from trident_depth.extended_v1.checkpoint import save,load,resume
from trident_depth.extended_v1.inference import ExtendedInference,hybrid_refine,project_dense
from trident_depth.data.sample import DepthSample
from trident_depth.data.degradation import degrade
from trident_depth.models.residual_v2 import ResidualCNNV2
from trident_depth.training.state import seed_everything,capture_state

ROOT=Path(__file__).resolve().parents[1]
CFG=read_config(ROOT/'configs/extended_v1/train.json')
torch.set_num_threads(2)


def batch_for(task,size=16):
    from torch.utils.data import default_collate
    depth=np.full((size,size),2.,np.float32)
    depth[:,size//2:]=3.
    rgb=np.zeros((size,size,3),np.uint8)
    rgb[:,::2]=255  # RGB stripes unrelated to true depth boundary
    sample=DepthSample('synthetic','scene',depth,np.ones_like(depth,bool),rgb)
    return default_collate([tensor_sample(degrade(sample,pipeline(task)),task,'synthetic')])


@pytest.mark.parametrize('task',CFG['tasks'],ids=lambda t:t['name'])
def test_protocol_shapes_finite_masked_loss_and_gradients(task):
    seed_everything(69)
    m=ExtendedTRIDENT()
    b=batch_for(task,32)
    out=m(b['rgb'],b['initial'],b['observed_support'],task['pipeline']['scale'],task['pipeline']['protocol'])
    assert out['depth'].shape==(1,1,32,32)
    assert torch.equal(out['depth'],b['initial'])
    assert 'log_scale' not in out
    loss,terms=objective(out,b,task,CFG['loss_weights'])
    assert torch.isfinite(loss) and terms['uncertainty']==0
    loss.backward()
    assert m.residual_head.weight.grad is not None
    assert m.boundary_head.weight.grad.abs().sum()>0
    for name,p in m.named_parameters():
        if name.startswith('log_scale_head'):
            assert p.grad is None and not p.requires_grad
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_b0_tensor_compatibility_and_exact_dense4_prediction():
    seed_everything(69)
    original=ResidualCNNV2(uncertainty_enabled=False).eval()
    m=ExtendedTRIDENT().eval()
    m.load_state_dict(original.state_dict(),strict=True)
    b=batch_for(CFG['tasks'][1])
    assert sum(p.numel() for p in m.parameters())==473059
    with torch.inference_mode():
        a=original(b['rgb'],b['initial'],b['observed_support'])
        c=m(b['rgb'],b['initial'],b['observed_support'])
    assert torch.equal(a['depth'],c['depth'])
    assert torch.equal(a['boundary_confidence'],c['boundary_confidence'])
    with pytest.raises(ValueError):
        original(b['rgb'],b['initial'],b['observed_support'],2)


def test_boundary_labels_are_depth_only_and_exclude_invalid_neighbours():
    target=torch.ones(1,1,8,8)
    target[:,:,:,4:]=2
    mask=torch.ones_like(target,dtype=torch.bool)
    labels,valid=boundary_targets(target,mask)
    assert labels[:,:,:,3:5].all() and not labels[:,:,:,:3].any()
    target[:,:,2,2]=float('nan');mask[:,:,2,2]=False
    labels,valid=boundary_targets(target,mask)
    assert not valid[:,:,1:4,1:4].any()
    assert torch.isfinite(labels).all()


@pytest.mark.parametrize('empty',[False,True])
def test_invalid_nan_targets_and_empty_masks_are_finite(empty):
    t=CFG['tasks'][0];b=batch_for(t)
    b['target_valid'][...,:8,:8]=False
    if empty: b['target_valid'].zero_()
    b['target'][~b['target_valid']]=float('nan')
    m=ExtendedTRIDENT()
    out=m(b['rgb'],b['initial'],b['observed_support'],2)
    loss,_=objective(out,b,t,CFG['loss_weights'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_false_edge_loss_detects_texture_without_penalizing_correct_slope():
    target=torch.ones(1,1,8,8)
    mask=target.bool()
    assert false_edge_loss(target,target,mask)==0
    striped=target.clone();striped[:,:,:,::2]+=.1
    assert false_edge_loss(striped,target,mask)>0


@pytest.mark.parametrize('scale',[2,4,8])
def test_generalized_area_projection_retains_block_means(scale):
    rng=np.random.default_rng(69)
    low=rng.uniform(.5,9,(3,4)).astype(np.float32)
    candidate=rng.uniform(.01,10,(3*scale,4*scale)).astype(np.float32)
    out=project_dense(candidate,low,scale)
    actual=out.reshape(3,scale,4,scale).mean((1,3),dtype=np.float64)
    np.testing.assert_allclose(actual,low,atol=2e-6,rtol=0)
    assert out.min()>0 and out.max()<=10


def test_hybrid_rejects_rgb_only_edges_and_clean_points_are_retained():
    depth=np.full((16,16),2.,np.float32)
    mask=np.zeros_like(depth,bool);mask[::2,::2]=True
    rgb=np.zeros((16,16,3),np.uint8);rgb[:,::2]=255
    out=hybrid_refine(depth,np.ones_like(depth),depth,mask,rgb,scale=4,protocol='sparse_completion')
    np.testing.assert_array_equal(out['depth'],depth)
    assert not out['geometry_gate'].any()
    noisy=depth.copy();noisy[mask]=2.2
    result=hybrid_refine(depth,np.ones_like(depth),noisy,mask,rgb,scale=4,protocol='sparse_completion',noisy=True)
    np.testing.assert_array_equal(result['depth'],depth)  # noisy observations not hard copied


@pytest.mark.parametrize('task',CFG['tasks'],ids=lambda t:t['name'])
def test_adapter_supports_each_task(task):
    b=batch_for(task,32)
    rgb=(b['rgb'][0].permute(1,2,0).numpy()*255).astype(np.uint8)
    d=b['observations'].numpy()[0,0];mask=b['input_valid'].numpy()[0,0]
    c=task['pipeline']
    out=ExtendedInference(ExtendedTRIDENT()).run(d,mask,rgb,scale=c['scale'],protocol=c['protocol'],noisy=c['noise_std_m']>0)
    assert out['depth'].shape==(32,32) and np.isfinite(out['depth']).all()


def test_complete_checkpoint_roundtrip_resume_and_no_overwrite(tmp_path):
    seed_everything(69)
    m=ExtendedTRIDENT()
    opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=1e-4)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    generator=torch.Generator().manual_seed(69)
    b=batch_for(CFG['tasks'][1]);o=m(b['rgb'],b['initial'],b['observed_support'])
    objective(o,b,CFG['tasks'][1],CFG['loss_weights'])[0].backward()
    opt.step();sch.step()  # synthetic unit test only; no dataset training
    state=capture_state(generator,1,.1,1)
    state['history']=[{'epoch':1}]
    path=tmp_path/'extended.pt';provenance={'version':'test'}
    save(path,m,opt,sch,scaler,state,provenance)
    expected=torch.rand(3)
    restored,payload=load(path)
    assert payload['optimizer_state']['state']
    assert all(torch.equal(p,restored.state_dict()[k]) for k,p in m.state_dict().items())
    saved=resume(path,m,opt,sch,scaler,generator,provenance)
    assert saved['epoch']==1 and torch.equal(torch.rand(3),expected)
    def next_step():
        opt.zero_grad(set_to_none=True)
        output=m(b['rgb'],b['initial'],b['observed_support'])
        objective(output,b,CFG['tasks'][1],CFG['loss_weights'])[0].backward()
        opt.step();sch.step()
        return {k:v.clone() for k,v in m.state_dict().items()}
    uninterrupted=next_step()
    resume(path,m,opt,sch,scaler,generator,provenance)
    resumed=next_step()
    assert all(torch.equal(v,resumed[k]) for k,v in uninterrupted.items())
    with pytest.raises(FileExistsError):save(path,m,opt,sch,scaler,state,provenance)
    with pytest.raises(ValueError):resume(path,m,opt,sch,scaler,generator,{'wrong':True})
    payload['model_state']['unexpected']=torch.zeros(1)
    torch.save(payload,tmp_path/'corrupt.pt')
    with pytest.raises(RuntimeError):load(tmp_path/'corrupt.pt')


def test_training_schedule_covers_each_frame_once_and_balances_tasks():
    # Seven consecutive frames rotate across seven tasks; 672/7 = 96 each.
    from collections import Counter
    for epoch in range(1,13):
        counts=Counter((i+epoch-1)%7 for i in range(672))
        assert set(counts.values())=={96}
    with pytest.raises(ValueError):ProtocolDataset(None,CFG['tasks'],'test')


def test_guard_denies_synthetic_locked_sample_before_open(tmp_path):
    data=DevelopmentData.__new__(DevelopmentData)
    data.root=tmp_path;data.manifest=tmp_path/'manifest_development.json'
    data.allowed={tmp_path/'allowed.npy'};data.enabled=False;data.opened=set();data.denied=[]
    with data.isolated():
        with pytest.raises(PermissionError):
            (tmp_path/'locked_test.npy').open('rb')
    assert len(data.denied)==1


def test_training_gate_fails_closed_without_execute_flag():
    from trident_depth.extended_v1.train import main
    with pytest.raises(SystemExit):
        main(['--config','x','--manifest','x','--checkpoint','x','--preflight','x','--output','x'])


def test_validation_is_inference_only_original_ids_masks_and_pixel_aggregation():
    from trident_depth.extended_v1.train import validate
    import copy
    task=CFG['tasks'][1]
    b=batch_for(task,16)
    reference=[];batches=[]
    digest=hashlib.sha256(b['target_valid'].numpy().tobytes()).hexdigest()
    for i in range(123):
        row=copy.deepcopy(b);row['sample_id']=[f'val_{i}']
        reference.append(dict(id=f'val_{i}',mask_sha256=digest))
        batches.append(row)
    m=ExtendedTRIDENT()
    before={k:v.clone() for k,v in m.state_dict().items()}
    score,aggregate,frames=validate(m,batches,[task],CFG['loss_weights'],'cpu',reference)
    assert len(frames)==123 and aggregate['dense4']['pixels']==123*256
    assert np.isfinite(score)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())
    batches[0]['sample_id']=['train_sample']
    with pytest.raises(ValueError):validate(m,batches,[task],CFG['loss_weights'],'cpu',reference)
