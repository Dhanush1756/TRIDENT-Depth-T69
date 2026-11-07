"""Synthetic app integration tests; no dataset/checkpoint artifacts are touched."""
import io
import base64
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.engine import ApplicationEngine


def png(a):
    s=io.BytesIO();Image.fromarray(a).save(s,format="PNG");return s.getvalue()


@pytest.fixture
def extended(monkeypatch):
    import trident_depth.extended_v1.checkpoint as checkpoint
    import trident_depth.training.colab as colab
    from trident_depth.extended_v1.model import ExtendedTRIDENT
    from backend.engine import EPOCH12_SHA256
    model=ExtendedTRIDENT()
    # Synthetic model only: actual checkpoint remains exclusively on Drive.
    monkeypatch.setattr(colab,"sha256",lambda _:EPOCH12_SHA256)
    monkeypatch.setattr(checkpoint,"load",lambda *a:(model,dict(training_state=dict(epoch=12))))
    monkeypatch.setenv("TRIDENT_CHECKPOINT","synthetic_test_fixture.pt")
    monkeypatch.setenv("TRIDENT_MODEL_FORMAT","extended_v1")
    with TestClient(app) as c:yield c,model


@pytest.mark.parametrize('scale,protocol,noisy',[(2,'dense_sr',False),(4,'dense_sr',False),(8,'dense_sr',False),
    (4,'sparse_completion',False),(4,'sparse_completion',True)])
@pytest.mark.parametrize('method',['deep','hybrid'])
def test_extended_endpoint_protocols_and_unchanged_weights(extended,scale,protocol,noisy,method):
    c,model=extended
    import torch
    before={k:v.clone() for k,v in model.state_dict().items()}
    n=32//scale if protocol=='dense_sr' else 32
    depth=np.full((n,n),2000,np.uint16)
    if protocol=='sparse_completion':depth[8:24,8:24]=0
    r=c.post('/api/reconstruct',files={'depth':('d.png',png(depth)),
        'rgb':('rgb.png',png(np.full((32,32,3),90,np.uint8)))},
        data=dict(method=method,scale=scale,protocol=protocol,noisy=str(noisy).lower()))
    assert r.status_code==200,r.text
    data=r.json();assert data['metrics'] is None and data['output_shape']==[32,32]
    with np.load(io.BytesIO(base64.b64decode(data['output_npz']))) as z:
        np.testing.assert_allclose(z['depth_m'],2,atol=1e-5)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert data['checkpoint']['model']=='trident_extended_v1'


@pytest.mark.parametrize('fields',[
    dict(scale=2,protocol='sparse_completion'),dict(noisy='true'),dict(max_depth_m=20)])
def test_extended_rejects_unqualified_input_modes(extended,fields):
    c,_=extended
    r=c.post('/api/reconstruct',files={'depth':('d.png',png(np.full((8,8),2000,np.uint16))),
        'rgb':('rgb.png',png(np.zeros((32,32,3),np.uint8)))},data=dict(method='deep',**fields))
    assert r.status_code==400


def test_model_requires_rgb_and_health_does_not_leak_paths(extended):
    c,_=extended
    h=c.get('/api/health').json()
    assert h['capabilities']['hybrid']['rgb_required']
    assert 'synthetic_test_fixture' not in str(h)
    r=c.post('/api/reconstruct',files={'depth':('d.png',png(np.full((8,8),2000,np.uint16)))},data={'method':'deep'})
    assert r.status_code==400 and 'RGB' in r.json()['detail']


def test_wrong_real_checkpoint_hash_fails_closed(tmp_path):
    p=tmp_path/'wrong.pt';p.write_bytes(b'not a real checkpoint')
    with pytest.raises(ValueError,match='SHA256'):ApplicationEngine(p,model_format='extended_v1')


def test_b0_advertises_only_dense4_deep(monkeypatch):
    from trident_depth.models.residual_v2 import ResidualCNNV2
    e=ApplicationEngine()
    e.model=ResidualCNNV2(uncertainty_enabled=False)
    caps=e.capabilities()
    assert 'hybrid' not in caps and caps['deep']['scales']==[4]
    assert caps['deep']['protocols']==['dense_sr']


def test_raw_deep_is_not_silently_replaced_by_observed_point_copy(extended):
    c,model=extended
    import torch
    # Only the temporary synthetic fixture is changed, never trained evidence.
    with torch.no_grad():model.residual_head.bias.fill_(.07)
    files={'depth':('d.png',png(np.full((16,16),2000,np.uint16))),
           'rgb':('r.png',png(np.zeros((16,16,3),np.uint8)))}
    for method,value in [('deep',2.07),('hybrid',2.0)]:
        r=c.post('/api/reconstruct',files=files,data=dict(method=method,protocol='sparse_completion'))
        assert r.status_code==200
        with np.load(io.BytesIO(base64.b64decode(r.json()['output_npz']))) as z:
            np.testing.assert_allclose(z['depth_m'],value,atol=1e-6)


def test_bad_mask_and_object_array_rejected_then_server_recovers(monkeypatch):
    monkeypatch.delenv('TRIDENT_CHECKPOINT',raising=False)
    with TestClient(app) as c:
        files={'depth':('d.png',png(np.full((8,8),2000,np.uint16))),
               'mask':('m.png',png(np.full((8,8),2,np.uint8)))}
        assert c.post('/api/reconstruct',files=files).status_code==400
        s=io.BytesIO();np.save(s,np.array([[object()]],dtype=object))
        assert c.post('/api/reconstruct',files={'depth':('bad.npy',s.getvalue())}).status_code==400
        assert c.post('/api/reconstruct',files={'depth':files['depth']}).status_code==200
