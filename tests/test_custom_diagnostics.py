"""Analytic diagnostics on synthetic tensors; no training/checkpoint writes."""
import math
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from torch.nn import functional as F
from trident_depth.training.losses import combined_loss,masked_l1
from trident_depth.evaluation.custom_diagnostics import independent_losses,calibration_summary


def test_laplace_nll_negative_value_and_analytic_gradients():
    prediction=torch.full((1,1,2,2),1.02,dtype=torch.float64,requires_grad=True)
    target=torch.ones_like(prediction)
    s=torch.full_like(prediction,math.log(.01),requires_grad=True)
    mask=torch.ones_like(prediction,dtype=torch.bool)
    weights=dict(depth=0.,gradient=0.,edge=0.,uncertainty=1.,measurement=0.)
    loss,_=combined_loss(dict(depth=prediction,log_scale=s),target,mask,weights=weights)
    expected=-torch.distributions.Laplace(prediction,s.exp()).log_prob(target).mean()-math.log(2)
    torch.testing.assert_close(loss,expected)
    assert float(loss.detach())<0  # Valid continuous-density NLL; not a sign bug.
    dp,ds=torch.autograd.grad(loss,(prediction,s))
    torch.testing.assert_close(dp,torch.full_like(prediction,25.))
    torch.testing.assert_close(ds,torch.full_like(s,-.25))


def test_log_scale_clamp_has_zero_gradient_when_saturated():
    prediction=torch.ones((1,1,2,2))
    s=torch.full_like(prediction,-7.,requires_grad=True)
    _,terms=combined_loss(dict(depth=prediction,log_scale=s),prediction*1.01,prediction.bool())
    gradient=torch.autograd.grad(terms["uncertainty"],s)[0]
    assert torch.equal(gradient,torch.zeros_like(s))
    assert torch.isfinite(terms["uncertainty"])


def test_independent_normalization_and_masks_match_existing_losses():
    gt=np.ones((8,8),np.float64)
    gt[4:]=3
    mask=np.ones_like(gt,bool)
    mask[0,0]=False
    gt[0,0]=np.nan
    prediction=np.nan_to_num(gt,nan=1)+np.arange(64).reshape(8,8)*.001
    s=np.full_like(gt,-4.)
    p,t,m,log_s=(torch.from_numpy(v)[None,None] for v in (prediction,gt,mask,s))
    observations=F.avg_pool2d(torch.nan_to_num(t,nan=1),4,4)
    input_mask=torch.ones_like(observations,dtype=torch.bool)
    input_mask[0,0,0,0]=False
    _,terms=combined_loss(dict(depth=p,log_scale=log_s),t,m)
    terms["measurement"]=masked_l1(F.avg_pool2d(p,4,4),observations,input_mask)
    reference=independent_losses(prediction,gt,mask,s,observations[0,0].numpy(),input_mask[0,0].numpy())
    for key,value in terms.items():
        assert reference[key]==pytest.approx(float(value.detach()),abs=1e-12)


def test_calibration_handles_collapsed_scale_and_preserves_bin_means():
    b=np.linspace(.001,.03,100)
    sample=np.stack([b,b,np.arange(100)*.001,-np.arange(100)*.001],axis=1)
    result=calibration_summary(sample)
    assert result["spearman_b_vs_abs_error"]==pytest.approx(1)
    assert all(row["predicted_b_m"]==pytest.approx(row["observed_mae_m"]) for row in result["decile_bins"])
    sample[:,1]=.01
    assert calibration_summary(sample)["spearman_b_vs_abs_error"] is None
