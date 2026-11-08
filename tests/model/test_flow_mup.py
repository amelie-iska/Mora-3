"""Native flow readout/width/recovery checks; synthetic fixtures, not training."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mora3.model.response_flow import ResponseFlowHead,ResponseFlowDenoiserV2
from mora3.model.response_flow_trainer import ResponseFlowTrainer,ResponseFlowBatch
from mora3.model.flow_mup import flow_mup_parameter_roles,flow_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from test_sciplex_mup import make
from test_sciplex_shared import task


def flow(cls=ResponseFlowHead,width=16):
    kwargs=dict(parameterization_policy='mup-response-flow-v1')
    if cls is ResponseFlowDenoiserV2:kwargs['sigma_min']=.1
    return cls(make(width),**kwargs).double()


def trees_equal(left,right):
    if type(left) is not type(right):return False
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(trees_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(trees_equal(a,b) for a,b in zip(left,right))
    return left==right


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
@pytest.mark.parametrize('width',[8,12,16])
def test_flow_readout_init_independent_source_recipe_rng(cls,width):
    torch.manual_seed(2026110631);native=make(width)
    state,time,out=nn.Linear(2,width),nn.Linear(1,width),nn.Linear(width,2)
    with torch.no_grad():out.weight.mul_(math.sqrt(width/8));out.bias.mul_(math.sqrt(width/8))
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110631);m=flow(cls,width)
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in [(m.state,state),(m.time,time),(m.output,out)]:
        assert torch.equal(a.weight,b.weight.double()) and torch.equal(a.bias,b.bias.double())
    assert m.config['version']==(3 if cls is ResponseFlowHead else 4)


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_flow_complete_output_state_time_and_parameter_gradients_independent(cls):
    m=flow(cls,16);ts=[task()]
    state=torch.tensor([[.2,-.3]],dtype=torch.float64,requires_grad=True)
    time=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    context=m.native.backbone([m.native.graph(ts[0])])[0][:,0]
    hidden=F.silu(context+F.linear(state,m.state.weight,m.state.bias)
                  +F.linear(time[:,None],m.time.weight,m.time.bias))
    expected=F.linear(hidden*(1.5/(16/8)),m.output.weight,m.output.bias)
    if cls is ResponseFlowDenoiserV2:expected=(expected-.9*state)/(1-.9*time[:,None])
    actual=m(ts,state,time);torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    owners=(state,time,*m.parameters())
    a=torch.autograd.grad(actual.sum(),owners,allow_unused=True)
    b=torch.autograd.grad(expected.sum(),owners,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_complete_native_flow_family_counts_fixed_axes_groups_and_no_updates(cls):
    ms=[flow(cls,w) for w in (8,12,16)];family=flow_mup_width_family(*ms)
    assert family.total_numel==(1314,2590,4282)
    for member,m in zip(('base','delta','target'),ms):
        roles=flow_mup_parameter_roles(m)
        opt=MuAdamW(m,family,roles,base_lr=.002,weight_decay=.01,
            decay_aliases=tuple(n for n,r in roles.items() if len(r.axes)==2),member=member)
        assert not opt.state
        group=next(g for g in opt.param_groups if g['mup_aliases']==('output.weight',))
        assert group['lr']==.002


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_exact_staged_model_muadamw_rng_and_cursor_resume_with_common_lr_schedule(cls):
    torch.manual_seed(2026110632)
    ms=[flow(cls,w) for w in (8,12,16)];family=flow_mup_width_family(*ms)
    batch=ResponseFlowBatch((task(),task(counts=(3,4),cell='b')))
    sigma=.1 if cls is ResponseFlowDenoiserV2 else .01
    roles=flow_mup_parameter_roles(ms[-1]);decay=tuple(n for n,r in roles.items() if len(r.axes)==2)
    trainer=ResponseFlowTrainer(ms[-1],batch,batch_size=1,sigma_min=sigma,lr=.002,
        mu_family=family,weight_decay=.01,decay_aliases=decay)
    assert isinstance(trainer.optimizer,MuAdamW)
    trainer.step()
    # Common-factor schedule preserves group ratios. Campaign owns any scheduler
    # object state separately; this check covers the scheduled optimizer LR.
    for group in trainer.optimizer.param_groups:group['lr']*=.5
    saved=copy.deepcopy(trainer.state_dict());trainer.step();expected=trainer.state_dict()
    fresh=ResponseFlowTrainer(flow(cls),batch,batch_size=1,sigma_min=sigma,lr=.002,
        mu_family=family,weight_decay=.01,decay_aliases=decay)
    fresh.load_state_dict(saved);fresh.step()
    assert trees_equal(expected,fresh.state_dict())


def test_opt_in_trainer_requires_qualified_family_and_refuses_corrupt_recovery_before_live_mutation():
    ms=[flow(width=w) for w in (8,12,16)];family=flow_mup_width_family(*ms)
    batch=ResponseFlowBatch((task(),))
    with pytest.raises(ValueError,match='family'):ResponseFlowTrainer(ms[-1],batch,batch_size=1)
    trainer=ResponseFlowTrainer(ms[-1],batch,batch_size=1,mu_family=family)
    bad=trainer.state_dict();first=next(k for k,v in bad['model'].items() if isinstance(v,torch.Tensor))
    bad['model'][first]=torch.full_like(bad['model'][first],19.)
    bad['optimizer']['mora3_mu_adamw']['family_sha256']='f'*64
    before=trainer.state_dict()
    with pytest.raises(ValueError):trainer.load_state_dict(bad)
    assert trees_equal(before,trainer.state_dict())


@pytest.mark.parametrize('failure',['native','flow-policy','mult','sigma','tie'])
def test_flow_family_refuses_owner_source_and_alias_drift(failure):
    ms=[flow(width=w) for w in (8,12,16)]
    if failure=='native':ms[1].native.config['mup']['base_width']=4
    if failure=='flow-policy':ms[1].config['mup']['policy']='other'
    if failure=='mult':ms[1].config['mup']['output_mult']=2.
    if failure=='sigma':ms[1]=flow(ResponseFlowDenoiserV2,12)
    if failure=='tie':ms[1].state.weight=ms[1].native.expression.weight
    with pytest.raises(ValueError):flow_mup_width_family(*ms)


def test_opt_in_denoiser_sampling_refuses_wrong_sigma_before_solver(monkeypatch):
    from mora3.model.response_sampling import ResponseContext,sample_response
    from mora3.model import flow_matching
    m=flow(ResponseFlowDenoiserV2)
    context=ResponseContext.from_task(task())
    def forbidden(*args,**kwargs):raise AssertionError('solver entered before sigma refusal')
    monkeypatch.setattr(flow_matching,'euler_flow',forbidden)
    rng=torch.get_rng_state().clone();modes=[module.training for module in m.modules()]
    with pytest.raises(ValueError,match='sigma'):
        sample_response(m,[context],steps=1,seed=2026110633,sigma_min=.01,negative_policy='preserve')
    assert torch.equal(rng,torch.get_rng_state())
    assert modes==[module.training for module in m.modules()]
