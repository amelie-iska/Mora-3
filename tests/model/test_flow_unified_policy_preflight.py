"""Cooperative flow refuses drift before touching live training state."""
import copy
import pytest
import torch
from test_flow_unified import build
from test_response_flow_trainer import equal
from mora3.model.flow_unified import FlowUnifiedTrainer


def corrupt(m, kind):
    if kind == 'declaration': m.configuration['flow']['native']['width'] = 99
    elif kind == 'wrapper': m.flow.config['native']['width'] = 99
    elif kind == 'native': m.native.response.config['width'] = 99
    else: m.flow.native = copy.deepcopy(m.native.response)


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('kind', ['declaration', 'wrapper', 'native', 'alias'])
@pytest.mark.parametrize('entry', ['export', 'extra-load', 'construct', 'step', 'save', 'load'])
def test_drift_refuses_before_live_mutation(version, kind, entry):
    tr = build(); m = tr.model
    if version == 2:
        from mora3.model.response_flow import ResponseFlowDenoiserV2
        from mora3.model.flow_unified import FlowUnifiedModel
        m = FlowUnifiedModel(m.native, m.annotation, flow_version=2)
        tr = FlowUnifiedTrainer(m, tr.batch, tr.pools, batch_size=1, flow_batch_size=1)
    saved = tr.state_dict()
    corrupt(m, kind)
    m.eval()
    for p in m.parameters(): p.grad = torch.ones_like(p)
    params = {k: p.clone() for k,p in m.named_parameters()}
    grads = {k: p.grad.clone() for k,p in m.named_parameters()}
    modes = [v.training for v in m.modules()]
    rng = torch.get_rng_state().clone()
    opt = copy.deepcopy(tr.optimizer.state_dict())
    with pytest.raises(ValueError, match='configuration|shared'):
        if entry == 'export': m.get_extra_state()
        elif entry == 'extra-load': m.set_extra_state(saved['model']['_extra_state'])
        elif entry == 'construct': FlowUnifiedTrainer(m,tr.batch,tr.pools,batch_size=1,flow_batch_size=1)
        elif entry == 'step': tr.step()
        elif entry == 'save': tr.state_dict()
        else: tr.load_state_dict(saved)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [v.training for v in m.modules()]
    equal(opt, tr.optimizer.state_dict())
    for k,p in m.named_parameters():
        assert torch.equal(params[k],p) and torch.equal(grads[k],p.grad)
    assert tr.steps == 0


def test_staged_corrupt_parent_metadata_is_atomic():
    tr=build(); before=tr.state_dict(); bad=copy.deepcopy(before)
    bad['model']['_extra_state']['flow']['native']['width']=99
    key=next(k for k,v in bad['model'].items() if isinstance(v,torch.Tensor))
    bad['model'][key].fill_(123)
    with pytest.raises(ValueError,match='configuration'):tr.load_state_dict(bad)
    equal(before,tr.state_dict())


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_preflight_preserves_flow_equations_gradients_aliases_and_rng(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(37201)
    tr=build(device);m=tr.model.to(dtype=dtype);ref=copy.deepcopy(m)
    rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    m.validate_configuration()
    tasks=tr.pools['response'][:1]
    x=torch.ones(1,2,device=device,dtype=dtype,requires_grad=True)
    t=x.new_tensor([.3]);out=m.flow(tasks,x,t)
    y=x.detach().clone().requires_grad_()
    f=ref.flow;context=f.native.backbone([f.native.graph(tasks[0])])[0][:,0]
    expected=f.output(torch.nn.functional.silu(context+f.state(y)+f.time(t[:,None])))
    assert torch.equal(out,expected)
    out.sum().backward();expected.sum().backward()
    assert torch.equal(x.grad,y.grad)
    for p,q in zip(m.parameters(),ref.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)
    equal(m.state_dict(),ref.state_dict())
    assert m.flow.native is m.native.response
    assert m.native.image.backbone is m.native.response.backbone is m.annotation.backbone
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())


@pytest.mark.parametrize('entry',['step','construct','save','load'])
def test_nonlegacy_native_policy_refuses_at_cooperative_boundary(entry):
    from mora3.model.tokengt import TokenGT
    tr=build();saved=tr.state_dict();m=tr.model
    # Replace every shared alias coherently; owner policy must still refuse.
    replacement=TokenGT(8,2,width=4,heads=2,layers=2,scaling_policy='mup-v1')
    m.native.response.backbone=replacement
    m.native.image.backbone=replacement
    m.annotation.backbone=replacement
    tr.steps=len(tr.kinds)-1  # The next scheduled branch is stochastic flow.
    m.eval();rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='configuration'):
        if entry=='step':tr.step()
        elif entry=='construct':FlowUnifiedTrainer(m,tr.batch,tr.pools,batch_size=1,flow_batch_size=1)
        elif entry=='save':tr.state_dict()
        else:tr.load_state_dict(saved)
    assert not m.training and torch.equal(rng,torch.get_rng_state())
    assert tr.steps==len(tr.kinds)-1
