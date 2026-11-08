import copy
import hashlib
import json
from dataclasses import replace
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_desired_flow import DesiredResponseFlow
from mora3.model.response_sampling import ResponseContext
from mora3.model.response_desired import DesiredExpressionOrdinal


def setup_case(device='cpu'):
    torch.manual_seed(26301)
    h = DesiredResponseFlow(SciPlexSharedModel(('A','B'), width=4)).to(device).double()
    c = ResponseContext.from_task(task())
    d = DesiredExpressionOrdinal(c, ((0,'A',0.,1.),(1,'B',None,None)))
    return h,c,d


def sample(h,c,d,**kw):
    from mora3.model.response_desired_sampling import sample_desired_response
    args = dict(steps=4,seed=26301,sigma_min=.01,negative_policy='preserve')
    args.update(kw)
    return sample_desired_response(h,[c],requests=[d],**args)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_euler_reference_provenance_modes_rng_and_state(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    h,c,d = setup_case(device)
    h.train();h.flow.output.eval()
    modes=[m.training for m in h.modules()]
    state=copy.deepcopy(h.state_dict())
    cpu=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=sample(h,c,d)
    assert modes==[m.training for m in h.modules()]
    assert torch.equal(cpu,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,h.state_dict()[k])
    gen=torch.Generator(device=device).manual_seed(26301)
    x=torch.randn((1,2),device=device,dtype=torch.float64,generator=gen)
    h.eval()
    with torch.no_grad():
        for i in range(4):
            t=x.new_full((1,),i/4)
            # Independent Euler recurrence and denoiser/velocity equation.
            context=h.native.backbone([c.graph(h.native)])[0][:,0]
            z=x.new_tensor([[1.,1.,1.,0.,0.,0.,1.,0.]])
            hidden=context+torch.nn.functional.linear(z,h.request.weight)
            hidden=hidden+torch.nn.functional.linear(x,h.flow.state.weight,h.flow.state.bias)+torch.nn.functional.linear(t[:,None],h.flow.time.weight,h.flow.time.bias)
            den=torch.nn.functional.linear(torch.nn.functional.silu(hidden),h.flow.output.weight,h.flow.output.bias)
            x=x+(den-.99*x)/(1-.99*t[:,None])/4
    torch.testing.assert_close(r['log_expression'],x,rtol=1e-12,atol=1e-12)
    assert r['counts'] is None and not r['log_expression'].requires_grad
    assert r['binding']==hashlib.sha256(json.dumps(r['configuration'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    assert r['prediction_sha256']==hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
    assert r['model_state_sha256']==r['configuration']['model_state_sha256']
    assert r['configuration']['requests'][0]['intervals']==d.intervals
    assert r['configuration']['roles']=='user-requested-not-observed'
    assert torch.equal(sample(h,c,d)['log_expression'],r['log_expression'])
    clamped=sample(h,c,d,negative_policy='clamp-zero')
    torch.testing.assert_close(clamped['counts'],torch.expm1(x.clamp_min(0)))
    changed=DesiredExpressionOrdinal(c,((0,'A',2.,3.),))
    assert sample(h,c,changed)['binding']!=r['binding']
    assert not torch.equal(sample(h,c,changed)['log_expression'],r['log_expression'])
    with torch.no_grad():h.request.weight.add_(.01)
    assert sample(h,c,d)['model_state_sha256']!=r['model_state_sha256']


@pytest.mark.parametrize('kw',[dict(steps=0),dict(steps=True),dict(seed=-1),dict(seed=True),dict(sigma_min=.1),dict(sigma_min=True),dict(negative_policy='auto')])
def test_invalid_solver_refuses(kw):
    h,c,d=setup_case()
    with pytest.raises(ValueError):sample(h,c,d,**kw)


def test_context_target_and_model_config_refusals():
    h,c,d=setup_case()
    with pytest.raises(ValueError):sample(h,replace(c,source_sha256='f'*64),d)
    with pytest.raises(ValueError):sample(h,task(),d)
    h.config['version']=9
    with pytest.raises(ValueError):sample(h,c,d)


def test_failure_restores_mixed_modes_and_rng():
    h,c,d=setup_case()
    h.train();h.flow.output.eval()
    modes=[m.training for m in h.modules()];rng=torch.get_rng_state().clone()
    # Finite weights with an overflowing affine force solver failure after eval.
    with torch.no_grad():h.flow.output.weight.fill_(1e308)
    with pytest.raises(ValueError):sample(h,c,d)
    assert modes==[m.training for m in h.modules()]
    assert torch.equal(rng,torch.get_rng_state())
