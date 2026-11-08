import copy
import pytest
import torch
from test_response_desired_sampling import setup_case, sample
from mora3.model.response_desired_sampling import assess_desired_sample
from mora3.model.response_desired import DesiredExpressionOrdinal


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_call_local_reuse_exact_outputs_bindings_modes_rng_and_freshness(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    h,c,d=setup_case(device)
    h.train();h.flow.output.eval()
    modes=[m.training for m in h.modules()]
    before=copy.deepcopy(h.state_dict())
    cpu=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    calls=[]
    hook=h.native.backbone.register_forward_hook(lambda *args:calls.append(1))
    try:
        reference=sample(h,c,d)
        assert len(calls)==4
        calls.clear()
        reused=sample(h,c,d,reuse_context=True)
        assert len(calls)==1
        for k in reference:
            if isinstance(reference[k],torch.Tensor):assert torch.equal(reference[k],reused[k])
            else:assert reference[k]==reused[k]
        assert assess_desired_sample(reused,[d])['sample_binding']==reused['binding']
        assert modes==[m.training for m in h.modules()]
        assert torch.equal(cpu,torch.get_rng_state())
        if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
        for k,v in before.items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,h.state_dict()[k])
        for changed in [d,DesiredExpressionOrdinal(c,((0,'A',2.,3.),))]:
            with torch.no_grad():h.request.weight.add_(.01)
            a=sample(h,c,changed,reuse_context=True)
            b=sample(h,c,changed)
            assert torch.equal(a['log_expression'],b['log_expression'])
            assert a['binding']==b['binding'] and a['binding']!=reference['binding']
        assert not hasattr(h,'cached_context')
    finally:hook.remove()


@pytest.mark.parametrize('value',[None,1,'yes',[],{}])
def test_reuse_requires_explicit_bool(value):
    h,c,d=setup_case()
    with pytest.raises(ValueError,match='bool'):sample(h,c,d,reuse_context=value)


def test_reuse_failure_restores_mixed_modes_rng():
    h,c,d=setup_case()
    h.train();h.flow.output.eval()
    modes=[m.training for m in h.modules()];rng=torch.get_rng_state().clone()
    with torch.no_grad():h.flow.output.weight.fill_(1e308)
    with pytest.raises(ValueError):sample(h,c,d,reuse_context=True)
    assert modes==[m.training for m in h.modules()]
    assert torch.equal(rng,torch.get_rng_state())
