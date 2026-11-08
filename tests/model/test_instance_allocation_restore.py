import pytest
import torch
from test_instance_optimizer import boundary
from test_instance_schedule import make, cases
from test_color_optimizer import fingerprint


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_restore_values_preserves_all_runtime_ownership(tmp_path,cases,device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    b=boundary(make(tmp_path,cases,device))
    assert hasattr(b,'load_allocation_state_dict'), 'missing transactional allocation restoration'
    saved=b.allocation_state_dict()
    with torch.no_grad():
        for p in b.model.parameters():p.add_(.25)
    b.model.eval();b.model.instance.train()
    next(b.model.parameters()).grad=torch.ones_like(next(b.model.parameters()))
    ids=tuple(map(id,b.model.parameters())); modules=tuple(map(id,b.model.modules()))
    modes=[m.training for m in b.model.modules()]
    grads=[None if p.grad is None else p.grad.clone() for p in b.model.parameters()]
    cpu=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    monkeypatch.setattr(b.model,'load_state_dict',lambda *a,**kw:pytest.fail('model loader hook called'))
    b.load_allocation_state_dict(saved)
    for name,value in b.model.state_dict().items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,saved['model'][name])
    assert ids==tuple(map(id,b.model.parameters()))
    assert modules==tuple(map(id,b.model.modules()))
    assert modes==[m.training for m in b.model.modules()]
    for p,g in zip(b.model.parameters(),grads):
        assert (p.grad is None) if g is None else torch.equal(p.grad,g)
    assert torch.equal(cpu,torch.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state())
    before=fingerprint(b)
    saved['model'][next(iter(saved['model']))] = None
    assert fingerprint(b)==before
    with pytest.raises(ValueError,match='admission'):b.step()
    assert fingerprint(b)==before


@pytest.mark.parametrize('change',['schema','provider','adam','alias','nan','shape'])
def test_restore_rejects_complete_invalid_candidate_before_mutation(tmp_path,cases,change):
    b=boundary(make(tmp_path,cases))
    assert hasattr(b,'load_allocation_state_dict'), 'missing transactional allocation restoration'
    s=b.allocation_state_dict()
    # Distinct valid prefix values expose accidental partial writes on rejection.
    for value in s['model'].values():
        if isinstance(value,torch.Tensor) and value.is_floating_point():value.add_(.25)
    if change=='schema':s['extra']=1
    elif change=='provider':s['provider']['weight']=10.
    elif change=='adam':s['optimizer']['state'][0]={'step':torch.tensor(1.)}
    elif change=='alias':s['parameter_aliases']=[]
    else:
        k=[k for k,v in s['model'].items() if isinstance(v,torch.Tensor)][-1]
        s['model'][k]=torch.full_like(s['model'][k],float('nan')) if change=='nan' else torch.zeros(1)
    before=fingerprint(b)
    with pytest.raises(ValueError):b.load_allocation_state_dict(s)
    assert fingerprint(b)==before


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_copy_failure_rolls_back_all_values(tmp_path,cases,device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    b=boundary(make(tmp_path,cases,device))
    assert hasattr(b,'load_allocation_state_dict'), 'missing transactional allocation restoration'
    s=b.allocation_state_dict()
    with torch.no_grad():
        for p in b.model.parameters():p.add_(.5)
    before=fingerprint(b); original=torch.Tensor.copy_;calls=[]
    def fail_once(self,other,*a,**kw):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('injected copy failure')
        return original(self,other,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'copy_',fail_once)
    with pytest.raises(RuntimeError,match='injected'):b.load_allocation_state_dict(s)
    assert len(calls)>3
    assert fingerprint(b)==before
