import copy
import pytest
import torch
from test_color_optimizer import boundary, fingerprint
from test_color_schedule import make, cases

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_snapshot_detached_exact_and_read_only(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    b=boundary(make(tmp_path,cases,device)); before=fingerprint(b)
    assert hasattr(b,'allocation_state_dict'), 'missing strict allocation snapshot'
    state=b.allocation_state_dict(); b.validate_allocation_state(state)
    assert fingerprint(b)==before
    for name,value in b.model.state_dict().items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,state['model'][name])
            assert value.data_ptr()!=state['model'][name].data_ptr()
    name=next(k for k,v in state['model'].items() if isinstance(v,torch.Tensor))
    state['model'][name].zero_()
    assert fingerprint(b)==before

@pytest.mark.parametrize('change',['historical','extra','steps','provider','optimizer','shape','dtype','nan','alias','configuration'])
def test_malformed_state_rejected_without_mutation(tmp_path,cases,change):
    b=boundary(make(tmp_path,cases));before=fingerprint(b)
    assert hasattr(b,'allocation_state_dict'), 'missing strict allocation snapshot'
    s=b.allocation_state_dict()
    if change=='historical':s={'model':s['model']}
    elif change=='extra':s['unknown']=True
    elif change=='steps':s['steps']=True
    elif change=='provider':s['provider']['weight']=99.
    elif change=='optimizer':s['optimizer']['state'][0]={'step':torch.tensor(1.)}
    elif change=='alias':s['parameter_aliases']=[]
    elif change=='configuration':s['configuration']['version']=99
    else:
        key=next(k for k,v in s['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point())
        if change=='shape':s['model'][key]=torch.zeros(1)
        if change=='dtype':s['model'][key]=s['model'][key].float()
        if change=='nan':s['model'][key].fill_(float('nan'))
    with pytest.raises(ValueError):b.validate_allocation_state(s)
    assert fingerprint(b)==before

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_load_restores_values_preserves_identity_and_runtime(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    b=boundary(make(tmp_path,cases,device));state=b.allocation_state_dict()
    assert hasattr(b,'load_allocation_state_dict'), 'missing transactional allocation loading'
    identities=[id(p) for p in b.model.parameters()]
    with torch.no_grad():
        for p in b.model.parameters():p.add_(.25)
    b.model.eval();b.model.color.train()
    next(b.model.parameters()).grad=torch.ones_like(next(b.model.parameters()))
    runtime=copy.deepcopy(([p.grad for p in b.model.parameters()],
        [m.training for m in b.model.modules()],torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None))
    from mora3.model.profile_reconstruction import _digest
    b.load_allocation_state_dict(state)
    assert _digest(b.model.state_dict())==_digest(state['model'])
    assert identities==[id(p) for p in b.model.parameters()]
    assert _digest(runtime)==_digest(([p.grad for p in b.model.parameters()],
        [m.training for m in b.model.modules()],torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None))
    assert not b.optimizer.state and b.steps==0
    before=fingerprint(b)
    with pytest.raises(ValueError):b.step()
    assert fingerprint(b)==before
    tensor=next(v for v in state['model'].values() if isinstance(v,torch.Tensor))
    tensor.zero_()
    assert fingerprint(b)==before

@pytest.mark.parametrize('change',['provider','alias','late_nan','optimizer'])
def test_load_rejects_entire_snapshot_before_any_write(tmp_path,cases,change):
    b=boundary(make(tmp_path,cases));state=b.allocation_state_dict()
    assert hasattr(b,'load_allocation_state_dict'), 'missing transactional allocation loading'
    if change=='provider':state['provider']['weight']=99.
    if change=='alias':state['parameter_aliases']=[]
    if change=='optimizer':state['optimizer']['state'][0]={'step':torch.tensor(1.)}
    if change=='late_nan':
        key=[k for k,v in state['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point()][-1]
        state['model'][key].fill_(float('nan'))
    with torch.no_grad():
        for p in b.model.parameters():p.add_(.125)
    before=fingerprint(b)
    with pytest.raises(ValueError):b.load_allocation_state_dict(state)
    assert fingerprint(b)==before
