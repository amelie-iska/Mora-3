import copy
import json
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model
from test_native_tox21 import fixture, view
from mora3.model.tox21 import SharedTox21Model


def api():
    from mora3.model import tox21_training
    return tox21_training


def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_adam_reference_and_full_state_replay(device):
    mod=api()
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(8801)
    m=SharedTox21Model(model(device).shared)
    v=view(fixture()[0]);batch=mod.Tox21Batch((v,))
    t=mod.Tox21Trainer(m,batch)
    ref=copy.deepcopy(m);opt=torch.optim.Adam(ref.parameters(),lr=.001)
    opt.zero_grad();loss=ref.objective(ref(batch.views),batch.views);loss.backward();opt.step()
    assert t.step()['total']==float(loss.detach())
    equal(m.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())
    state=t.state_dict();tail=t.step();end=t.state_dict()
    t.load_state_dict(state);assert t.step()==tail;equal(t.state_dict(),end)
    bad=copy.deepcopy(state);bad['binding']='bad';before=t.state_dict()
    with pytest.raises(ValueError):t.load_state_dict(bad)
    equal(t.state_dict(),before)
    bad=copy.deepcopy(state);bad['model']['output.weight']=torch.zeros(1,device=device)
    with pytest.raises((ValueError,RuntimeError)):t.load_state_dict(bad)
    equal(t.state_dict(),before)


def test_native_binding_rejects_eval_inverse_and_binds_actual_bytes():
    mod=api();v=view(fixture()[0]);b=mod.Tox21Batch((v,))
    for row in (fixture(split='test')[0],fixture('source_Tox21_observed_call_pattern_to_molecule_prior')[0]):
        with pytest.raises(ValueError):mod.Tox21Batch((view(row),))
    for field in ('input_bytes','target_bytes','provenance_bytes'):
        data=json.loads(getattr(v,field));data['mutation']=True
        changed=replace(v,**{field:json.dumps(data).encode()})
        assert mod.Tox21Batch((changed,)).binding()!=b.binding()
    with pytest.raises(ValueError):mod.Tox21Batch(())
    m=SharedTox21Model(model().shared)
    for lr in (0,-1,float('nan')):
        with pytest.raises(ValueError):mod.Tox21Trainer(m,b,lr=lr)
