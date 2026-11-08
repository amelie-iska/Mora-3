import copy
import pytest
import torch
from test_rfam_family_objective import model, view
from mora3.model.rfam_family_training import RfamFamilyBatch, RfamFamilyTrainer


def batch():return RfamFamilyBatch((view(),),(view('generation'),))


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_adam_reference_and_full_state_recovery(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(10001);m=model(device);ref=copy.deepcopy(m)
    t=RfamFamilyTrainer(m,batch());opt=torch.optim.Adam(ref.parameters(),lr=.001)
    for direction in ('prediction','generation'):
        opt.zero_grad();loss=getattr(ref,direction+'_objective')(getattr(t.batch,direction));loss.backward();opt.step()
        got=t.step();assert got['loss']==loss.item() and got['direction']==direction
        assert equal(m.state_dict(),ref.state_dict()) and equal(t.optimizer.state_dict(),opt.state_dict())
    saved=t.state_dict();tail=t.step();expected=t.state_dict()
    t.load_state_dict(saved);assert t.step()==tail;assert equal(t.state_dict(),expected)


@pytest.mark.parametrize('bad',['binding','steps','model','adam_shape','adam_nan','rng','schedule'])
def test_rejection_is_transactional(bad):
    t=RfamFamilyTrainer(model(),batch());t.step();before=t.state_dict();s=copy.deepcopy(before)
    if bad=='binding':s['binding']='wrong'
    if bad=='steps':s['steps']=-1
    if bad=='model':s['model']['family_output.weight']=torch.zeros(1)
    if bad=='adam_shape':next(iter(s['optimizer']['state'].values()))['exp_avg']=torch.zeros(1)
    if bad=='adam_nan':next(iter(s['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
    if bad=='rng':s['cpu_rng']=torch.zeros(1,dtype=torch.uint8)
    if bad=='schedule':s['binding']='generation-first'
    with pytest.raises((ValueError,RuntimeError)):t.load_state_dict(s)
    assert equal(before,t.state_dict())


def test_exact_bytes_order_and_configuration_binding():
    b=batch();v=view();changed=type(v)(v.condition_bytes+b' ',v.target_bytes,v.provenance_bytes)
    assert b.binding()!=RfamFamilyBatch((changed,),b.generation).binding()
    with pytest.raises(ValueError):RfamFamilyBatch(b.generation,b.prediction)
    t=RfamFamilyTrainer(model(),b);s=t.state_dict()
    other=RfamFamilyTrainer(model(),RfamFamilyBatch((changed,),b.generation))
    with pytest.raises(ValueError):other.load_state_dict(s)


def test_legacy_import_requires_exact_binding_and_policy():
    t=RfamFamilyTrainer(model(),batch());t.step();t.step();s=t.state_dict()
    old=dict(model=s['model'],adam=s['optimizer'],step=4,cpu_rng=s['cpu_rng'],cuda_rng=None,
             binding=t.batch.legacy_binding(),schedule=('prediction','generation'),seed=9901,device='cpu',dtype='FP32')
    t.import_i99_state(old);assert t.steps==4
    old['binding']['prediction']='wrong';before=t.state_dict()
    with pytest.raises(ValueError):t.import_i99_state(old)
    assert equal(before,t.state_dict())


def test_changed_runtime_policy_rejects_restore_without_mutation():
    t=RfamFamilyTrainer(model(),batch());s=t.state_dict()
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(not previous)
        with pytest.raises(ValueError):t.load_state_dict(s)
        assert equal(s,t.state_dict())
    finally:torch.use_deterministic_algorithms(previous)
