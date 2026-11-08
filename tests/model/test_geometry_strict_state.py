"""Opt-in full-state recovery; legacy geometry snapshot remains unchanged."""
import copy
from dataclasses import replace
import pytest
import torch
from test_geometry_joint_trainer import setup, response_task, equal, deterministic
from mora3.model import cooperative_training as module


def make(tmp_path, device):
    torch.manual_seed(13101)
    task,native,request=setup(tmp_path,device)
    model=module.GeometryCooperativeModel(('A','B'),width=8,id_width=16).to(device)
    return module.StrictGeometryCooperativeTrainer(model,
        {'image':(task,), 'response':(response_task(),)},
        image_requests=(request,),native=native,batch_size=1,
        prediction_asset='declared.pt')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_strict_modes_gradients_and_tail(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=make(tmp_path,device);t.step()
    t.model.eval();t.model.response.train()
    state=t.snapshot(); untouched=copy.deepcopy(state)
    assert state['version']==2
    expected=t.step();final=t.snapshot()
    t.restore(state)
    assert {n:m.training for n,m in t.model.named_modules()}==state['modes']
    for n,p in t.model.named_parameters():equal(p.grad,state['gradients'][n])
    assert t.step()==expected
    equal(t.snapshot(),final);equal(state,untouched)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('bad_field',['mode','missing_gradient','nan_gradient','shape','dtype','sparse','version'])
def test_strict_corruption_rejects_without_mutation(tmp_path,device,bad_field):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=make(tmp_path,device);t.step();state=t.snapshot();t.step()
    before=t.snapshot();bad=copy.deepcopy(state)
    name=next(n for n,g in bad['gradients'].items() if g is not None)
    if bad_field=='mode':bad['modes']['']=1
    elif bad_field=='missing_gradient':del bad['gradients'][name]
    elif bad_field=='nan_gradient':bad['gradients'][name].fill_(float('nan'))
    elif bad_field=='shape':bad['gradients'][name]=torch.zeros(1,device=device)
    elif bad_field=='dtype':bad['gradients'][name]=bad['gradients'][name].double()
    elif bad_field=='sparse':bad['gradients'][name]=bad['gradients'][name].to_sparse()
    else:bad['version']=True
    with pytest.raises(ValueError):t.restore(bad)
    equal(t.snapshot(),before)


@pytest.mark.parametrize('change',['request','pool','batch','precision','threads','asset'])
def test_live_binding_drift_rejects_before_update(tmp_path,change):
    t=make(tmp_path,'cpu');t.step();state=t.snapshot()
    threads=torch.get_num_threads()
    if change=='request':t.requests=(replace(t.requests[0],supported_objects=('nucleus',)),)
    elif change=='pool':t.pools['response']=()
    elif change=='batch':t.batch_size=2
    elif change=='precision':t.model.double()
    elif change=='threads':torch.set_num_threads(threads+1)
    else:t.prediction_asset='different.pt'
    weights=copy.deepcopy(t.model.state_dict());adam=copy.deepcopy(t.optimizer.state_dict())
    rng=torch.get_rng_state().clone();steps=t.steps
    try:
        for action in [t.snapshot,lambda:t.restore(state),t.step]:
            with pytest.raises(ValueError):action()
        equal(weights,t.model.state_dict());equal(adam,t.optimizer.state_dict())
        equal(rng,torch.get_rng_state());assert t.steps==steps
    finally:torch.set_num_threads(threads)
