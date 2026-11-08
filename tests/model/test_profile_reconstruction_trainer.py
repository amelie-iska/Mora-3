import copy
import pytest
import torch
from test_profile_latent_decoder import build
from test_molecule_profile_model import view
from mora3.model import profile_reconstruction as module


def make(device):
    m,p=build(device)
    vs=(view('profile_to_molecule'),)
    b=module.FixedProfileReconstruction(m,p,vs,seed=12301)
    assert hasattr(module,'ProfileReconstructionTrainer'), 'strict reconstruction trainer missing'
    return module.ProfileReconstructionTrainer(m,b)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_adam_and_recovery(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=make(device);ref=copy.deepcopy(t.model);opt=torch.optim.Adam(ref.parameters(),lr=.001)
    loss=t.batch.objective(ref);loss.backward();opt.step()
    r=t.step();assert r['loss']==loss.item()
    assert module._digest(t.model.state_dict())==module._digest(ref.state_dict())
    state=t.state_dict();expected=t.step();digest=module._digest(t.state_dict())
    t.load_state_dict(state)
    assert t.step()==expected
    assert module._digest(t.state_dict())==digest

@pytest.mark.parametrize('mutation',['moment','model','step','mode','grad','rng','binding','slot','dtype'])
def test_rejection_before_mutation(mutation):
    t=make('cpu');t.step();state=t.state_dict();before=module._digest(state)
    if mutation=='moment':next(iter(state['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
    if mutation=='model':next(v for v in state['model'].values() if isinstance(v,torch.Tensor)).fill_(float('nan'))
    if mutation=='step':state['steps']=True
    if mutation=='mode':state['modes']['']=1
    if mutation=='grad':next(g for g in state['gradients'].values() if g is not None).fill_(float('nan'))
    if mutation=='rng':state['cpu_rng']=torch.zeros(3,dtype=torch.uint8)
    if mutation=='binding':state['binding']['rows']=999
    if mutation=='slot':state['optimizer']['state'].clear()
    if mutation=='dtype':
        name=next(k for k,v in state['model'].items() if isinstance(v,torch.Tensor))
        state['model'][name]=state['model'][name].float()
    with pytest.raises((ValueError,RuntimeError)):t.load_state_dict(state)
    assert module._digest(t.state_dict())==before


def test_runtime_optimizer_and_source_drift_rejected():
    t=make('cpu');t.optimizer.param_groups[0]['lr']=.4
    with pytest.raises(ValueError):t.step()
