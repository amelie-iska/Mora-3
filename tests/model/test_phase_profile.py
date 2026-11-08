import copy
import pytest
import torch
from mora3.model.phase_profile import preserved_training_state, measure_phase

class Trainer:
    def __init__(self):
        self.model=torch.nn.Linear(2,1)
        self.optimizer=torch.optim.Adam(self.model.parameters())
        self.steps=0
    def state_dict(self):
        return copy.deepcopy(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),steps=self.steps,rng=torch.get_rng_state()))
    def load_state_dict(self,s):
        self.model.load_state_dict(s['model']);self.optimizer.load_state_dict(s['optimizer']);self.steps=s['steps'];torch.set_rng_state(s['rng'])

@pytest.mark.parametrize('fail',[False,True])
def test_restores_state_gradients_modes_and_rng(fail):
    t=Trainer();t.model.eval()
    t.model.weight.grad=torch.ones_like(t.model.weight)
    before=t.state_dict(); grad=t.model.weight.grad.clone();rng=torch.get_rng_state()
    try:
        with preserved_training_state(t):
            t.model.train();t.optimizer.zero_grad();t.model(torch.randn(3,2)).sum().backward();t.optimizer.step();t.steps+=1
            if fail: raise RuntimeError('probe')
    except RuntimeError:
        assert fail
    assert t.steps==0 and not t.model.training
    assert torch.equal(t.model.weight.grad,grad) and t.model.bias.grad is None
    assert torch.equal(torch.get_rng_state(),rng)
    for k,v in before['model'].items():assert torch.equal(v,t.model.state_dict()[k])
    assert t.optimizer.state_dict()==before['optimizer']

def test_measurement_returns_value_and_finite_seconds():
    value,record=measure_phase('forward',lambda:torch.ones(2).sum(),device='cpu')
    assert value==2 and record['phase']=='forward' and record['seconds']>=0
    assert record['device']=='cpu' and record['peak_allocated_bytes'] is None

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_rng_and_measurement_restored():
    t=Trainer();t.model.cuda()
    rng=torch.cuda.get_rng_state().clone()
    with preserved_training_state(t):
        value,record=measure_phase('cuda_forward',lambda:t.model(torch.randn(3,2,device='cuda')).sum(),device='cuda')
        value.backward();t.optimizer.step()
    assert torch.equal(torch.cuda.get_rng_state(),rng)
    assert record['peak_allocated_bytes']>0 and record['seconds']>=0
    assert all(p.grad is None for p in t.model.parameters())
