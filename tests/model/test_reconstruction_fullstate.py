import copy
import pytest
import torch
from test_reconstruction_schedule import build
from mora3.model.profile_reconstruction import _digest

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_modes_and_gradients_roundtrip_and_next_update(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=build(tmp_path,device);t.steps=13;t.step()
    t.model.eval();t.model.decoder.train()
    state=t.state_dict()
    assert 'modes' in state and 'gradients' in state, 'full-state fields missing'
    before=_digest(state)
    t.steps=27;expected=t.step();after=_digest(t.state_dict())
    t.model.train();t.optimizer.zero_grad(set_to_none=True)
    t.load_state_dict(state)
    assert _digest(t.state_dict())==before
    t.steps=27;assert t.step()==expected
    assert _digest(t.state_dict())==after

@pytest.mark.parametrize('mutation',['missing','mode','gradkeys','shape','dtype','nan'])
def test_fullstate_corruption_rejected_transactionally(tmp_path,mutation):
    t=build(tmp_path);t.steps=13;t.step();state=t.state_dict()
    assert 'modes' in state and 'gradients' in state, 'full-state fields missing'
    before=_digest(state)
    if mutation=='missing':del state['modes']
    if mutation=='mode':state['modes']['']=1
    if mutation=='gradkeys':state['gradients']['unknown']=None
    key=next(k for k,v in state['gradients'].items() if v is not None)
    if mutation=='shape':state['gradients'][key]=torch.zeros(1)
    if mutation=='dtype':state['gradients'][key]=state['gradients'][key].float()
    if mutation=='nan':state['gradients'][key].fill_(float('nan'))
    with pytest.raises(ValueError,match='mode|gradient|Gradient'):t.load_state_dict(state)
    assert _digest(t.state_dict())==before
