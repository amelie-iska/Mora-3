import copy
import pytest
import torch
from test_reconstruction_schedule import build
from mora3.model.profile_reconstruction import _digest

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_archive_included_detached_and_roundtrip(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=build(tmp_path,device);t.steps=13;t.step()
    state=t.state_dict()
    assert 'reconstruction_archive' in state, 'checkpoint omits fixed latent archive'
    assert _digest(state['reconstruction_archive'])==_digest(t.reconstruction_batch.state_dict())
    saved=copy.deepcopy(state)
    state['reconstruction_archive']['latents'].add_(1)
    assert _digest(saved['reconstruction_archive'])==_digest(t.reconstruction_batch.state_dict())
    t.steps=27;expected=t.step();after=_digest(t.state_dict())
    t.load_state_dict(saved);t.steps=27
    assert t.step()==expected
    assert _digest(t.state_dict())==after

@pytest.mark.parametrize('mutation',['missing','latents','binding','extra'])
def test_archive_rejected_before_mutation(tmp_path,mutation):
    t=build(tmp_path);t.steps=13;t.step();state=t.state_dict();before=_digest(state)
    assert 'reconstruction_archive' in state, 'checkpoint omits fixed latent archive'
    if mutation=='missing':del state['reconstruction_archive']
    if mutation=='latents':state['reconstruction_archive']['latents'].add_(1)
    if mutation=='binding':state['reconstruction_archive']['binding']['rows']+=1
    if mutation=='extra':state['reconstruction_archive']['unexpected']=True
    with pytest.raises(ValueError,match='archive'):t.load_state_dict(state)
    assert _digest(t.state_dict())==before
