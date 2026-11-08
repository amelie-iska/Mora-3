import copy
import importlib
import pytest
import torch
from test_profile_latent_decoder import build
from test_molecule_profile_model import view

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_draw_binding_and_reference_update(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    assert importlib.util.find_spec('mora3.model.profile_reconstruction') is not None, 'source-bound reconstruction missing'
    from mora3.model.profile_reconstruction import FixedProfileReconstruction
    torch.manual_seed(12101)
    m,p=build(device);vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    rng=torch.get_rng_state().clone()
    b=FixedProfileReconstruction(m,p,vs,seed=12101)
    assert torch.equal(rng,torch.get_rng_state())
    assert not b.latents().requires_grad
    duplicate=FixedProfileReconstruction(m,p,vs,seed=12101)
    assert b.binding()==duplicate.binding()
    assert torch.equal(b.latents(),duplicate.latents())
    x=b.latents();x.zero_();assert not torch.equal(x,b.latents())
    ref=copy.deepcopy(m);opt=torch.optim.Adam(m.parameters(),lr=.001)
    ropt=torch.optim.Adam(ref.parameters(),lr=.001)
    expected=ref.objective(vs,b.latents());expected.backward();ropt.step()
    actual=b.objective(m);actual.backward();opt.step()
    assert actual.item()==expected.item()
    for a,z in zip(m.parameters(),ref.parameters()):assert torch.equal(a,z)
    # Policy snapshot is external and stays fixed, even as shared encoder trains.
    assert torch.isfinite(b.objective(m))
    with pytest.raises(ValueError,match='batch'):b.objective(m,views=vs[::-1])
    with torch.no_grad():p.forward_head.weight.add_(1)
    with pytest.raises(ValueError,match='policy'):b.validate_policy(p)

@pytest.mark.parametrize('mutation',['targets','source','decoder','latents'])
def test_tamper_rejected(mutation):
    assert importlib.util.find_spec('mora3.model.profile_reconstruction') is not None, 'source-bound reconstruction missing'
    from mora3.model.profile_reconstruction import FixedProfileReconstruction
    from dataclasses import replace
    m,p=build('cpu');vs=(view('profile_to_molecule'),)
    b=FixedProfileReconstruction(m,p,vs,seed=12101)
    if mutation=='targets':
        with pytest.raises(ValueError,match='batch'):b.objective(m,views=(replace(vs[0],target_bytes=b'changed'),))
    elif mutation=='source':
        m._policy['source_binding']={'changed':True}
        with pytest.raises(ValueError,match='configuration'):b.objective(m)
    elif mutation=='decoder':
        m.decoder.max_tokens+=1
        with pytest.raises(ValueError,match='configuration'):b.objective(m)
    else:
        b._latents.add_(1)
        with pytest.raises(ValueError,match='latent'):b.objective(m)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_batch_restore_without_sampling(device, monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model.profile_reconstruction import FixedProfileReconstruction
    import mora3.model.profile_reconstruction as module
    m,p=build(device);vs=(view('profile_to_molecule'),)
    b=FixedProfileReconstruction(m,p,vs,seed=12201)
    assert hasattr(b,'state_dict'), 'fixed batch persistence missing'
    state=b.state_dict()
    def forbidden(*args,**kwargs):raise AssertionError('restore must not resample')
    monkeypatch.setattr(module,'sample_paths',forbidden)
    rng=torch.get_rng_state().clone()
    restored=FixedProfileReconstruction.from_state_dict(m,p,vs,state)
    assert torch.equal(rng,torch.get_rng_state())
    assert restored.binding()==b.binding()
    assert torch.equal(restored.latents(),b.latents())
    assert torch.equal(restored.objective(m),b.objective(m))
    state['latents'].zero_()
    assert torch.equal(restored.latents(),b.latents())

@pytest.mark.parametrize('mutation',['latent','binding','policy','order','shape','nonfinite'])
def test_fixed_batch_restore_rejects_tampering(mutation):
    from mora3.model.profile_reconstruction import FixedProfileReconstruction
    m,p=build('cpu');vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    b=FixedProfileReconstruction(m,p,vs,seed=12201)
    assert hasattr(b,'state_dict'), 'fixed batch persistence missing'
    state=b.state_dict();before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    if mutation=='latent':state['latents'].add_(1)
    if mutation=='binding':state['binding']['seed']+=1
    if mutation=='policy':
        with torch.no_grad():p.forward_head.weight.add_(1)
        before=copy.deepcopy(m.state_dict())
    if mutation=='order':vs=vs[::-1]
    if mutation=='shape':state['latents']=state['latents'][:1]
    if mutation=='nonfinite':state['latents'][0,0]=float('nan')
    with pytest.raises(ValueError):FixedProfileReconstruction.from_state_dict(m,p,vs,state)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,m.state_dict()[k])
