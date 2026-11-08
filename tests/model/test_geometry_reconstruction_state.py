"""Shared parameter aliases must agree before any checkpoint commit."""
import copy
import pytest
import torch
from test_geometry_reconstruction_schedule import build
from mora3.model.profile_reconstruction import _digest

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_conflicting_shared_parameter_alias_rejected_transactionally(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    t = build(tmp_path, device)
    t.steps = 14; t.step()
    before = _digest(t.state_dict())
    state = copy.deepcopy(t.state_dict())
    aliases = {}
    for name, p in t.model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
    names = next(v for v in aliases.values() if len(v) > 1)
    state['model'][names[-1]] = state['model'][names[-1]].clone() + 1
    with pytest.raises(ValueError, match='alias'):
        t.load_state_dict(state)
    assert _digest(t.state_dict()) == before

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_geometry_fullstate_roundtrip_next_geometry_update(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    t = build(tmp_path, device)
    t.steps = 14; t.step(); t.model.eval(); t.model.decoder.train()
    state = t.state_dict(); before = _digest(state)
    t.steps = 29; expected = t.step(); final = _digest(t.state_dict())
    t.load_state_dict(state)
    assert _digest(t.state_dict()) == before
    t.steps = 29; assert t.step() == expected
    assert _digest(t.state_dict()) == final

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('corruption', ['version_bool', 'cpu_rng', 'cuda_rng', 'adam_nan',
    'adam_negative', 'adam_step', 'mode', 'gradient', 'archive'])
def test_geometry_malformed_state_is_transactional(tmp_path, device, corruption):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    t = build(tmp_path, device)
    t.steps = 14; t.step()
    t.model.eval(); t.model.decoder.train()
    before = _digest(t.state_dict())
    state = copy.deepcopy(t.state_dict())
    slot = next(iter(state['optimizer']['state'].values()))
    if corruption == 'version_bool': state['version'] = True
    elif corruption == 'cpu_rng': state['cpu_rng'] = torch.zeros(3, dtype=torch.uint8)
    elif corruption == 'cuda_rng':
        state['cuda_rng'] = [torch.zeros(3, dtype=torch.uint8)] if device == 'cuda' else []
    elif corruption == 'adam_nan': slot['exp_avg'].fill_(float('nan'))
    elif corruption == 'adam_negative': slot['exp_avg_sq'].fill_(-1)
    elif corruption == 'adam_step': slot['step'].fill_(.5)
    elif corruption == 'mode': state['modes'][''] = 1
    elif corruption == 'gradient':
        key = next(n for n,g in state['gradients'].items() if g is not None)
        state['gradients'][key] = state['gradients'][key].to_sparse()
    elif corruption == 'archive': state['reconstruction_archive'] = {}
    with pytest.raises(ValueError): t.load_state_dict(state)
    assert _digest(t.state_dict()) == before

@pytest.mark.parametrize('change', ['request', 'pool', 'precision', 'threads', 'asset', 'archive'])
def test_geometry_restore_live_drift_is_transactional(tmp_path, change):
    from dataclasses import replace
    t = build(tmp_path)
    t.steps = 14; t.step()
    state = t.state_dict()
    threads = torch.get_num_threads()
    if change == 'request': t.image_requests = (replace(t.image_requests[0], supported_objects=('nucleus',)),) + t.image_requests[1:]
    elif change == 'pool': t.pools['image'] = ()
    elif change == 'precision': t.model.float()
    elif change == 'threads': torch.set_num_threads(threads + 1)
    elif change == 'asset': t.prediction_assets = t.prediction_assets[::-1]
    elif change == 'archive': t.reconstruction_batch._latents[0,0] += 1
    def live_digest():
        return _digest(dict(model=t.model.state_dict(), adam=t.optimizer.state_dict(),
            steps=t.steps, cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            modes={n:m.training for n,m in t.model.named_modules()},
            gradients={n:p.grad for n,p in t.model.named_parameters()}))
    before = live_digest()
    try:
        with pytest.raises(ValueError): t.load_state_dict(state)
        assert live_digest() == before
    finally:
        torch.set_num_threads(threads)
