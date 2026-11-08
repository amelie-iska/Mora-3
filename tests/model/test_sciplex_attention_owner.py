"""Legacy SciPlex schema must refuse policies it cannot reconstruct."""
import copy
import pytest
import torch
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.tokengt import TokenGT
from test_sciplex_shared import task


@pytest.mark.parametrize('drift', ['policy', 'multiplier', 'heads', 'layers', 'variant', 'owner'])
@pytest.mark.parametrize('entry', ['extra', 'binding', 'forward', 'graph', 'restore'])
def test_legacy_owner_refuses_drift_before_rng_or_tensor_mutation(drift, entry):
    m = SciPlexSharedModel(('A', 'B'), width=8)
    kwargs = dict(width=8, heads=2, layers=2, attention='vanilla')
    if drift == 'policy': kwargs['scaling_policy'] = 'mup-v1'
    if drift == 'multiplier': kwargs.update(scaling_policy='mup-v1', attn_mult=2.0)
    if drift == 'heads': kwargs['heads'] = 4
    if drift == 'layers': kwargs['layers'] = 1
    if drift == 'variant': kwargs['attention'] = 'tropical-code'
    if drift == 'owner': m.config['attention'] = 'tropical'
    else: m.backbone = TokenGT(8, 2, **kwargs)
    before = {k: v.detach().clone() for k,v in m.named_parameters()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='SciPlex.*configuration'):
        if entry == 'extra': m.get_extra_state()
        elif entry == 'binding': m.binding([task()])
        elif entry == 'graph': m.graph(task())
        elif entry == 'restore': m.set_extra_state(m.config.copy())
        else: m([task()])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k,v in m.named_parameters())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_legacy_metadata_validation_preserves_output_gradient_state_rng(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(37001)
    m = SciPlexSharedModel(('A','B'), width=8).to(device=device, dtype=dtype)
    reference = copy.deepcopy(m)
    expected = dict(version=1, axis_ids=('A','B'), width=8, attention='vanilla',
        policy=m.config['policy'], transform='log1p-counts', graph='context-to-observation-v1')
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    assert m.get_extra_state() == expected
    m.set_extra_state(expected)
    assert m.binding([task()])['config'] == expected
    assert torch.equal(rng, torch.get_rng_state())
    if gpu_rng is not None: assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    a,b = m.loss([task()]), reference.loss([task()])
    assert torch.equal(a,b)
    a.backward(); b.backward()
    for p,q in zip(m.parameters(), reference.parameters()):
        assert torch.equal(p,q)
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)
    fresh = SciPlexSharedModel(('A','B'), width=8).to(device=device,dtype=dtype)
    fresh.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(fresh([task()]), m([task()]))
