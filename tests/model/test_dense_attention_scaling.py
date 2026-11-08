"""Independent ordinary-attention scaling equations; synthetic tests only."""
import copy
import math

import pytest
import torch
from mora3.model.tokengt import DenseAttention


def make(width=8, heads=2, **kwargs):
    assert hasattr(DenseAttention, 'scaling_config'), 'missing versioned scaling interface'
    return DenseAttention(width, heads, **kwargs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('width', [8, 12, 16])
def test_mup_equation_mask_and_all_gradients(device, dtype, width):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(36801)
    a = make(width, scaling_policy='mup-v1', attn_mult=2.25).to(device=device, dtype=dtype)
    x = (torch.randn(2, 4, width, device=device, dtype=dtype) / 4).requires_grad_()
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], device=device, dtype=torch.bool)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    config = a.scaling_config()
    assert config == dict(schema='dense-attention-scaling-v1', width=width, heads=2,
                          scaling_policy='mup-v1', attn_mult=2.25)
    # Independently split each projection matrix and each head.
    q, k, v = [x @ w.T + b for w, b in zip(a.qkv.weight.chunk(3), a.qkv.bias.chunk(3))]
    rows = []
    d = width // 2
    for batch in range(2):
        hs = []
        for h in range(2):
            sl = slice(h*d, (h+1)*d)
            logits = (q[batch, :, sl] * (1.5 / d)) @ k[batch, :, sl].T
            weights = logits.masked_fill(~valid[batch][None], -torch.inf).softmax(-1)
            assert torch.count_nonzero(weights[:, ~valid[batch]]) == 0
            hs.append(weights @ v[batch, :, sl])
        rows.append(torch.cat(hs, -1) @ a.output.weight.T + a.output.bias)
    expected = torch.stack(rows)
    actual = a(x, valid)
    tol = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    variables = (x, *a.parameters())
    probe = torch.linspace(-.5, .8, actual.numel(), device=device, dtype=dtype).reshape_as(actual)
    ga = torch.autograd.grad((actual*probe).sum(), variables, retain_graph=True)
    ge = torch.autograd.grad((expected*probe).sum(), variables)
    for aa, ee in zip(ga, ge):
        torch.testing.assert_close(aa, ee, atol=tol, rtol=tol)
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in a.state_dict().items())
    assert all(p.grad is None for p in a.parameters())
    # Explicit config plus strict state reconstruction: never infer policy from tensors.
    restored = DenseAttention.from_scaling_config(config).to(device=device, dtype=dtype)
    restored.load_state_dict(a.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, valid), actual, atol=0, rtol=0)
    changed = x.detach().clone(); changed[0, 2:] += 10
    torch.testing.assert_close(a(changed, valid)[0, :2], actual[0, :2], atol=tol, rtol=tol)


def test_legacy_defaults_exact_and_state_keys_unchanged():
    torch.manual_seed(36801)
    old = DenseAttention(8, 2).double()
    new = make(scaling_policy='legacy').double()
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(1, 3, 8, dtype=torch.float64)
    valid = torch.ones(1, 3, dtype=torch.bool)
    q, k, v = old.qkv(x).reshape(1, 3, 3, 2, 4).unbind(2)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))
    expected = old.output((((q @ k.transpose(-1, -2))/2).softmax(-1) @ v).transpose(1,2).reshape(1,3,8))
    assert torch.equal(old(x, valid), expected)
    assert torch.equal(old(x, valid), new(x, valid))
    assert set(new.state_dict()) == {'qkv.weight', 'qkv.bias', 'output.weight', 'output.bias'}
    assert new.scaling_config()['attn_mult'] == 1.0


@pytest.mark.parametrize('kwargs', [
    {'scaling_policy':'tropical-code'}, {'scaling_policy':'mup'},
    {'attn_mult':0}, {'attn_mult':-1}, {'attn_mult':float('nan')},
    {'attn_mult':float('inf')}, {'attn_mult':True}, {'attn_mult':'2'},
    {'attn_mult':2}, {'width':True}, {'heads':True}, {'width':8.0},
    {'heads':2.0}, {'width':0}, {'heads':0}, {'width':7, 'heads':2},
])
def test_bad_configuration_refuses_before_rng_mutation(kwargs):
    args = dict(width=8, heads=2); args.update(kwargs)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        make(**args)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('change', ['schema', 'unknown', 'missing', 'heads', 'multiplier'])
def test_reconstruction_rejects_invalid_config(change):
    cfg = make(scaling_policy='mup-v1').scaling_config()
    if change == 'schema': cfg['schema'] = 'future'
    if change == 'unknown': cfg['attention'] = 'tropical-code'
    if change == 'missing': del cfg['attn_mult']
    if change == 'heads': cfg['heads'] = 3
    if change == 'multiplier': cfg['attn_mult'] = False
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError): DenseAttention.from_scaling_config(cfg)
    assert torch.equal(rng, torch.random.get_rng_state())
