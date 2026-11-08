"""Synthetic equation fixtures; no source or geometric-frame admission."""
import copy
import hashlib
import json
import pytest
import torch
from mora3.model import invariant_atom_projection as api
from mora3.model.parameterization import parameter_ledger, width_family

AXIS = ('element:C', 'element:N', 'charge:known', 'charge:value', 'stereo:unknown')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('width', [8, 12, 16])
def test_split_equation_gradients_and_complete_inventory(device, width):
    torch.manual_seed(31701)
    m = api.InvariantAtomProjection(AXIS, width, width_identity='geometry.hidden').to(device=device, dtype=torch.float64)
    x = torch.randn(4, 5, device=device, dtype=torch.float64, requires_grad=True)
    s = torch.randn(4, width, device=device, dtype=torch.float64, requires_grad=True)
    y = m(x, s)
    # Independent concatenated affine map, including exactly one bias.
    reference = torch.cat((x, s), -1) @ torch.cat((m.finite.weight, m.sequence.weight), -1).T + m.finite.bias
    torch.testing.assert_close(y, reference, rtol=1e-12, atol=1e-12)
    variables = (x, s, m.finite.weight, m.sequence.weight, m.finite.bias)
    actual = torch.autograd.grad(y.square().sum(), variables, retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), variables)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
    ledger = parameter_ledger(m, m.parameter_declarations())
    assert ledger.total_numel == width * 5 + width * width + width
    assert len(ledger.parameters) == 3 and m.sequence.bias is None
    roles = m.parameter_declarations()
    assert roles['finite.weight'].role == 'input'
    assert [a.kind for a in roles['finite.weight'].axes] == ['width', 'finite']
    assert roles['sequence.weight'].role == 'hidden'
    assert [a.kind for a in roles['sequence.weight'].axes] == ['width', 'width']
    digest = hashlib.sha256(json.dumps(['geometry.invariant-atom.finite.v1', AXIS], separators=(',', ':')).encode()).hexdigest()
    assert roles['finite.weight'].axes[1].identity == digest
    assert {a.identity for a in roles['sequence.weight'].axes} == {'geometry.hidden'}


def test_width_family_fixed_channel_order_and_one_bias_counts():
    models = [api.InvariantAtomProjection(AXIS, w, width_identity='geometry.hidden') for w in (8, 12, 16)]
    ledgers = [parameter_ledger(m, m.parameter_declarations()) for m in models]
    family = width_family(*ledgers)
    assert family.total_numel == (112, 216, 352)
    reordered = api.InvariantAtomProjection(tuple(reversed(AXIS)), 16, width_identity='geometry.hidden')
    with pytest.raises(ValueError):
        width_family(*ledgers[:2], parameter_ledger(reordered, reordered.parameter_declarations()))

@pytest.mark.parametrize('axis,width,identity', [((),8,'g'), (('x','x'),8,'g'), (['x'],8,'g'), (('',),8,'g'), ((1,),8,'g'), (AXIS,True,'g'), (AXIS,0,'g'), (AXIS,8,''), (AXIS,8,None)])
def test_invalid_declarations_refuse(axis, width, identity):
    with pytest.raises(ValueError):
        api.InvariantAtomProjection(axis, width, width_identity=identity)

@pytest.mark.parametrize('case', ['finite_width','sequence_width','rows','rank','dtype','nan','inf','empty'])
def test_malformed_inputs_refuse_before_projection(case):
    m = api.InvariantAtomProjection(AXIS, 8, width_identity='g')
    x, s = torch.zeros(2,5), torch.zeros(2,8)
    if case == 'finite_width': x = torch.zeros(2,6)
    if case == 'sequence_width': s = torch.zeros(2,7)
    if case == 'rows': s = torch.zeros(1,8)
    if case == 'rank': x = x.unsqueeze(0)
    if case == 'dtype': x = x.double()
    if case == 'nan': x[0,0] = float('nan')
    if case == 'inf': s[0,0] = float('inf')
    if case == 'empty': x, s = x[:0], s[:0]
    calls=[]
    handles=[m.finite.register_forward_pre_hook(lambda *args: calls.append(1)), m.sequence.register_forward_pre_hook(lambda *args: calls.append(1))]
    try:
        with pytest.raises(ValueError): m(x,s)
        assert not calls
    finally:
        for h in handles: h.remove()


def test_state_order_width_identity_and_extra_state_are_strict():
    m = api.InvariantAtomProjection(AXIS,8,width_identity='g')
    state = copy.deepcopy(m.state_dict())
    clone = api.InvariantAtomProjection(AXIS,8,width_identity='g')
    clone.load_state_dict(state)
    torch.testing.assert_close(m(torch.ones(2,5),torch.ones(2,8)), clone(torch.ones(2,5),torch.ones(2,8)))
    for axis, ident in [(tuple(reversed(AXIS)),'g'), (AXIS,'other')]:
        with pytest.raises(ValueError): api.InvariantAtomProjection(axis,8,width_identity=ident).load_state_dict(state)
    extra=m.get_extra_state(); extra['finite_axis']=('poison',)
    assert m.get_extra_state()['finite_axis'] == AXIS


def test_target_metadata_has_no_projection_input_channel():
    m = api.InvariantAtomProjection(AXIS,8,width_identity='g')
    x,s=torch.ones(2,5),torch.ones(2,8)
    baseline=m(x,s).detach().clone()
    for key in ['target_coordinates','target_metadata','reference_coordinates','frame_registration']:
        with pytest.raises(TypeError): m(x,s,**{key:torch.randn(2,3)})
    torch.testing.assert_close(m(x,s), baseline, rtol=0, atol=0)
    # This leaf cannot certify caller features are target-free or frame-registered.
    assert m.get_extra_state()['feature_semantics'] == 'caller-declared-invariants-only'


def test_inventory_refuses_mutated_operator_or_extra_parameter():
    m=api.InvariantAtomProjection(AXIS,8,width_identity='g')
    m.sequence=torch.nn.Linear(8,8,bias=True)
    with pytest.raises(ValueError): m.parameter_declarations()
    m=api.InvariantAtomProjection(AXIS,8,width_identity='g')
    m.register_parameter('extra',torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(ValueError): m.parameter_declarations()
