"""Actual native assay fixture ABIs, nonunit source-head readout equations."""
import copy
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from mora3.model.tox21 import SharedTox21Model
from mora3.model.lincs_model import SharedLincsModel
from mora3.model.assay_mup import assay_mup_parameter_roles,assay_mup_width_family
from test_annotation_mup import make as annotation
from test_lincs_native import cases,project,api
from test_native_tox21 import fixture as tox_fixture,view as tox_view


def lincs(width,cases):
    views=tuple(project(api(),c) for c in cases.values())
    axes={v.modality:c[7] for v,c in zip(views,cases.values())}
    return SharedLincsModel(annotation(width,max_tokens=8192),axes,parameterization_policy='mup-lincs-v1'),views

def tox(width):return SharedTox21Model(annotation(width,max_tokens=8192),parameterization_policy='mup-tox21-v1')


@pytest.mark.parametrize('kind',['tox21','lincs'])
def test_complete_typed_assay_owner_families_and_exact_source_axes(cases,kind):
    ms=[(tox(w) if kind=='tox21' else lincs(w,cases)[0]) for w in (8,12,16)]
    family=assay_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    roles=assay_mup_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    assert not any(r.role=='custom' for r in roles.values())
    if kind=='lincs':assert roles['profile_mask.L1000.weight'].role=='input'
    else:assert roles['call_input.weight'].role=='input'


def test_tox21_actual_nonunit_readout_all_gradients_and_target_exclusion():
    torch.manual_seed(2026110680);m=tox(16).double();v=tox_view(tox_fixture()[0]);vs=(v,)
    latent=m.condition(vs)
    expected=F.linear(latent*(1.5/(16/8)),m.output.weight,m.output.bias);actual=m(vs)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True,allow_unused=True),
                   torch.autograd.grad(expected.sum(),params,allow_unused=True)):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    altered=replace(v,target_bytes=b'{}',provenance_bytes=b'{}')
    assert torch.equal(actual,m((altered,)))


def test_lincs_actual_native_axes_nonunit_readouts_and_all_gradients(cases):
    torch.manual_seed(2026110681);m,views=lincs(16,cases);m=m.double()
    views=tuple(v for v in views if v.direction=='molecule_to_profile')
    context=m.condition(views)
    expected=tuple(F.linear(c*(1.5/(16/8)),m.output[v.modality].weight,m.output[v.modality].bias) for c,v in zip(context,views))
    actual=m(views)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(sum(x.sum() for x in actual),params,retain_graph=True,allow_unused=True),
                   torch.autograd.grad(sum(x.sum() for x in expected),params,allow_unused=True)):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('kind',['tox21','lincs'])
def test_source_policy_checkpoint_refuses_before_shared_tensor_copy(cases,kind):
    m=tox(16) if kind=='tox21' else lincs(16,cases)[0]
    state=copy.deepcopy(m.state_dict());state['_extra_state']['mup']['base_width']=4
    state['shared.payload.embedding.weight'].fill_(99)
    before={n:p.clone() for n,p in m.named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):m.load_state_dict(state)
    for n,p in m.named_parameters():assert torch.equal(p,before[n])
