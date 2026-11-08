import copy
from dataclasses import replace
import json
import pytest
import torch
from test_flow_unified import build
from test_native_molecule_profile import fixture
from mora3.data.molecule_profile import molecule_profile_view
from mora3.model import molecule_profile as module


def view(direction='molecule_to_profile', modality='L1000'):
    row, _ = fixture(direction, modality)
    return molecule_profile_view(row, effective_groups=('group',))


def model(device='cpu'):
    shared = build(device).model.annotation
    return module.SharedMoleculeProfileModel(shared, {
        m: view(modality=m).axis for m in ('CellPainting', 'L1000')}).to(device)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_complete_axis_reference_gradients_and_target_exclusion(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(8001)
    m = model(device); views = (view(), view(modality='CellPainting'))
    assert m.shared.backbone is m.backbone
    assert len(list(m.parameters())) == len({id(p) for p in m.parameters()})
    prediction = m(views)
    assert [len(x) for x in prediction] == [977,1783]
    ref = copy.deepcopy(m)
    loss = m.objective(prediction, views)
    z = ref(views)
    expected = torch.stack([(x - x.new_full(x.shape,.25)).square().mean() for x in z]).mean()
    assert torch.equal(loss, expected)
    loss.backward(); expected.backward()
    for (name,p),(other,q) in zip(m.named_parameters(),ref.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None: assert torch.equal(p.grad,q.grad),name
    assert m.shared.payload.embedding.weight.grad.abs().sum() > 0
    assert m.backbone.feature.weight.grad.abs().sum() > 0
    assert m.shared.go.output.weight.grad is None
    changed = replace(views[0],target_bytes=views[0].target_bytes.replace(b'0.25',b'9.25'),provenance_bytes=b'{}')
    assert torch.equal(m((views[0],))[0], m((changed,))[0])
    assert m.objective(m((changed,)), (changed,)) != m.objective(m((views[0],)), (views[0],))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_inverse_condition_is_target_free_and_reaches_shared_backbone(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m=model(device); v=view('profile_to_molecule')
    c=m.condition((v,))
    assert c.shape == (1,4)
    other=replace(v,target_bytes=b'{"canonical_isomeric_smiles":"CCN"}',provenance_bytes=b'{}')
    assert torch.equal(c,m.condition((other,)))
    changed=json.loads(v.input_bytes);changed['source_profile_array']['values'][0]=100
    assert not torch.equal(c,m.condition((replace(v,input_bytes=json.dumps(changed).encode()),)))
    c.square().sum().backward()
    assert m.profile_input['L1000'].weight.grad.abs().sum()>0
    assert m.backbone.feature.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='decoder'): m.generate((v,))
    with pytest.raises(ValueError,match='direction'): m((v,))


def test_axis_and_checkpoint_configuration_rejected():
    m=model();v=view()
    with pytest.raises(ValueError,match='axis'): m((replace(v,axis=v.axis[::-1]),))
    state=copy.deepcopy(m.get_extra_state());state['axes']['L1000']=('bad',)
    with pytest.raises(ValueError,match='configuration'): m.set_extra_state(state)
    z=m((v,))[0]
    with pytest.raises(ValueError):m.objective((z[:-1],),(v,))
    with pytest.raises(ValueError):m(())
