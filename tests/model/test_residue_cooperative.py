"""Native reference equations must survive sharing with all cooperative heads."""
import copy
from dataclasses import replace
import pytest
import torch
from test_flow_unified import build
from test_annotation_vocabulary import admitted
from test_response_flow_trainer import equal
from mora3.model import residue_cooperative as module
from mora3.model.residue_objective import SharedResidueModel


def view():
    return replace(admitted(), target_bytes=b'{"annotation_packet":{"features":[{"type":"Region","location":{"start":{"value":1,"modifier":"EXACT"},"end":{"value":2,"modifier":"EXACT"}}}]}}')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_shared_residue_equations_every_gradient_and_inactive_heads(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(7301)
    base = build(device).model
    reference = SharedResidueModel(copy.deepcopy(base.annotation), ('Region',))
    model = module.ResidueCooperativeModel(base.native, base.annotation, ('Region',))
    model.residue.load_state_dict(reference.state_dict())
    assert model.residue.shared is model.annotation
    assert model.residue.shared.backbone is model.native.image.backbone
    assert model.residue.shared.backbone is model.flow.native.backbone
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})
    v = view(); mask = torch.tensor([True], device=device)
    loss = model.residue_objective((v,), applicable=mask)
    expected = reference.objective(reference((v,)), (v,), applicable=mask)['total']
    assert torch.equal(loss, expected)
    loss.backward(); expected.backward()
    for (name, p), (other, q) in zip(model.residue.named_parameters(), reference.named_parameters()):
        assert name == other
        if q.grad is None: assert p.grad is None
        else: assert torch.equal(p.grad, q.grad), name
    assert model.native.response.backbone.feature.weight.grad.abs().sum() > 0
    assert model.annotation.payload.embedding.weight.grad.abs().sum() > 0
    assert model.residue.head.output.weight.grad.abs().sum() > 0
    assert model.native.image.head.weight.grad is None
    assert model.flow.output.weight.grad is None
    assert model.annotation.physical.unit.weight.grad is None
    assert torch.equal(model.residue((v,))[0], model.residue((admitted(),))[0])
    assert model.residue_objective((v,), applicable=~mask).item() == 0


@pytest.mark.parametrize('version', [1, 2])
def test_configuration_roundtrip_and_axis_flow_version_rejection(version):
    base = build().model
    model = module.ResidueCooperativeModel(base.native, base.annotation, ('Region',), flow_version=version)
    state = copy.deepcopy(model.state_dict())
    clone = copy.deepcopy(model); clone.load_state_dict(state)
    equal(state, clone.state_dict())
    config = model.get_extra_state()
    config['residue']['head']['feature_axis'] = ('Other',)
    with pytest.raises(ValueError, match='configuration'): model.set_extra_state(config)
    assert model.get_extra_state() != config
    assert model.get_extra_state()['flow']['version'] == version
