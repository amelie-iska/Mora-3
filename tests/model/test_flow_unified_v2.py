import copy
import pytest
import torch
from test_flow_unified import build
from test_response_flow_trainer import equal
from mora3.model.flow_unified import FlowUnifiedModel, FlowUnifiedTrainer
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.flow_matching import gaussian_path, velocity_loss


def versioned(device='cpu', version=2, sigma=.01):
    base = build(device)
    model = FlowUnifiedModel(base.model.native, base.model.annotation,
                             flow_version=version, sigma_min=sigma).to(device)
    return FlowUnifiedTrainer(model, base.batch, base.pools, batch_size=1,
                              flow_batch_size=1, sigma_min=sigma)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_v2_cycle_equations_shared_gradients_and_exact_restore(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    torch.manual_seed(6301)
    tr = versioned(device)
    assert isinstance(tr.model.flow, ResponseFlowDenoiserV2)
    assert tr.model.flow.native is tr.model.native.response
    for kind in ('annotation', 'image', 'response', 'intervention'):
        m = tr.model
        expected = (m.annotation.objective(m.annotation(tr.batch.views), tr.batch.views,
            applicable=tr.batch.applicability(device))['total'] if kind == 'annotation'
            else m.native.objective(kind, tr.pools[kind][:1]))
        result = tr.step()
        assert result['kind'] == kind
        assert result['loss'] == pytest.approx(float(expected.detach()), abs=1e-7)
        assert m.flow.output.weight.grad is None
    saved = tr.state_dict()
    tasks = tr.pools['response'][:1]
    y = torch.log1p(tasks[0].project(device=device, dtype=torch.float32)['targets']['expression'])[None]
    noise = torch.randn_like(y)
    time = torch.rand(1, device=device)
    state, target = gaussian_path(noise, y, time, sigma_min=.01)
    expected = velocity_loss(tr.model.flow(tasks, state, time), target, torch.ones_like(y, dtype=torch.bool))
    tr.load_state_dict(saved)
    result = tr.step()
    assert result['loss'] == float(expected.detach())
    assert tr.model.flow.output.weight.grad.abs().sum() > 0
    assert tr.model.native.response.backbone.feature.weight.grad.abs().sum() > 0
    assert tr.model.native.image.head.weight.grad is None
    end = tr.state_dict()
    fresh = versioned(device)
    fresh.load_state_dict(saved)
    assert fresh.step() == result
    equal(fresh.state_dict(), end)


def test_v1_default_and_version_sigma_binding_atomicity():
    assert type(build().model.flow) is ResponseFlowHead
    original = versioned()
    saved = original.state_dict()
    for other in (versioned(version=1), versioned(sigma=.1)):
        before = copy.deepcopy(other.model.state_dict())
        with pytest.raises(ValueError, match='binding'):
            other.load_state_dict(saved)
        equal(before, other.model.state_dict())
    with pytest.raises(ValueError, match='sigma'):
        FlowUnifiedTrainer(original.model, original.batch, original.pools,
                           batch_size=1, flow_batch_size=1, sigma_min=.1)
    for version in (0, 3, True, '2'):
        base = build()
        with pytest.raises(ValueError, match='version'):
            FlowUnifiedModel(base.model.native, base.model.annotation, flow_version=version)
