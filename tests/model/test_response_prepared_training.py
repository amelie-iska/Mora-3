"""Prepared training must preserve full CFM gradients and reference resume."""
import copy
import pytest
import torch
from test_sciplex_shared import task
from test_response_flow_trainer import equal
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_flow_trainer import ResponseFlowBatch, ResponseFlowTrainer
from mora3.model.sciplex_preparation import PreparedSciPlexTask
from mora3.model.flow_matching import gaussian_path, velocity_loss


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('head', [ResponseFlowHead, ResponseFlowDenoiserV2])
def test_prepared_full_loss_gradients_repeated_backwards_and_target_exclusion(device, head):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    torch.manual_seed(26001)
    tasks = (task(), task(counts=(3, 7), cell='b'))
    batch = ResponseFlowBatch(tasks)
    model = head(SciPlexSharedModel(('A','B'), width=4)).to(device).double()
    ref = copy.deepcopy(model)
    trainer = ResponseFlowTrainer(model, batch, batch_size=2, prepare_tasks=True)
    prepared = trainer.training_tasks
    assert all(isinstance(t, PreparedSciPlexTask) for t in prepared)
    assert [t.canonical_bytes() for t in prepared] == [t.canonical_bytes() for t in tasks]
    # Returned target tensors cannot poison subsequent training preparations.
    prepared[0].project()['targets']['expression'].fill_(999)
    assert prepared[0].project()['targets']['expression'].tolist() == [1, 2]
    changed = PreparedSciPlexTask.from_admitted(task(counts=(90,80), cell='hidden'),
        axis_ids=('A','B'), device=device, dtype=torch.float64)
    for key, value in model.native.graph(prepared[0]).items():
        assert torch.equal(value, model.native.graph(changed)[key])
    state = torch.tensor([[.1,.2],[.3,.4]], device=device, dtype=torch.float64)
    times = torch.tensor([.2,.8], device=device, dtype=torch.float64)
    y = torch.stack([torch.log1p(t.project(device=device,dtype=torch.float64)['targets']['expression']) for t in tasks])
    x, velocity = gaussian_path(state, y, times, sigma_min=.01)
    for _ in range(2):
        model.zero_grad(set_to_none=True); ref.zero_grad(set_to_none=True)
        a = velocity_loss(model(prepared, x, times), velocity, torch.ones_like(y,dtype=torch.bool))
        b = ((ref(tasks, x, times)-velocity)**2).mean()
        assert torch.equal(a,b)
        a.backward(); b.backward()
        for p,q in zip(model.parameters(),ref.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None: assert torch.equal(p.grad,q.grad)
    assert trainer.steps == 0 and not trainer.optimizer.state


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_prepared_schedule_updates_and_reference_checkpoint_are_exact(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(26002)
    batch = ResponseFlowBatch((task(),task(counts=(3,7),cell='b')))
    model = ResponseFlowHead(SciPlexSharedModel(('A','B'),width=4)).to(device).double()
    a = ResponseFlowTrainer(model,batch,batch_size=1)
    b = ResponseFlowTrainer(copy.deepcopy(model),batch,batch_size=1,prepare_tasks=True)
    assert a.training_tasks is batch.tasks
    assert a.binding == b.binding
    for _ in range(2):
        state = a.state_dict()
        expected = a.step()
        b.load_state_dict(state)
        assert b.step() == expected
        equal(a.state_dict(),b.state_dict())
    # Prepared state is runtime-only; reverse restoration needs no migration.
    a.load_state_dict(b.state_dict())
    assert set(a.state_dict()) == {'version','binding','steps','model','optimizer','cpu_rng','cuda_rng'}


@pytest.mark.parametrize('bad', [None, 0, 1, 'yes'])
def test_preparation_policy_requires_explicit_bool(bad):
    model = ResponseFlowHead(SciPlexSharedModel(('A','B'),width=4))
    before = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state()
    with pytest.raises(ValueError, match='prepare_tasks'):
        ResponseFlowTrainer(model,ResponseFlowBatch((task(),)),batch_size=1,prepare_tasks=bad)
    equal(model.state_dict(),before)
    assert torch.equal(rng,torch.get_rng_state())
