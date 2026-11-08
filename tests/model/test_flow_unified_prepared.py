"""Catch prepared flow changing cooperative equations, ownership or resume."""
import copy
import pytest
import torch
from test_flow_unified_v2 import versioned
from test_sciplex_shared import task
from test_response_flow_trainer import equal
from mora3.model.flow_unified import FlowUnifiedTrainer
from mora3.model.sciplex_preparation import PreparedSciPlexTask


@pytest.fixture(autouse=True)
def exact_runtime():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('version', [1, 2])
def test_prepared_cooperative_full_cycle_gradients_and_bidirectional_resume(device, version):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    torch.manual_seed(26101)
    a = versioned(device, version=version)
    b = FlowUnifiedTrainer(copy.deepcopy(a.model), a.batch, a.pools,
        batch_size=1, flow_batch_size=1, prepare_flow_tasks=True)
    assert a.binding == b.binding
    assert a.flow_training_tasks is a.flow_batch.tasks
    assert all(isinstance(t, PreparedSciPlexTask) for t in b.flow_training_tasks)
    assert all(not isinstance(t, PreparedSciPlexTask) for t in b.pools['response'])
    assert [t.canonical_bytes() for t in b.flow_training_tasks] == [t.canonical_bytes() for t in a.flow_batch.tasks]
    assert b.model.flow.native is b.model.native.response
    owned = [p for group in b.optimizer.param_groups for p in group['params']]
    assert len(owned) == len({id(p) for p in owned}) == len(list(b.model.parameters()))
    # Hidden targets/cell IDs do not enter the learned shared graph context.
    changed = PreparedSciPlexTask.from_admitted(task(counts=(90,80),cell='hidden'),
        axis_ids=('A','B'), device=device, dtype=next(b.model.parameters()).dtype)
    for key, value in b.model.native.response.graph(b.flow_training_tasks[0]).items():
        assert torch.equal(value, b.model.native.response.graph(changed)[key])
    b.flow_training_tasks[0].project()['targets']['expression'].fill_(999)
    assert b.flow_training_tasks[0].project()['targets']['expression'].tolist() == [1,2]
    # Two complete cycles exercise independent pool cursors and fresh backward graphs.
    for index, kind in enumerate(('annotation','image','response','intervention','flow') * 2):
        state = a.state_dict()
        expected = a.step()
        b.load_state_dict(state)
        actual = b.step()
        assert actual == expected and actual['kind'] == kind
        if kind == 'flow':
            assert actual['indices'] == (index // 5,)
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                assert torch.equal(p.grad, q.grad)
        equal(a.state_dict(), b.state_dict())
    saved = b.state_dict()
    expected = b.step()
    a.load_state_dict(saved)
    assert a.step() == expected
    equal(a.state_dict(), b.state_dict())


@pytest.mark.parametrize('bad', [None, 0, 1, 'yes'])
def test_prepared_cooperative_requires_bool_without_mutation(bad):
    base = versioned()
    before = base.state_dict()
    with pytest.raises(ValueError, match='prepare_flow_tasks'):
        FlowUnifiedTrainer(base.model, base.batch, base.pools,
            batch_size=1, flow_batch_size=1, prepare_flow_tasks=bad)
    equal(before, base.state_dict())


def test_prepared_cooperative_does_not_weaken_source_or_path_binding():
    base = versioned()
    prepared = FlowUnifiedTrainer(copy.deepcopy(base.model), base.batch, base.pools,
        batch_size=1, flow_batch_size=1, prepare_flow_tasks=True)
    pools = dict(base.pools, response=(task(counts=(99,88)),))
    other = FlowUnifiedTrainer(copy.deepcopy(base.model), base.batch, pools,
        batch_size=1, flow_batch_size=1, prepare_flow_tasks=True)
    before = other.state_dict()
    with pytest.raises(ValueError, match='binding'):
        other.load_state_dict(prepared.state_dict())
    equal(before, other.state_dict())
