import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from dataclasses import replace
import pytest
import torch
from test_flow_unified import build as base_build
from test_residue_cooperative import view
from test_response_flow_trainer import equal
from mora3.model.residue_cooperative import ResidueCooperativeModel
from mora3.model.residue_trainer import ResidueTrainingBatch
from mora3.model import residue_cooperative as module


@pytest.fixture(autouse=True)
def deterministic():
    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prior)


def build(device='cpu', *, applicable=(True,), axis=('Region',), lr=.001):
    base = base_build(device)
    model = ResidueCooperativeModel(base.model.native, base.model.annotation, axis).to(device)
    batch = ResidueTrainingBatch((view(),), axis, applicable)
    return module.ResidueCooperativeTrainer(model, base.batch, base.pools,
        residue_batch=batch, batch_size=1, flow_batch_size=1, lr=lr)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cycle_reference_residue_gradients_and_exact_continuation(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(7401)
    trainer = build(device)
    assert trainer.kinds == ('annotation','image','response','intervention','flow','residue')
    params = [p for g in trainer.optimizer.param_groups for p in g['params']]
    assert len(params) == len({id(p) for p in trainer.model.parameters()}) == len({id(p) for p in params})
    for kind in trainer.kinds[:-1]: assert trainer.step()['kind'] == kind
    saved = trainer.state_dict()
    reference = copy.deepcopy(trainer.model)
    expected = reference.residue_objective(trainer.residue_batch.views,
        applicable=trainer.residue_batch.applicability(device))
    result = trainer.step()
    assert result['kind'] == 'residue' and result['examples'] == 1
    assert result['loss'] == float(expected.detach())
    assert trainer.model.residue.head.output.weight.grad.abs().sum() > 0
    assert trainer.model.native.image.head.weight.grad is None
    assert trainer.model.flow.output.weight.grad is None
    end = trainer.state_dict()
    fresh = build(device); fresh.load_state_dict(saved)
    assert fresh.step() == result
    equal(end, fresh.state_dict())
    # Continuation across the boundary keeps native pool cursors and flow RNG exact.
    tail = [trainer.step() for _ in range(12)]
    fresh.load_state_dict(end)
    assert [fresh.step() for _ in range(12)] == tail
    equal(trainer.state_dict(), fresh.state_dict())
    equal(saved['steps'], 5)


@pytest.mark.parametrize('changes', [dict(applicable=(False,)),dict(axis=('Region','Other')),dict(lr=.002)])
def test_binding_mismatch_rejected_before_mutation(changes):
    trainer = build(); saved = trainer.state_dict()
    other = build(**changes); before = other.state_dict()
    with pytest.raises(ValueError, match='binding'): other.load_state_dict(saved)
    equal(before, other.state_dict())


def test_residue_batch_axis_and_native_model_required():
    base = base_build()
    batch = ResidueTrainingBatch((view(),), ('Region','Other'), (True,))
    model = ResidueCooperativeModel(base.model.native, base.model.annotation, ('Region',))
    with pytest.raises(ValueError, match='axis'):
        module.ResidueCooperativeTrainer(model,base.batch,base.pools,residue_batch=batch,
            batch_size=1,flow_batch_size=1)
    with pytest.raises(ValueError, match='residue'):
        module.ResidueCooperativeTrainer(base.model,base.batch,base.pools,residue_batch=batch,
            batch_size=1,flow_batch_size=1)


def test_residue_target_and_source_order_bound_to_checkpoint():
    trainer = build(); saved = trainer.state_dict()
    original = view()
    changed = replace(original, target_bytes=original.target_bytes.replace(
        b'"value":2', b'"value":1'))
    for views in ((changed,), (original, original)):
        batch = ResidueTrainingBatch(views, ('Region',), (True,)*len(views))
        other = module.ResidueCooperativeTrainer(copy.deepcopy(trainer.model),
            trainer.batch, trainer.pools, residue_batch=batch,
            batch_size=1, flow_batch_size=1)
        before = other.state_dict()
        with pytest.raises(ValueError, match='binding'): other.load_state_dict(saved)
        equal(before, other.state_dict())
