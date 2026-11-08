"""Fifteenth objective: exact source binding, independent rows, shared Adam."""
import copy
import hashlib
from dataclasses import replace
import pytest
import torch
from test_geometry_reconstruction import build as model_build
from test_reconstruction_schedule import build as trainer_build
from test_segmentation_inference import setup
from test_tox21_training import equal
from mora3.model import reconstruction_cooperative as module


def build(tmp_path, device='cpu', **changes):
    assert hasattr(module, 'GeometryReconstructionCooperativeTrainer'), 'geometry scheduler missing'
    base = trainer_build(tmp_path, device)
    model, packet = model_build(tmp_path, device)
    task, native, request = setup(tmp_path, device)
    other = replace(task, source_id='second.tif')
    request2 = replace(request, artifacts=(replace(request.artifacts[0], asset='archive#second.tif'),))
    options = dict(geometry_tasks=(task, other), image_requests=(request, request2),
        admitted_task_sha256s=tuple(hashlib.sha256(t.canonical_bytes()).hexdigest() for t in (task, other)),
        prediction_assets=('first.pt', 'second.pt'), native=native, geometry_batch_size=1)
    options.update(changes)
    return module.GeometryReconstructionCooperativeTrainer(model, base.batch, base.pools,
        reconstruction_batch=base.reconstruction_batch, reconstruction_batch_size=1,
        residue_batch=base.residue_batch, profile_batch=base.profile_batch, profile_batch_size=2,
        decoder_batch=base.decoder_batch, decoder_batch_size=2, tox21_batch=base.tox21_batch,
        tox21_batch_size=1, batch_size=1, flow_batch_size=1,
        nucleotide_batch=base.nucleotide_batch, nucleotide_batch_size=1,
        rfam_batch=base.rfam_batch, rfam_prediction_batch_size=1, rfam_generation_batch_size=2,
        goa_packets=(packet,), goa_batch_size=1, **options)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_geometry_scheduler_matches_direct_adam_independent_cursor(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    t = build(tmp_path, device)
    assert len(t.kinds) == 15 and t.kinds[-2:] == ('profile_reconstruction', 'geometry')
    ps = [p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps) == len(set(map(id, ps))) == len(tuple(t.model.parameters()))
    assert not t.optimizer.state and t.steps == 0  # explicit fresh Adam boundary
    archive = copy.deepcopy(t.reconstruction_batch.state_dict())
    for cycle in range(3):
        t.steps = cycle*15+14
        i = cycle % 2
        ref = copy.deepcopy(t.model)
        opt = torch.optim.Adam(ref.parameters(), lr=.001)
        opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
        opt.zero_grad()
        loss = ref.geometry_objective(t.geometry_tasks[i], request=t.image_requests[i],
            native=t.geometry_native, admitted_task_sha256=t.admitted_task_sha256s[i],
            prediction_asset=t.prediction_assets[i])
        loss.backward(); opt.step()
        row = t.step()
        assert row['kind'] == 'geometry' and row['examples'] == 1 and row['loss'] == loss.item()
        equal(t.model.state_dict(), ref.state_dict()); equal(t.optimizer.state_dict(), opt.state_dict())
        assert all(p.grad is None for p in t.geometry_native.model.parameters())
    equal(archive, t.reconstruction_batch.state_dict())
    # Prior reconstruction cursor uses its own source rows on the expanded cycle.
    t.steps = 2*15+13
    actual, count = t.additional_objective('profile_reconstruction')
    expected = t.model.reconstruction_objective(t.reconstruction_batch, indices=(0,))
    assert count == 1 and torch.equal(actual, expected)


@pytest.mark.parametrize('change', ['size', 'empty', 'length', 'digest', 'asset'])
def test_constructor_rejects_invalid_geometry_pool(tmp_path, change):
    options = dict(size={'geometry_batch_size':True}, empty={'geometry_tasks':()},
        length={'image_requests':()}, digest={'admitted_task_sha256s':('0'*64,)*2},
        asset={'prediction_assets':('same.pt',)*2})[change]
    with pytest.raises(ValueError): build(tmp_path, **options)


@pytest.mark.parametrize('change', ['task', 'request', 'asset', 'size', 'schedule', 'native', 'source_size', 'pool'])
def test_live_binding_rejected_before_any_update(tmp_path, change):
    t = build(tmp_path)
    before = copy.deepcopy(t.model.state_dict()); adam = copy.deepcopy(t.optimizer.state_dict())
    if change == 'task': t.geometry_tasks = (replace(t.geometry_tasks[0], split='test'),) + t.geometry_tasks[1:]
    if change == 'request': t.image_requests = t.image_requests[::-1]
    if change == 'asset': t.prediction_assets = t.prediction_assets[::-1]
    if change == 'size': t.geometry_batch_size = 2
    if change == 'schedule': t.kinds = t.kinds[::-1]
    if change == 'native': t.geometry_native.identity += '-changed'
    if change == 'source_size': t.decoder_batch_size += 1
    if change == 'pool': t.pools['image'] = ()
    # Rejection even on the first annotation slot, before mode/gradient mutation.
    with pytest.raises(ValueError): t.step()
    equal(before, t.model.state_dict()); equal(adam, t.optimizer.state_dict()); assert t.steps == 0
