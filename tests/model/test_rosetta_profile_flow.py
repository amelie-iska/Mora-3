"""Exact source-array boundary, flow equations and native width roles."""
from dataclasses import replace
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mora3.data.array_tasks import ArrayTask, OrderedArray
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.model.mu_optimizer import MuAdamW
import test_finite_union_native_rows as native_fixture
from test_sciplex_mup import make as sciplex


def binding():
    from mora3.data.rosetta_unconditional_profiles import RosettaProfileAxis
    return (
        RosettaProfileAxis('CellPainting', 'studyA:CP', ('a', 'b', 'c'), 'source-native'),
        RosettaProfileAxis('CellPainting', 'studyB:CP', ('a', 'b', 'c'), 'source-native'),
        RosettaProfileAxis('L1000', 'studyA:L1000', ('1', '2'), 'source-native'),
        RosettaProfileAxis('L1000', 'studyB:L1000', ('2', '1'), 'source-native'),
    )


def fixture_view(tmp_path):
    fixture = native_fixture.NativeViewTests()
    fixture.root = tmp_path
    axes = binding()
    rows = []
    for axis, values in zip(axes[:2], ((1., 2., 3.), (4., 5., 6.))):
        arr = OrderedArray('profile', axis.axis_ids, values, axis.axis_identity, axis.units, 'observed')
        raw = ArrayTask((arr,), (('profile', 'target'),), 'a' * 64).canonical_bytes()
        rows.append(dict(task_id=axis.axis_identity, source_family='Rosetta-CellPainting', native_schema='mora3.ArrayTask/v1',
                         task_name='unconditional_source_profile_modeling', split='train',
                         source_sha256='a' * 64, native_payload=raw, full_axis_values=3,
                         native_payload_sha256=hashlib.sha256(raw).hexdigest(),
                         condition_context_json='{"intervention_identity":"excluded source audit"}'))
    path = tmp_path / 'original.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = path.read_bytes()
    descriptor = fixture.descriptor([0, 0])
    descriptor['source_shard'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    descriptor['source_full9'] = list(full9(path))
    descriptor['original_group_sidecar'] = fixture.put('groups.parquet', b'exact audit')
    path, sha = fixture.manifest([descriptor])
    view = FiniteUnionNativeView(path, sha, root=tmp_path)
    batch = next(view.iter_native_task_batches(split='train', batch_size=2))
    return view, batch


def model(width=8, view=None):
    from mora3.data.rosetta_unconditional_profiles import rosetta_view_policy_sha256
    from mora3.model.rosetta_profile_flow import RosettaProfileFlow
    return RosettaProfileFlow(
        sciplex(width).double().backbone, binding(), source_census_sha256='1' * 64,
        native_view_policy_sha256=rosetta_view_policy_sha256(view) if view is not None else '2' * 64,
        readout_mult=1.5, sigma_min=.05,
    )


def test_complete_authentic_source_arrays_and_masked_audit_are_separate(tmp_path):
    from mora3.data.rosetta_unconditional_profiles import rosetta_profile_targets
    view, batch = fixture_view(tmp_path)
    axes = binding()
    first, second = rosetta_profile_targets(view, batch, axes, split='train')
    assert first.axis == axes[0] and second.axis == axes[1]
    assert first.values == (1., 2., 3.) and second.values == (4., 5., 6.)
    assert not hasattr(first, 'condition') and not hasattr(first, 'condition_context_json')
    with pytest.raises(ValueError):
        rosetta_profile_targets(view, batch, axes[1:], split='train')
    with pytest.raises(ValueError):
        rosetta_profile_targets(view, batch, axes, split='test')
    with pytest.raises(ValueError):
        rosetta_profile_targets(view, replace(batch, source_row_ordinals=(1, 0)), axes, split='train')


def test_exact_flow_objective_mixed_study_reassembly_and_all_gradients(tmp_path):
    view, batch = fixture_view(tmp_path)
    m = model(12, view)
    noise = torch.tensor([[.1, -.2, .3], [-.4, .5, -.6]], dtype=torch.float64)
    times = noise.new_tensor([.2, .8])
    target = noise.new_tensor([[1., 2., 3.], [4., 5., 6.]])
    state = (1 - .95 * times[:, None]) * noise + times[:, None] * target
    expected_predictions = torch.cat([m(axis.sha256, x[None], t[None]) for axis, x, t in
                                      zip(binding()[:2], state, times)])
    result = m.objective(view, batch, split='train', noise=noise, time=times,
                         return_predictions=True)
    torch.testing.assert_close(result['predictions'], expected_predictions, rtol=0, atol=0)
    expected = (expected_predictions - (target - .95 * noise)).square().mean()
    torch.testing.assert_close(result['loss'], expected, rtol=1e-14, atol=1e-14)
    result['loss'].backward()
    for axis in binding()[:2]:
        assert torch.count_nonzero(m.state_projection[axis.sha256].weight.grad)
        assert torch.count_nonzero(m.readout[axis.sha256].weight.grad)
    assert torch.count_nonzero(m.time_projection.weight.grad)
    assert torch.count_nonzero(m.backbone.layers[0].attention.qkv.weight.grad)
    assert m.readout[binding()[2].sha256].weight.grad is None


def test_sampler_is_target_free_seeded_and_fixed_axis():
    m = model()
    axis = binding()[2]
    rng = torch.get_rng_state().clone()
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    left = m.sample(axis.sha256, 3, steps=3, generator=g1)
    right = m.sample(axis.sha256, 3, steps=3, generator=g2)
    assert left.shape == (3, 2) and torch.isfinite(left).all()
    assert torch.equal(left, right) and torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError):
        m.sample('f' * 64, 3, steps=3, generator=g1)
    with pytest.raises(ValueError):
        m.sample(axis.sha256, 3, steps=3, generator=None)


def test_fixed_axes_aliases_and_mup_optimizer_groups():
    from mora3.model.rosetta_profile_flow import rosetta_profile_flow_roles, rosetta_profile_flow_width_family
    members = tuple(model(w) for w in (8, 12, 16))
    family = rosetta_profile_flow_width_family(*members)
    roles = rosetta_profile_flow_roles(members[-1])
    axis, other = binding()[2:]
    input_name = f'state_projection.{axis.sha256}.weight'
    output_name = f'readout.{axis.sha256}.weight'
    assert roles[input_name].role == 'input' and roles[output_name].role == 'output'
    assert roles[input_name].axes[1] == roles[output_name].axes[0]
    assert roles[output_name].axes[0] != roles[f'readout.{other.sha256}.weight'].axes[0]
    opt = MuAdamW(members[-1], family, roles, base_lr=.02, weight_decay=0., decay_aliases=())
    by_alias = {name: group for group in opt.param_groups for name in group['mup_aliases']}
    assert by_alias[output_name]['lr'] == .02
    assert by_alias['backbone.layers.0.attention.qkv.weight']['lr'] == .01
    assert not opt.state
    altered = model(16)
    altered.native_view_policy_sha256 = 'f' * 64
    with pytest.raises(ValueError):
        rosetta_profile_flow_width_family(*members[:2], altered)
