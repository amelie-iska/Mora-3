"""Complete mask-only generation geometry and μP flow roles."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mora3.data.provenance import Provenance
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.raster_source_envelope import RasterSourceEnvelope
from test_sciplex_mup import make as sciplex


def condition(labels=None):
    from mora3.data.deepbacs_native import NativeGrayMaskCondition
    labels = np.asarray(
        [[0, 2, 2, 0], [7, 0, 9, 2], [7, 0, 9, 0]] if labels is None else labels,
        dtype='>u2',
    )
    return NativeGrayMaskCondition(
        labels.shape[0], labels.shape[1], '>u2', labels.tobytes(),
        RasterSourceEnvelope('a' * 64, 4, 5, None),
        Provenance('source', 'v1', 'observed-mask', 'b' * 64, 'CC-BY-4.0'),
    )


def make(width=8, source_envelope=None):
    from mora3.model.native_grayscale_flow import NativeGrayscaleFlow
    backbone = sciplex(width).backbone
    geometry = GeometryConditioner(backbone, id_width=4,
                                   parameterization_policy='mup-geometry-v1', base_width=8)
    return NativeGrayscaleFlow(geometry, source_envelope or condition().source_envelope, detector_dtype='>u2').double()


def test_complete_mask_partition_preserves_disconnected_membership_and_ignores_native_ids():
    original = condition()
    renamed = condition([[0, 81, 81, 0], [42, 0, 3, 81], [42, 0, 3, 0]])
    a, b = original.partition(), renamed.partition()
    assert np.array_equal(a, b)
    assert a.tolist() == [[0, 1, 1, 0], [2, 0, 3, 1], [2, 0, 3, 0]]
    assert not hasattr(original, 'pixels') and not hasattr(original, 'image_dtype')
    with pytest.raises(ValueError):
        replace(original, label_bytes=original.label_bytes[:-1])


def test_exact_forward_flow_and_target_free_generation():
    m = make(12)
    observed = condition()
    renamed = condition([[0, 81, 81, 0], [42, 0, 3, 81], [42, 0, 3, 0]])
    state = torch.linspace(-1., 1., 12, dtype=torch.float64).reshape(3, 4)
    t = state.new_tensor(.3)
    seen = []
    hook = m.geometry.register_forward_pre_hook(
        lambda _module, args: seen.append((len(args[0].nodes), len(args[0].edges))))
    a, b = m(observed, state, t), m(renamed, state, t)
    hook.remove()
    assert torch.equal(a, b) and a.shape == (3, 4)
    assert seen == [(4, 3), (4, 3)]
    a.square().mean().backward()
    assert torch.count_nonzero(m.output.weight.grad)
    assert torch.count_nonzero(m.pixels.enc0[0].weight.grad)
    assert torch.count_nonzero(m.geometry.graph.incidence.weight.grad)
    rng = torch.get_rng_state().clone()
    samples = [m.sample(observed, steps=3, generator=torch.Generator().manual_seed(74)) for _ in range(2)]
    assert torch.equal(samples[0], samples[1]) and torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(samples[0]).all() and samples[0].shape == (3, 4)


def test_width_roles_use_exact_geometry_and_fixed_mask_bit_input_axis():
    from mora3.model.native_grayscale_flow import native_grayscale_flow_roles, native_grayscale_flow_width_family
    members = tuple(make(width) for width in (8, 12, 16))
    geometry_axis = ('frame', 'observed-ordinal1', 'observed-ordinal2', 'observed-ordinal3')
    roles = native_grayscale_flow_roles(members[-1], geometry_identifier_axis=geometry_axis)
    first = next(role for name, role in roles.items() if name.startswith('pixels.enc0') and name.endswith('.weight'))
    assert first.role == 'input' and [axis.kind for axis in first.axes[:2]] == ['width', 'finite']
    assert roles['output.weight'].role == 'output'
    assert [axis.kind for axis in roles['output.weight'].axes[:2]] == ['finite', 'width']
    family = native_grayscale_flow_width_family(*members, geometry_identifier_axis=geometry_axis)
    assert family.total_numel == tuple(sum(p.numel() for p in member.parameters()) for member in members)
    with pytest.raises(ValueError):
        members[-1](condition([[1, 2, 3, 4], [0, 0, 0, 0]]), torch.zeros(2, 4).double(),
                    torch.tensor(.2).double())


def test_public_source_reverse_objective_preserves_uint16_pixels_and_masks(tmp_path):
    from test_deepbacs_native_objective import fixture_source
    source, batches, originals = fixture_source(tmp_path, direction='mask_to_image', detector_dtype='>u2')
    m = make(8, source.envelope)
    packet, = source.packets(batches[0], split='train', direction='mask_to_image')
    known = packet.mask_condition()
    assert known.provenance.sha256 == originals[0].mask_sha256
    assert not hasattr(known, 'pixels')
    noise = torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(4, 5)
    time = noise.new_tensor(.6)
    clean = originals[0].input_image().double() / 65535
    state = (1 - .99 * time) * noise + time * clean
    expected_velocity = clean - .99 * noise
    direct = m(known, state, time)
    out = m.objective(source, batches[0], split='train', noise=noise, time=time, return_prediction=True)
    torch.testing.assert_close(out['velocity'], direct, rtol=0, atol=0)
    torch.testing.assert_close(out['loss'], (direct - expected_velocity).square().mean(), rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match='evaluation'):
        m.objective(source, batches[1], split='test', noise=noise, time=time)
