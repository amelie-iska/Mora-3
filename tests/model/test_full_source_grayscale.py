"""Complete grayscale frame ingress and source-bound shared ownership."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from mora3.data.provenance import Provenance
from mora3.model.cooperative_training import GeometryCooperativeModel
from mora3.model.morphology import MorphologyInstance
from mora3.model.raster_source_envelope import RasterSourceEnvelope


def make_input(array, envelope):
    from mora3.data.full_source_raster_input import FullSourceRasterInput
    pixels = array.tobytes()
    return FullSourceRasterInput(
        *array.shape, array.dtype.str, pixels, hashlib.sha256(pixels).hexdigest(), envelope,
        Provenance('synthetic', 'v1', 'frame', 'a' * 64, 'unit-only'),
    )


def test_complete_1024_uint16_frame_preserves_native_pixels_and_source_binding():
    envelope = RasterSourceEnvelope('b' * 64, 1024, 1024, None)
    original = np.arange(1024 * 1024, dtype=np.uint32).astype('>u2').reshape(1024, 1024)
    frame = make_input(original, envelope)
    assert frame.pixels == original.tobytes()
    np.testing.assert_array_equal(frame.input_image().numpy(), original)
    graph = frame.input_graph()
    assert len(graph.nodes) == 1 and not graph.edges and not graph.nodes[0].measurements
    with pytest.raises(ValueError, match='pixels'):
        replace(frame, pixels=frame.pixels[:-2])
    with pytest.raises(ValueError, match='envelope'):
        replace(frame, width=1025)


def test_native_gray_uses_every_pixel_and_legacy_rejects_new_input_before_cnn():
    envelope = RasterSourceEnvelope('b' * 64, 1024, 1024, None)
    raw = (np.arange(63, dtype=np.uint16) * 1037).astype('>u2').reshape(7, 9)
    frame = make_input(raw, envelope)
    model = MorphologyInstance(8, parameterization_policy='mup-image-v1', base_width=8, source_envelope=envelope).double()
    inputs = []
    hook = model.pixels.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().clone()))
    try:
        output = model(frame)
    finally:
        hook.remove()
    assert output.shape == (3, 7, 9)
    torch.testing.assert_close(inputs[0][0, 0], torch.tensor(raw.astype('float64')) / 65535, rtol=0, atol=0)
    with pytest.raises(ValueError, match='envelope'):
        MorphologyInstance(8)(frame)
    with pytest.raises(ValueError, match='admission'):
        model(frame, for_training=True)
    with pytest.raises(ValueError, match='envelope'):
        model(replace(frame, source_envelope=RasterSourceEnvelope('c' * 64, 1024, 1024, None)))


def test_gray_cooperative_declaration_preserves_shared_backbone_and_strict_checkpoint():
    envelope = RasterSourceEnvelope('b' * 64, 1024, 1024, None)
    options = dict(width=8, id_width=2, parameterization_policy='mup-cooperative-v1', base_width=8)
    model = GeometryCooperativeModel(('gene',), grayscale_source_envelope=envelope, **options)
    assert model.image.source_envelope is envelope
    assert model.image.backbone is model.response.backbone
    assert model.image.get_extra_state()['mup']['source_envelope'] == envelope.binding()
    state = model.state_dict()
    same = GeometryCooperativeModel(('gene',), grayscale_source_envelope=envelope, **options)
    same.load_state_dict(state)
    default = GeometryCooperativeModel(('gene',), **options)
    with pytest.raises(ValueError):
        default.load_state_dict(state)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        GeometryCooperativeModel(('gene',), grayscale_source_envelope=envelope)
    assert torch.equal(rng, torch.get_rng_state())
