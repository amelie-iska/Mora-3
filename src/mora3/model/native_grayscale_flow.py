"""Complete mask-to-image Gaussian flow in a fixed native detector coordinate."""

import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.deepbacs_native import DeepBacsNativeSource, NativeGrayMaskCondition
from mora3.data.schema import Edge, GraphRecord, Measurement, Node
from mora3.data.tasks import ConditionalTask
from .flow_matching import euler_flow, gaussian_path
from .geometry_conditioning import GeometryConditioner
from .mixed_payload_projection import exact_wire
from .morphology import MultiscalePixels
from .morphology_mup import _geometry_roles, _without_member_dimensions, pixel_mup_parameter_roles
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger, width_family
from .raster_source_envelope import RasterSourceEnvelope


class NativeGrayscaleFlow(nn.Module):
    """Known masks condition both local pixels and the existing shared geometry owner."""

    def __init__(self, geometry, source_envelope, *, detector_dtype,
                 readout_mult=1., sigma_min=.01, max_sample_steps=256):
        super().__init__()
        if (
            type(geometry) is not GeometryConditioner or type(source_envelope) is not RasterSourceEnvelope
            or type(detector_dtype) is not str or np.dtype(detector_dtype).kind != 'u'
            or np.dtype(detector_dtype).itemsize not in (1, 2)
            or type(readout_mult) not in (int, float) or not math.isfinite(readout_mult) or readout_mult <= 0
            or type(sigma_min) not in (int, float) or not 0 < sigma_min < 1
            or type(max_sample_steps) is not int or max_sample_steps < 1
        ):
            raise ValueError('Complete native grayscale geometry, detector and flow policy required')
        geometry_config = geometry.get_extra_state()
        if geometry_config['version'] != 2 or geometry_config['parameterization_policy'] != 'mup-geometry-v1':
            raise ValueError('Exact applied shared native geometry owner required')
        self.geometry = geometry
        self._geometry_reference = (geometry,)
        self.source_envelope = source_envelope
        self.detector_dtype = detector_dtype
        self.detector_max = int(np.iinfo(np.dtype(detector_dtype)).max)
        self.width = geometry_config['width']
        self.base_width = geometry_config['native_graph']['parameterization']['base_width']
        self.readout_mult = float(readout_mult)
        self.sigma_min = float(sigma_min)
        self.max_sample_steps = max_sample_steps
        self.rank = geometry_config['id_width']
        self.bit_count = (self.rank - 1).bit_length()
        self.channels = ('noisy-detector-state',) + tuple(f'known-mask-ordinal-bit:{i}' for i in range(self.bit_count))
        self.pixels = MultiscalePixels(self.width, input_channels=len(self.channels),
                                      parameterization_policy='mup-image-v1', base_width=self.base_width)
        self.time = nn.Linear(1, self.width)
        self.output = nn.Conv2d(self.width, 1, 1)
        with torch.no_grad():
            ratio = math.sqrt(self.width / self.base_width)
            self.output.weight.mul_(ratio)
            self.output.bias.mul_(ratio)
        reference = self.geometry.graph.feature.weight
        for module in (self.pixels, self.time, self.output):
            module.to(device=reference.device, dtype=reference.dtype)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeGrayscaleMaskToDetectorFlow/v1',
            geometry=self.geometry.get_extra_state(),
            source_envelope=self.source_envelope.binding(),
            width=self.width, base_width=self.base_width, rank=self.rank,
            detector_dtype=self.detector_dtype, detector_max=self.detector_max,
            input_channels=self.channels, partition_bits=self.bit_count,
            readout_mult=self.readout_mult, sigma_min=self.sigma_min, max_sample_steps=self.max_sample_steps,
            pixels=self.pixels.parameterization_configuration(),
            partition='complete known mask first-spatial-occurrence ordinal bit planes and all observed mask nodes',
            intensity='native unsigned detector coordinate divided by fixed source dtype maximum',
            objective='positive-sigma Eq21 Gaussian flow on every source detector pixel',
            generation='continuous native detector coordinate; fixed-step Euler, caller-owned generator',
        )

    def get_extra_state(self):
        if (
            set(self._modules) != {'geometry', 'pixels', 'time', 'output'}
            or self.geometry is not self._geometry_reference[0] or type(self.pixels) is not MultiscalePixels
            or type(self.time) is not nn.Linear or self.time._modules or self.time._buffers
            or set(self.time._parameters) != {'weight', 'bias'} or self.time.bias is None
            or (self.time.in_features, self.time.out_features) != (1, self.width)
            or tuple(self.time.weight.shape) != (self.width, 1) or tuple(self.time.bias.shape) != (self.width,)
            or type(self.output) is not nn.Conv2d or self.output._modules or self.output._buffers
            or set(self.output._parameters) != {'weight', 'bias'} or self.output.bias is None
            or (self.output.in_channels, self.output.out_channels, self.output.kernel_size, self.output.stride,
                self.output.padding, self.output.dilation, self.output.groups, self.output.padding_mode)
            != (self.width, 1, (1, 1), (1, 1), (0, 0), (1, 1), 1, 'zeros')
            or tuple(self.output.weight.shape) != (1, self.width, 1, 1) or tuple(self.output.bias.shape) != (1,)
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Native grayscale reverse geometry, source, or operator policy drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Native grayscale reverse source/configuration checkpoint mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native grayscale reverse source/configuration checkpoint mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _check_condition(self, condition):
        if type(condition) is not NativeGrayMaskCondition or condition.source_envelope != self.source_envelope:
            raise ValueError('Exact observed native mask condition required')
        condition.__post_init__()

    def _prepare(self, condition):
        labels = condition.partition()
        cells = int(labels.max())
        if cells + 1 > self.rank:
            raise ValueError('Complete observed mask geometry exceeds native identifier rank')
        record, task = _condition_graph(condition, labels, cells)
        context = self.geometry(record, task)[0]
        codes = torch.as_tensor(labels, device=context.device)
        shifts = torch.arange(self.bit_count, device=context.device)[:, None, None]
        bits = ((codes[None] >> shifts) & 1).to(context.dtype)
        return context, bits

    def _check_state(self, condition, state, time):
        reference = self.output.weight
        if (
            not isinstance(state, torch.Tensor) or state.shape != (condition.height, condition.width)
            or state.device != reference.device or state.dtype != reference.dtype or not torch.isfinite(state).all()
            or not isinstance(time, torch.Tensor) or time.shape != ()
            or time.device != state.device or time.dtype != state.dtype or not torch.isfinite(time)
            or not bool((0 <= time) & (time <= 1))
        ):
            raise ValueError('Exact complete finite noisy detector frame and scalar time in [0,1] required')

    def _velocity(self, context, bits, state, time):
        local = self.pixels(torch.cat((state[None], bits))[None])
        offset = context + self.time(time.reshape(1, 1))[0]
        hidden = F.silu(local + offset[None, :, None, None])
        denoised = self.output(hidden * (self.readout_mult / (self.width / self.base_width)))[0, 0]
        rate = 1 - self.sigma_min
        return (denoised - rate * state) / (1 - rate * time)

    def forward(self, condition, state, time):
        self.get_extra_state()
        self._check_condition(condition)
        self._check_state(condition, state, time)
        return self._velocity(*self._prepare(condition), state, time)

    def objective(self, source, batch, *, split, noise, time, return_prediction=False):
        self.get_extra_state()
        if (
            type(source) is not DeepBacsNativeSource or source.envelope != self.source_envelope
            or type(return_prediction) is not bool
        ):
            raise ValueError('Exact source grayscale reverse envelope required')
        if split == 'test' and torch.is_grad_enabled():
            raise ValueError('Development-exposed original test is evaluation only')
        packets = source.packets(batch, split=split, direction='mask_to_image')
        if len(packets) != 1 or packets[0].image.image_dtype != self.detector_dtype:
            raise ValueError('One complete source reverse packet in the declared detector dtype required')
        packet, = packets
        condition = packet.mask_condition()
        self._check_condition(condition)
        self._check_state(condition, noise, time)
        target = packet.image.input_image(device=noise.device).to(noise.dtype) / self.detector_max
        state, velocity = gaussian_path(
            noise.reshape(1, -1), target.reshape(1, -1), time.reshape(1), sigma_min=self.sigma_min)
        predicted = self(condition, state.reshape_as(noise), time)
        loss = (predicted - velocity.reshape_as(noise).detach()).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Finite full-pixel native grayscale flow loss required')
        return dict(loss=loss, velocity=predicted) if return_prediction else loss

    @torch.no_grad()
    def sample(self, condition, *, steps, generator):
        self.get_extra_state()
        self._check_condition(condition)
        reference = self.output.weight
        if (
            type(steps) is not int or not 1 <= steps <= self.max_sample_steps
            or not isinstance(generator, torch.Generator) or generator.device != reference.device
            or generator is torch.default_generator
            or (reference.is_cuda and generator is torch.cuda.default_generators[reference.device.index])
        ):
            raise ValueError('Bounded Euler steps and private device-local generator required')
        context, bits = self._prepare(condition)
        shape = (condition.height, condition.width)
        initial = torch.randn((1, shape[0] * shape[1]), device=reference.device,
                              dtype=reference.dtype, generator=generator)
        result = euler_flow(
            lambda state, time: self._velocity(context, bits, state.reshape(shape), time[0]).reshape(1, -1),
            initial, steps=steps,
        )
        return result.reshape(shape) * self.detector_max


def _condition_graph(condition, labels, cells):
    height, width = labels.shape
    indices = np.arange(height * width, dtype=np.int64)
    yy, xx = indices // width, indices % width
    flat = labels.ravel()
    area = np.bincount(flat, minlength=cells + 1)
    xmin, ymin = np.full(cells + 1, width), np.full(cells + 1, height)
    xmax, ymax = np.zeros(cells + 1, dtype=np.int64), np.zeros(cells + 1, dtype=np.int64)
    np.minimum.at(xmin, flat, xx)
    np.minimum.at(ymin, flat, yy)
    np.maximum.at(xmax, flat, xx)
    np.maximum.at(ymax, flat, yy)
    nodes = [Node('microscopy_mask_frame', (
        Measurement('height', height, 'pixel', 'observed', True),
        Measurement('width', width, 'pixel', 'observed', True),
        Measurement('requested_detector_frame', None, None, 'missing', False),
    ))]
    edges = []
    for i in range(1, cells + 1):
        nodes.append(Node('observed_whole_cell', tuple(
            Measurement(name, int(value[i]), unit, 'observed', True)
            for name, value, unit in (
                ('area', area, 'pixel^2'), ('x_min', xmin, 'pixel'), ('x_max', xmax, 'pixel'),
                ('y_min', ymin, 'pixel'), ('y_max', ymax, 'pixel'),
            )
        )))
        edges.append(Edge(0, i, 'contains_observed_cell', True))
    record = GraphRecord('known-mask-condition', tuple(nodes), tuple(edges), (condition.provenance,))
    roles = tuple(
        (('node', i, field.name), 'condition' if field.available else 'target')
        for i, node in enumerate(nodes) for field in node.measurements
    )
    return record, ConditionalTask(roles, topology_known=True)


def _finite(namespace, values):
    raw = json.dumps([namespace, values], separators=(',', ':')).encode()
    return Axis('finite', hashlib.sha256(raw).hexdigest())


def native_grayscale_flow_roles(model, *, geometry_identifier_axis):
    if type(model) is not NativeGrayscaleFlow:
        raise ValueError('Exact shared native grayscale mask-to-image owner required')
    model.get_extra_state()
    roles = {f'geometry.{name}': role for name, role in
             _geometry_roles(model.geometry, identifier_axis=geometry_identifier_axis).items()}
    roles.update({f'pixels.{name}': role for name, role in
                  pixel_mup_parameter_roles(model.pixels, channel_axis=model.channels).items()})
    hidden = Axis('width', 'hidden')
    time = _finite('native-grayscale-flow-time-v1', ('t',))
    detector = _finite('native-grayscale-detector-coordinate-v1', (model.detector_dtype,))
    kernel = _finite('native-grayscale-pointwise-kernel-spatial-v1', (0,))
    roles['time.weight'] = ParameterRole('input', (hidden, time))
    roles['time.bias'] = ParameterRole('vector', (hidden,))
    roles['output.weight'] = ParameterRole('output', (detector, hidden, kernel, kernel))
    roles['output.bias'] = ParameterRole('vector', (detector,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def native_grayscale_flow_width_family(base, delta, target, *, geometry_identifier_axis):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, native_grayscale_flow_roles(
        model, geometry_identifier_axis=geometry_identifier_axis)) for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if base.width != base.base_width or any(
        not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models
    ):
        raise ValueError('Native grayscale reverse source family binding drift')
    return width_family(*ledgers)
