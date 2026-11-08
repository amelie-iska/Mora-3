"""Source-unconditional Rosetta flow with exact study-assay axes and shared TokenGT."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.rosetta_unconditional_profiles import (
    RosettaProfileAxis, _is_sha256, _json_sha256, rosetta_profile_targets, rosetta_view_policy_sha256,
)
from .flow_matching import euler_flow, gaussian_path
from .mixed_payload_projection import exact_wire
from .morphology_mup import _native_roles, _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger, width_family
from .tokengt import TokenGT


class RosettaProfileFlow(nn.Module):
    """One declared axis selects a marginal. Only noisy state and time enter the graph."""

    def __init__(self, backbone, axes, *, source_census_sha256, native_view_policy_sha256,
                 readout_mult=1.0, sigma_min=.01, max_batch_size=256, max_sample_steps=256,
                 backbone_identifier_axis=('context', 'observation')):
        super().__init__()
        if type(backbone) is not TokenGT or type(axes) is not tuple or not axes or any(
            type(axis) is not RosettaProfileAxis for axis in axes
        ):
            raise ValueError('Exact shared TokenGT and declared Rosetta axes required')
        native = backbone.configuration()
        if (
            native['schema'] != 'tokengt-native-configuration-v3' or native['attention'] != 'vanilla'
            or native['feature_width'] != native['width']
            or len({axis.sha256 for axis in axes}) != len(axes)
            or not _is_sha256(source_census_sha256) or not _is_sha256(native_view_policy_sha256)
            or type(readout_mult) not in (int, float) or not math.isfinite(readout_mult) or readout_mult <= 0
            or type(sigma_min) not in (int, float) or not 0 < sigma_min < 1
            or any(type(n) is not int or n < 1 for n in (max_batch_size, max_sample_steps))
            or type(backbone_identifier_axis) is not tuple
            or len(backbone_identifier_axis) != native['id_width']
            or any(type(label) is not str or not label for label in backbone_identifier_axis)
            or len(set(backbone_identifier_axis)) != len(backbone_identifier_axis)
        ):
            raise ValueError('Exact width family, source axis binding and bounded positive-sigma flow required')
        self.backbone = backbone
        self._shared_reference = (backbone,)
        self.axes = tuple(sorted(axes, key=lambda axis: axis.sha256))
        self._axes_by_sha = {axis.sha256: axis for axis in self.axes}
        self.source_census_sha256 = source_census_sha256
        self.native_view_policy_sha256 = native_view_policy_sha256
        self.readout_mult = float(readout_mult)
        self.sigma_min = float(sigma_min)
        self.max_batch_size = max_batch_size
        self.max_sample_steps = max_sample_steps
        self.backbone_identifier_axis = backbone_identifier_axis
        self.width = native['width']
        self.base_width = native['parameterization']['base_width']
        self.state_projection = nn.ModuleDict({
            key: nn.Linear(len(axis.axis_ids), self.width) for key, axis in self._axes_by_sha.items()
        })
        self.time_projection = nn.Linear(1, self.width)
        self.readout = nn.ModuleDict({
            key: nn.Linear(self.width, len(axis.axis_ids)) for key, axis in self._axes_by_sha.items()
        })
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            for output in self.readout.values():
                output.weight.mul_(scale)
                output.bias.mul_(scale)
        for module in (self.state_projection, self.time_projection, self.readout):
            module.to(device=backbone.feature.weight.device, dtype=backbone.feature.weight.dtype)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.RosettaProfileFlow/v1',
            source_census_sha256=self.source_census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256,
            axes=tuple(axis.binding() for axis in self.axes),
            backbone=self.backbone.configuration(),
            backbone_identifier_axis=self.backbone_identifier_axis,
            width=self.width, base_width=self.base_width, readout_mult=self.readout_mult,
            sigma_min=self.sigma_min, max_batch_size=self.max_batch_size,
            max_sample_steps=self.max_sample_steps,
            conditions='fixed source axis only; no intervention, molecule, record or clean profile context',
            path='complete native source units, positive-sigma Gaussian flow',
            objective='Eq21 denoiser velocity; equal weight per record, mean over all native coordinates',
            sampler='bounded fixed-step Euler, caller-owned generator',
        )

    def get_extra_state(self):
        if (
            self.backbone is not self._shared_reference[0] or type(self.backbone) is not TokenGT
            or set(self._modules) != {'backbone', 'state_projection', 'time_projection', 'readout'}
            or self._axes_by_sha != {axis.sha256: axis for axis in self.axes}
            or type(self.state_projection) is not nn.ModuleDict or type(self.readout) is not nn.ModuleDict
            or set(self.state_projection) != set(self._axes_by_sha) or set(self.readout) != set(self._axes_by_sha)
            or not _linear(self.time_projection, 1, self.width)
            or any(not _linear(self.state_projection[key], len(axis.axis_ids), self.width)
                   or not _linear(self.readout[key], self.width, len(axis.axis_ids))
                   for key, axis in self._axes_by_sha.items())
            or not exact_wire(self._configuration, self._current_configuration())
        ):
            raise ValueError('Rosetta source axis or flow configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Rosetta profile flow checkpoint configuration mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Rosetta profile flow checkpoint configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _axis(self, axis_sha256):
        if type(axis_sha256) is not str or axis_sha256 not in self._axes_by_sha:
            raise ValueError('Exact bound Rosetta profile axis required')
        return self._axes_by_sha[axis_sha256]

    def forward(self, axis_sha256, state, time):
        self.get_extra_state()
        axis = self._axis(axis_sha256)
        reference = self.backbone.feature.weight
        if (
            not isinstance(state, torch.Tensor) or state.ndim != 2
            or not 1 <= state.shape[0] <= self.max_batch_size or state.shape[1] != len(axis.axis_ids)
            or state.dtype != reference.dtype or state.device != reference.device
            or not isinstance(time, torch.Tensor) or time.shape != (state.shape[0],)
            or time.dtype != state.dtype or time.device != state.device
            or not torch.isfinite(state).all() or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError('Finite complete source-axis noisy state and matching flow time required')
        features = self.state_projection[axis_sha256](state) + self.time_projection(time[:, None])
        role = torch.zeros(1, dtype=torch.long, device=state.device)
        endpoints = role.expand(1, 2)
        identifiers = torch.eye(1, len(self.backbone_identifier_axis), dtype=state.dtype, device=state.device)
        graphs = [dict(features=x[None], endpoints=endpoints, roles=role, identifiers=identifiers) for x in features]
        hidden = F.silu(self.backbone(graphs)[0][:, 0])
        denoised = self.readout[axis_sha256](hidden * (self.readout_mult / (self.width / self.base_width)))
        rate = 1 - self.sigma_min
        return (denoised - rate * state) / (1 - rate * time[:, None])

    def objective(self, view, batch, *, split, noise, time, return_predictions=False):
        self.get_extra_state()
        if type(return_predictions) is not bool or rosetta_view_policy_sha256(view) != self.native_view_policy_sha256:
            raise ValueError('Exact bound current native Rosetta view and explicit reporting policy required')
        packets = rosetta_profile_targets(view, batch, self.axes, split=split)
        dimensions = {len(packet.values) for packet in packets}
        reference = self.backbone.feature.weight
        if (
            not 1 <= len(packets) <= self.max_batch_size or len(dimensions) != 1
            or not isinstance(noise, torch.Tensor) or noise.shape != (len(packets), next(iter(dimensions)))
            or noise.dtype != reference.dtype or noise.device != reference.device
        ):
            raise ValueError('Bounded native Rosetta batch and caller-supplied complete matching noise required')
        target = reference.new_tensor([packet.values for packet in packets])
        state, velocity = gaussian_path(noise, target, time, sigma_min=self.sigma_min)
        by_axis = {}
        for i, packet in enumerate(packets):
            by_axis.setdefault(packet.axis.sha256, []).append(i)
        predictions = [None] * len(packets)
        for axis_sha, indices in by_axis.items():
            values = self(axis_sha, state[indices], time[indices])
            for i, prediction in zip(indices, values):
                predictions[i] = prediction
        prediction = torch.stack(predictions)
        if not torch.isfinite(prediction).all():
            raise ValueError('Finite native Rosetta profile velocity required')
        loss = (prediction - velocity.detach()).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Native Rosetta loss overflow in model precision')
        return dict(loss=loss, predictions=prediction) if return_predictions else loss

    @torch.no_grad()
    def sample(self, axis_sha256, count, *, steps, generator):
        self.get_extra_state()
        axis = self._axis(axis_sha256)
        reference = self.backbone.feature.weight
        if (
            type(count) is not int or not 1 <= count <= self.max_batch_size
            or type(steps) is not int or not 1 <= steps <= self.max_sample_steps
            or not isinstance(generator, torch.Generator) or generator.device != reference.device
            or generator is torch.default_generator
            or (reference.is_cuda and generator is torch.cuda.default_generators[reference.device.index])
        ):
            raise ValueError('Bounded sample count, steps and private device-local generator required')
        noise = torch.randn(count, len(axis.axis_ids), generator=generator,
                            dtype=reference.dtype, device=reference.device)
        return euler_flow(lambda state, time: self(axis_sha256, state, time), noise, steps=steps)


def _linear(layer, in_features, out_features):
    return (
        type(layer) is nn.Linear and not layer._modules and not layer._buffers
        and set(layer._parameters) == {'weight', 'bias'}
        and (layer.in_features, layer.out_features) == (in_features, out_features)
        and tuple(layer.weight.shape) == (out_features, in_features)
        and layer.bias is not None and tuple(layer.bias.shape) == (out_features,)
    )


def rosetta_profile_flow_roles(model):
    if type(model) is not RosettaProfileFlow:
        raise ValueError('Exact shared Rosetta profile flow required')
    model.get_extra_state()
    roles = {f'backbone.{name}': role for name, role in
             _native_roles(model.backbone, model.backbone_identifier_axis).items()}
    hidden = Axis('width', 'hidden')
    time = Axis('finite', _json_sha256(['Rosetta-profile-Gaussian-path-time-v1', ['time']]))
    roles['time_projection.weight'] = ParameterRole('input', (hidden, time))
    roles['time_projection.bias'] = ParameterRole('vector', (hidden,))
    for axis in model.axes:
        finite = Axis('finite', axis.sha256)
        roles[f'state_projection.{axis.sha256}.weight'] = ParameterRole('input', (hidden, finite))
        roles[f'state_projection.{axis.sha256}.bias'] = ParameterRole('vector', (hidden,))
        roles[f'readout.{axis.sha256}.weight'] = ParameterRole('output', (finite, hidden))
        roles[f'readout.{axis.sha256}.bias'] = ParameterRole('vector', (finite,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def rosetta_profile_flow_width_family(base, delta, target):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, rosetta_profile_flow_roles(model)) for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if base.width != base.base_width or any(
        not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models
    ):
        raise ValueError('Rosetta source flow family binding drift')
    return width_family(*ledgers)
