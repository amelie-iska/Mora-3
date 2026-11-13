"""Shared native BETSE readout. Registered construction leaves roles inactive."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.betse_membrane import AXES, BETSESource, ROLE, _need
from .annotation_multihead import SharedAnnotationModel
from .annotation_mup import annotation_mup_parameter_roles
from .bioelectric_outcomes import _graph
from .mixed_payload_projection import exact_wire
from .parameterization import Axis, ParameterRole, parameter_ledger


@dataclass(frozen=True)
class BETSEPrediction:
    values: tuple
    units: tuple
    time_s: np.ndarray
    cell_offsets: np.ndarray
    membrane_offsets: np.ndarray


class NativeBETSEMembraneHead(nn.Module):
    def __init__(self, shared, source):
        super().__init__()
        _need(isinstance(shared, SharedAnnotationModel) and type(source) is BETSESource,
              'Exact shared annotation owner and sealed BETSE source required')
        shared.validate_parameterization_configuration()
        cfg = shared.get_extra_state()
        _need(shared.payload.embedding.weight.device.type == 'cpu', 'BETSE reference construction requires CPU')
        self.shared = shared
        self._shared = (shared,)
        self.source = source
        self._source = source
        self.width = cfg['width']
        self.base_width = cfg['mup']['base_width']
        self.readout_mult = cfg['mup']['output_mult']
        self.initial = nn.Linear(14, self.width, device='cpu', dtype=torch.float64)
        self.readout = nn.Linear(self.width, 9, device='cpu', dtype=torch.float64)
        with torch.no_grad():
            factor = math.sqrt(self.width / self.base_width)
            self.readout.weight.mul_(factor)
            self.readout.bias.mul_(factor)
        self.delta = nn.Linear(10, 6, device='cpu', dtype=torch.float64)
        self._configuration = self._local()

    def _local(self):
        return {'schema': 'Mora3.NativeBETSEMembraneHead/v1', 'source': self.source.binding(),
                'width': self.width, 'base_width': self.base_width, 'readout_mult': self.readout_mult,
                'axes': AXES, 'operator': 'ordered quadratic native time and INIT membrane residual v1',
                'training_admission': False}

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration()
        cfg = self.shared.get_extra_state()
        device = self.shared.payload.embedding.weight.device
        _need(self.shared is self._shared[0] and self.source is self._source
              and set(self._modules) == {'shared', 'initial', 'readout', 'delta'}
              and not self._parameters and not self._buffers
              and (self.width, self.base_width, self.readout_mult)
              == (cfg['width'], cfg['mup']['base_width'], cfg['mup']['output_mult'])
              and device.type in ('cpu', 'cuda')
              and self.shared.backbone.feature.weight.device == device
              and exact_wire(self._configuration, self._local()), 'BETSE operator or source binding differs')
        for layer, size in ((self.initial, (self.width, 14)), (self.readout, (9, self.width)), (self.delta, (6, 10))):
            _need(type(layer) is nn.Linear and not layer._modules and not layer._buffers
                  and set(layer._parameters) == {'weight', 'bias'}
                  and tuple(layer.weight.shape) == size and tuple(layer.bias.shape) == size[:1]
                  and all(p.dtype == torch.float64 and p.device == device and p.requires_grad
                          for p in (layer.weight, layer.bias)), 'Exact native BETSE affine parameter layout required')
        return {**self._local(), 'shared': cfg}

    def set_extra_state(self, value):
        _need(exact_wire(value, self.get_extra_state()), 'Exact BETSE checkpoint source and axes required')

    def _predict(self, x):
        query = [{'member': k, 'unit': u, 'axes': ['sim_time_s', 'native_' + axis]}
                 for k, u, axis, _, _ in AXES]
        records = tuple(_graph({'assay_query': query, 'intervention': event,
                                'sim_sampled_time_s': x.query_time.tolist()}) for event in x.protocol)
        graphs = [self.shared.payload.forward_task(record, task=task, identifiers=self.shared.sequence_identifiers())
                  for record, task in records]
        context = self.shared.backbone(graphs)[0][:, 0].double()
        device = context.device
        cell_case = torch.repeat_interleave(
            torch.arange(len(records), device=device, dtype=torch.long),
            torch.tensor(np.diff(x.cell_offsets), device=device, dtype=torch.long),
        )
        cell_v = torch.tensor(x.cell_voltage.T, device=device, dtype=torch.float64)
        init_time = torch.tensor(x.init_time, device=device, dtype=torch.float64).expand(len(cell_v), -1)
        state = torch.tanh(context[cell_case] + self.initial(torch.cat((cell_v / 0.1, init_time / 0.08), dim=1)))
        co = self.readout(state * (self.readout_mult / (self.width / self.base_width)))
        t = torch.tensor(x.query_time, device=device, dtype=torch.float64) / 1e-3
        basis = torch.stack((torch.ones_like(t), t, t.square()), dim=1)
        idx = torch.tensor(x.membrane_cell, device=device, dtype=torch.long)
        voltage = AXES[0][3] * (basis @ co[:, :3].T)
        k = AXES[1][3] * F.softplus(basis @ co[:, 3:6][idx].T)
        gap = torch.sigmoid(basis @ co[:, 6:9][idx].T)
        local = torch.cat((torch.tensor(x.membrane_voltage.T, device=device, dtype=torch.float64) / 0.1,
                           torch.tensor(x.membrane_area[:, None], device=device, dtype=torch.float64) / 1e-10,
                           torch.tensor(x.k_assignment[:, None], device=device, dtype=torch.float64),
                           torch.tensor(x.gap_assignment[:, None], device=device, dtype=torch.float64)), dim=1)
        delta = self.delta(local)
        potassium = k * torch.exp(torch.tanh(basis @ delta[:, :3].T))
        fraction = torch.sigmoid(torch.logit(gap, eps=torch.finfo(torch.float64).eps) + basis @ delta[:, 3:].T)
        return BETSEPrediction((voltage, potassium, fraction), tuple(axis[1] for axis in AXES),
                               x.query_time, x.cell_offsets, x.membrane_offsets)

    def forward(self, role):
        self.get_extra_state()
        return self._predict(self.source.conditions(role))

    def prepare_candidate(self, role, cursor):
        from mora3.research.primary_native_routes import PreparedNativeTargets
        _need(type(role) is str and role == ROLE, 'Only the explicit inactive BETSE candidate may prepare')
        first, end = self.source.candidate_cursors
        _need(type(cursor) is bytes and cursor in (first, end), 'Exact source-bound BETSE candidate cursor required')
        self.get_extra_state()
        if cursor == end:
            return PreparedNativeTargets(0, 'Inactive BETSE candidate window exhausted', None)
        return PreparedNativeTargets(sum(axis[-1] for axis in AXES), None, self._evaluate_candidate)

    def _evaluate_candidate(self, owner):
        from .full_owner import SourceBoundOwner
        _need(type(owner) is SourceBoundOwner and owner.betse is self
              and owner.get_extra_state()['source_snapshot_sha256'] == self.source.owner_snapshot_sha256,
              'Exact registered native BETSE owner and source snapshot required')
        condition = self.source.conditions('train')
        targets = self.source._candidate_targets(condition)
        prediction = self._predict(condition)
        errors = []
        for predicted, (target, mask), axis in zip(prediction.values, targets, AXES, strict=True):
            _need(tuple(predicted.shape) == target.shape == mask.shape and predicted.dtype == torch.float64
                  and int(mask.sum()) == axis[-1], 'Exact native BETSE observed position mask required')
            m = torch.tensor(mask, device=predicted.device, dtype=torch.bool)
            values = torch.tensor(target, device=predicted.device, dtype=torch.float64)
            error = (predicted[m] - values[m]) / axis[3]
            errors.append(error.square().mean())
        loss = torch.stack(errors).mean()
        _need(torch.isfinite(loss).item(), 'Finite native BETSE reference loss required')
        return {'loss': loss, 'observed_count': sum(axis[-1] for axis in AXES),
                'observed_by_channel': tuple(axis[-1] for axis in AXES),
                'cursor_after': self.source.candidate_cursors[1]}


def betse_membrane_parameter_roles(model, *, identifier_axis):
    _need(type(model) is NativeBETSEMembraneHead, 'Exact registered BETSE operator required')
    model.get_extra_state()
    hidden = Axis('width', 'hidden')
    axes = {name: Axis('finite', identity) for name, identity in
            model.source.binding()['route']['finite_axis_sha256'].items()}
    roles = {
        'initial.weight': ParameterRole('input', (hidden, axes['cell_input'])),
        'initial.bias': ParameterRole('vector', (hidden,)),
        'readout.weight': ParameterRole('output', (axes['base_output'], hidden)),
        'readout.bias': ParameterRole('vector', (axes['base_output'],)),
        'delta.weight': ParameterRole('output', (axes['membrane_output'], axes['membrane_input'])),
        'delta.bias': ParameterRole('vector', (axes['membrane_output'],)),
    }
    roles.update({'shared.' + k: v for k, v in annotation_mup_parameter_roles(
        model.shared, identifier_axis=identifier_axis).items()})
    parameter_ledger(model, roles)
    return roles
