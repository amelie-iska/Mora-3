"""Joint literal BioGRID chemical/gene source annotations with shared TokenGT."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.admet import canonical, digest
from mora3.data.biogrid_chemical_current import (
    KNOWN_FIELDS, BioGRIDChemicalAxes, BioGRIDChemicalCurrentSource, BioGRIDChemicalPacket,
)
from mora3.data.provenance import Provenance
from mora3.data.schema import Edge, GraphRecord, Measurement, Node
from .annotation_multihead import SharedAnnotationModel
from .annotation_mup import annotation_mup_parameter_roles
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger
from .physical_trainer import SharedPhysicalModel


_CHEMICAL = 'native-bioGRID-chemical-reference'
_GENE = 'native-bioGRID-gene-reference'
_RELATION = 'known-source-annotation-packet'
_PROVENANCE = (Provenance('biogrid-chemical-known-input', 'BioGRID5.0.262', 'input', '0'*64, 'source-audit-external'),)


def _entity_stream_bound(axes):
    bounds = dict(zip(KNOWN_FIELDS, axes.known_field_max_utf8_bytes, strict=True))
    return max(len(canonical(dict(kind=kind, directed=None))) + sum(
        2 + len(canonical(dict(name=field, units=None, status='observed', value_type='str'))) + bounds[field]
        for field in fields
    ) for kind, fields in ((_CHEMICAL, KNOWN_FIELDS[:2]), (_GENE, KNOWN_FIELDS[2:])))


def _graph(condition):
    values = condition.inputs()
    nodes = tuple(Node(kind, tuple(Measurement(name, values[name], None, 'observed', True) for name in fields))
                  for kind, fields in ((_CHEMICAL, KNOWN_FIELDS[:2]), (_GENE, KNOWN_FIELDS[2:])))
    return GraphRecord('biogrid-chemical-known-input', nodes, (Edge(0, 1, _RELATION, True),), _PROVENANCE)


class NativeBioGRIDChemicalModel(nn.Module):
    """Model a literal joint category per original source annotation occurrence."""

    def __init__(self, shared, axes, *, max_batch_size=128):
        super().__init__()
        if type(shared) not in (SharedAnnotationModel, SharedPhysicalModel) or type(axes) is not BioGRIDChemicalAxes:
            raise ValueError('Exact shared annotation and complete BioGRID chemical axes required')
        axes.binding()
        shared.validate_parameterization_configuration()
        configuration = shared.get_extra_state()
        native = shared.backbone.configuration()
        if (
            native['schema'] != 'tokengt-native-configuration-v3' or native['id_width'] != 2
            or shared.payload.max_tokens < _entity_stream_bound(axes)
            or type(max_batch_size) is not int or max_batch_size < 1
        ):
            raise ValueError('Complete two-node native BioGRID source and payload capacity required')
        self.shared = shared
        self._shared_owner = (shared,)
        self.axes = axes
        self.width = configuration['width']
        self.base_width = configuration['mup']['base_width']
        self.readout_mult = configuration['mup']['output_mult']
        self.max_batch_size = max_batch_size
        reference = shared.payload.embedding.weight
        self.readout = nn.Linear(self.width, len(axes.target_pairs), device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            self.readout.weight.mul_(scale)
            self.readout.bias.mul_(scale)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeBioGRIDChemicalModel/v1', source=self.axes.binding(),
            shared=self.shared.get_extra_state(), width=self.width, base_width=self.base_width,
            readout_mult=self.readout_mult, max_batch_size=self.max_batch_size,
            max_entity_stream_upper_bound=_entity_stream_bound(self.axes),
            objective='joint literal action and interaction type CE per original source occurrence',
            semantics='curated publication-supported source annotation; no identity/binding or missing-pair inference',
            training_admission=False,
        )

    def get_extra_state(self):
        count = len(self.axes.target_pairs)
        if (
            self.shared is not self._shared_owner[0] or set(self._modules) != {'shared', 'readout'}
            or self._parameters or self._buffers or type(self.readout) is not nn.Linear
            or self.readout._modules or self.readout._buffers or self.readout.bias is None
            or (self.readout.in_features, self.readout.out_features) != (self.width, count)
            or tuple(self.readout.weight.shape) != (count, self.width) or tuple(self.readout.bias.shape) != (count,)
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Current BioGRID chemical source or exact shared owner configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact current BioGRID chemical checkpoint configuration required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native BioGRID chemical checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, conditions):
        self.get_extra_state()
        conditions = tuple(conditions)
        if not 1 <= len(conditions) <= self.max_batch_size:
            raise ValueError('Complete bounded native BioGRID chemical conditions required')
        for condition in conditions:
            self.axes.require_condition(condition)
        reference = self.shared.backbone.feature.weight
        identifiers = torch.eye(2, device=reference.device, dtype=reference.dtype)
        fields = {('node', i // 2, field) for i, field in enumerate(KNOWN_FIELDS)}
        graphs = [self.shared.payload(_graph(condition), input_fields=fields, identifiers=identifiers)
                  for condition in conditions]
        hidden = self.shared.backbone(graphs)[0][:, 0]
        return self.readout(hidden * (self.readout_mult / (self.width / self.base_width))).double()

    def _reference_objective(self, packets):
        packets = tuple(packets)
        if not 1 <= len(packets) <= self.max_batch_size or any(
            type(packet) is not BioGRIDChemicalPacket or packet.target not in self.axes.target_pairs
            for packet in packets
        ):
            raise ValueError('Complete original BioGRID occurrences on the exact native joint axis required')
        logits = self(tuple(packet.condition for packet in packets))
        targets = torch.tensor([self.axes.target_pairs.index(packet.target) for packet in packets], device=logits.device)
        return dict(loss=F.cross_entropy(logits, targets), logits=logits, original_occurrences=len(packets))

    def objective(self, source, batch, *, split='train'):
        if type(source) is not BioGRIDChemicalCurrentSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact bound current BioGRID chemical source required')
        return self._reference_objective(source.packets(batch, split=split))


def biogrid_chemical_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeBioGRIDChemicalModel:
        raise ValueError('Exact native BioGRID chemical model required')
    model.get_extra_state()
    roles = {'shared.' + name: role for name, role in
             annotation_mup_parameter_roles(model.shared, identifier_axis=identifier_axis).items()}
    axis = Axis('finite', digest(canonical(['BioGRID5.0.262-original-joint-class-v1', model.axes.target_pairs])))
    roles['readout.weight'] = ParameterRole('output', (axis, Axis('width', 'hidden')))
    roles['readout.bias'] = ParameterRole('vector', (axis,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles
