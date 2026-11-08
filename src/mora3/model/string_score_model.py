"""Predict all four STRING native scores from an unordered exact sequence pair."""

import copy
import math

import torch
from torch import nn

from mora3.data.admet import canonical, digest
from mora3.data.provenance import Provenance
from mora3.data.schema import Edge, GraphRecord, Measurement, Node
from mora3.data.string_score_current import (
    SCORE_AXIS, SCOPE, TAXON, UNIT, StringScoreAxes, StringScoreCurrentSource, StringScorePacket, valid_score_profile,
)
from .annotation_multihead import SharedAnnotationModel
from .annotation_mup import annotation_mup_parameter_roles
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger
from .physical_trainer import SharedPhysicalModel


_PROTEIN = 'same-release-native-protein'
_EDGE = 'unordered-source-score-query'
_SEQUENCE = 'native_amino_acid_sequence'
_SCOPE = 'association_scope'
_TAXON = 'organism_taxon'
_PROVENANCE = (Provenance('string-score-known-input', 'STRING-v12.0', 'input', '0'*64, 'source-audit-external'),)


def _stream_bound(axes):
    # Complete byte coverage bounds the tokens by the original UTF-8 byte
    # length, including exact entity/field metadata and boundary controls.
    node = len(canonical(dict(kind=_PROTEIN, directed=None)))
    edge = len(canonical(dict(kind=_EDGE, directed=True)))
    def field(name, byte_count):
        return 2 + len(canonical(dict(name=name, units=None, status='observed', value_type='str'))) + byte_count
    return max(node + field(_SEQUENCE, axes.max_sequence_utf8_bytes),
               edge + field(_SCOPE, len(SCOPE.encode('utf-8'))) + field(_TAXON, len(str(TAXON))))


def _graph(condition):
    # This explicit lossless sequence sort fixes the model orientation without
    # disclosing provider IDs or scores. Source orientation remains in audit.
    nodes = tuple(Node(_PROTEIN, (Measurement(_SEQUENCE, value, None, 'observed', True),))
                  for value in sorted(condition.sequences))
    fields = (Measurement(_SCOPE, SCOPE, None, 'observed', True),
              Measurement(_TAXON, str(TAXON), None, 'observed', True))
    edge = Edge(0, 1, _EDGE, True, fields)
    return GraphRecord('string-score-known-input', nodes, (edge,), _PROVENANCE)


class NativeStringScoreModel(nn.Module):
    def __init__(self, shared, axes, *, max_batch_size=64, max_batch_sequence_bytes=None):
        super().__init__()
        if type(shared) not in (SharedAnnotationModel, SharedPhysicalModel) or type(axes) is not StringScoreAxes:
            raise ValueError('Exact shared annotation and complete original STRING axes required')
        axes.binding()
        shared.validate_parameterization_configuration()
        configuration = shared.get_extra_state()
        native = shared.backbone.configuration()
        if max_batch_sequence_bytes is None and type(max_batch_size) is int:
            max_batch_sequence_bytes = 2 * axes.max_sequence_utf8_bytes * max_batch_size
        if (
            native['schema'] != 'tokengt-native-configuration-v3' or native['id_width'] != 2
            or shared.payload.max_tokens < _stream_bound(axes)
            or type(max_batch_size) is not int or max_batch_size < 1
            or type(max_batch_sequence_bytes) is not int or max_batch_sequence_bytes < 2 * axes.max_sequence_utf8_bytes
        ):
            raise ValueError('Complete two-node same-release STRING payload capacity required')
        self.shared = shared
        self._shared_owner = (shared,)
        self.axes = axes
        self.width = configuration['width']
        self.base_width = configuration['mup']['base_width']
        self.readout_mult = configuration['mup']['output_mult']
        self.max_batch_size = max_batch_size
        self.max_batch_sequence_bytes = max_batch_sequence_bytes
        reference = shared.payload.embedding.weight
        self.readout = nn.Linear(self.width, len(SCORE_AXIS), device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            self.readout.weight.mul_(scale)
            self.readout.bias.mul_(scale)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeStringScoreModel/v1', source=self.axes.binding(),
            shared=self.shared.get_extra_state(), width=self.width, base_width=self.base_width,
            readout_mult=self.readout_mult, max_batch_size=self.max_batch_size,
            max_batch_sequence_bytes=self.max_batch_sequence_bytes,
            max_entity_stream_upper_bound=_stream_bound(self.axes),
            condition='complete exact same-release FASTA sequence multiset; lexical full-sequence ordering; provider IDs audit only',
            objective='uniform mean squared error over four native score fractions and original observed source pairs',
            score_policy='literal 0..1000 native integers, zero retained as source value, no unobserved-pair or binding labels',
            training_admission=False,
        )

    def get_extra_state(self):
        if (
            self.shared is not self._shared_owner[0] or set(self._modules) != {'shared', 'readout'}
            or self._parameters or self._buffers or type(self.readout) is not nn.Linear
            or self.readout._modules or self.readout._buffers or self.readout.bias is None
            or (self.readout.in_features, self.readout.out_features) != (self.width, len(SCORE_AXIS))
            or tuple(self.readout.weight.shape) != (len(SCORE_AXIS), self.width)
            or tuple(self.readout.bias.shape) != (len(SCORE_AXIS),)
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Current STRING source or exact shared operator configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact current STRING score-profile checkpoint configuration required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native STRING score-profile checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, conditions):
        self.get_extra_state()
        conditions = tuple(conditions)
        if not 1 <= len(conditions) <= self.max_batch_size:
            raise ValueError('Complete bounded native STRING sequence pairs required')
        for condition in conditions:
            self.axes.require_condition(condition)
        if sum(len(sequence) for condition in conditions for sequence in condition.sequences) > self.max_batch_sequence_bytes:
            raise ValueError('Complete original native STRING sequence batch exceeds resource budget')
        reference = self.shared.backbone.feature.weight
        identifiers = torch.eye(2, device=reference.device, dtype=reference.dtype)
        fields = {('node', 0, _SEQUENCE), ('node', 1, _SEQUENCE), ('edge', 0, _SCOPE), ('edge', 0, _TAXON)}
        graphs = [self.shared.payload(_graph(condition), input_fields=fields, identifiers=identifiers)
                  for condition in conditions]
        hidden = self.shared.backbone(graphs)[0][:, 0]
        logits = self.readout(hidden * (self.readout_mult / (self.width / self.base_width)))
        return logits.double().sigmoid()

    def _reference_objective(self, packets):
        packets = tuple(packets)
        if not 1 <= len(packets) <= self.max_batch_size or any(
            type(packet) is not StringScorePacket or not valid_score_profile(packet.scores) for packet in packets
        ):
            raise ValueError('Complete original STRING source score vectors required')
        predictions = self(tuple(packet.condition for packet in packets))
        native_scores = predictions.new_tensor([packet.scores for packet in packets])
        return dict(
            loss=(predictions - native_scores / 1000.).square().mean(),
            predicted_fractions=predictions, predicted_native_scores=predictions * 1000.,
            observed_source_score_positions=native_scores.numel(),
            literal_zero_positions=sum(score == 0 for packet in packets for score in packet.scores),
        )

    def objective(self, source, batch, *, split='train'):
        if type(source) is not StringScoreCurrentSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact bound current STRING score source required')
        return self._reference_objective(source.packets(batch, split=split))


def string_score_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeStringScoreModel:
        raise ValueError('Exact native STRING score-profile owner required')
    model.get_extra_state()
    roles = {'shared.' + name: role for name, role in
             annotation_mup_parameter_roles(model.shared, identifier_axis=identifier_axis).items()}
    axis = Axis('finite', digest(canonical(['STRING-v12.0-source-native-score-profile-v1', SCORE_AXIS, UNIT])))
    roles['readout.weight'] = ParameterRole('output', (axis, Axis('width', 'hidden')))
    roles['readout.bias'] = ParameterRole('vector', (axis,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles
