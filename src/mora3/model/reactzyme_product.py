"""Complete native enzyme/substrate context and original product bytes."""

import copy

import torch
from torch import nn

from mora3.data.admet import canonical
from mora3.data.provenance import Provenance
from mora3.data.reactzyme_current import ReactZymeCurrentSource, ReactZymeNativeTask, ReactZymeSourceAxes
from mora3.data.schema import Edge, GraphRecord, Measurement, Node
from mora3.data.tasks import ConditionalTask
from .mixed_payload_projection import exact_wire
from .morphology_mup import _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import parameter_ledger, width_family
from .pubchem_decoder import PubChemDecoder
from .sequence_mup import sequence_mup_parameter_roles
from .source_byte_objective import source_byte_objective, validate_byte_targets


_ENZYME = 'native-enzyme-sequence'
_SUBSTRATE = 'native-source-substrate'
_RELATION = 'known-source-reaction-packet'
_FIELDS = ('amino_acid_sequence', 'substrate_smiles')


def _stream_bound(axes):
    bounds = []
    for kind, field, size in zip((_ENZYME, _SUBSTRATE), _FIELDS, axes.max_source_bytes):
        entity = len(canonical(dict(kind=kind, directed=None)))
        field_metadata = len(canonical(dict(name=field, units=None, status='observed', value_type='str')))
        bounds.append(entity + 2 + field_metadata + size)
    return max(*bounds, len(canonical(dict(kind=_RELATION, directed=True))))


class NativeReactZymeProductModel(nn.Module):
    """Shares the existing sequence byte decoder and every graph parameter."""

    def __init__(self, decoder, axes, *, max_batch_size=64, max_total_target_bytes=None):
        super().__init__()
        if type(decoder) is not PubChemDecoder or type(axes) is not ReactZymeSourceAxes:
            raise ValueError('Exact shared byte decoder and complete ReactZyme source axes required')
        axes.binding()
        backbone = decoder.shared.backbone.configuration()
        if max_total_target_bytes is None and type(max_batch_size) is int:
            max_total_target_bytes = max_batch_size * axes.max_source_bytes[2]
        if (
            decoder.get_extra_state()['version'] != 2
            or backbone['schema'] != 'tokengt-native-configuration-v3' or backbone['id_width'] != 2
            or decoder.shared.payload.max_tokens < _stream_bound(axes)
            or type(max_batch_size) is not int or max_batch_size < 1
            or type(max_total_target_bytes) is not int or max_total_target_bytes < axes.max_source_bytes[2]
        ):
            raise ValueError('Complete two-entity native graph, source payload and product byte capacity required')
        self.decoder = decoder
        self._owner = (decoder,)
        self.axes = axes
        self.max_batch_size = max_batch_size
        self.max_total_target_bytes = max_total_target_bytes
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeReactZymeProductModel/v1', axes=self.axes.binding(),
            known_context='complete separate native enzyme sequence and substrate strings',
            relation='source packet membership, no physical association inference',
            target='complete original native product UTF8 and EOS',
            max_entity_stream_upper_bound=_stream_bound(self.axes),
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            decoder=self.decoder.get_extra_state(),
        )

    def get_extra_state(self):
        if (
            self.decoder is not self._owner[0] or set(self._modules) != {'decoder'}
            or self._parameters or self._buffers
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Current ReactZyme source or exact shared decoder ownership drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact current ReactZyme source/decoder checkpoint required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native ReactZyme checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def condition(self, conditions):
        self.get_extra_state()
        conditions = tuple(conditions)
        if not 1 <= len(conditions) <= self.max_batch_size:
            raise ValueError('Complete bounded native ReactZyme conditions required')
        for condition in conditions:
            self.axes.require_condition(condition)
        shared = self.decoder.shared
        reference = shared.backbone.feature.weight
        identifiers = torch.eye(2, dtype=reference.dtype, device=reference.device)
        task = ConditionalTask(
            ((('node', 0, _FIELDS[0]), 'condition'), (('node', 1, _FIELDS[1]), 'condition'),
             (('node', 1, 'requested_product'), 'target')),
            topology_known=True,
        )
        graphs = []
        for condition in conditions:
            nodes = (
                Node(_ENZYME, (Measurement(_FIELDS[0], condition.amino_acid_sequence, None, 'observed', True),)),
                Node(_SUBSTRATE, (Measurement(_FIELDS[1], condition.substrate_smiles, None, 'observed', True),
                                  Measurement('requested_product', None, None, 'missing', False))),
            )
            record = GraphRecord(
                'reactzyme-known-native-input', nodes, (Edge(0, 1, _RELATION, True),),
                (Provenance('native-model-input', 'reactzyme-v1', 'input', '0' * 64, 'source-admission-external'),),
            )
            graphs.append(shared.payload.forward_task(record, task=task, identifiers=identifiers))
        return shared.backbone(graphs)[0][:, 0]

    def _reference_objective(self, tasks, *, chunk_size, recompute):
        tasks = tuple(tasks)
        if not 1 <= len(tasks) <= self.max_batch_size or any(type(task) is not ReactZymeNativeTask for task in tasks):
            raise ValueError('Complete bounded native ReactZyme tasks required')
        targets = validate_byte_targets(
            (task.product_bytes for task in tasks), max_batch_size=self.max_batch_size,
            max_total_target_bytes=self.max_total_target_bytes, chunk_size=chunk_size, recompute=recompute,
        )
        allowed = set(self.axes.native_alphabets[2])
        if any(not 1 <= len(raw) <= self.axes.max_source_bytes[2] or set(raw.decode('utf-8')) - allowed for raw in targets):
            raise ValueError('Complete native ReactZyme product bytes outside the bound source scope')
        context = self.condition(task.condition for task in tasks)
        return source_byte_objective(
            self.decoder, context, targets, chunk_size=chunk_size, recompute=recompute,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
        )

    def objective(self, source, batch, *, split='train', chunk_size=1024, recompute=False):
        if type(source) is not ReactZymeCurrentSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact current guarded ReactZyme source task batch required')
        return self._reference_objective(source.tasks(batch, split=split), chunk_size=chunk_size, recompute=recompute)

    @torch.no_grad()
    def generate(self, conditions, *, max_new_tokens):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.axes.max_source_bytes[2] + 1:
            raise ValueError('Explicit bounded full native ReactZyme product token budget required')
        conditions = tuple(conditions)
        result = self.decoder._generate_from_context(self.condition(conditions), max_new_tokens=max_new_tokens)
        result['target_fields'] = ('product_smiles',) * len(conditions)
        return result


def reactzyme_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeReactZymeProductModel:
        raise ValueError('Exact current source ReactZyme shared decoder owner required')
    model.get_extra_state()
    roles = {'decoder.' + name: role for name, role in
             sequence_mup_parameter_roles(model.decoder, identifier_axis=identifier_axis).items()}
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def reactzyme_width_family(base, delta, target, *, identifier_axis):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, reactzyme_parameter_roles(model, identifier_axis=identifier_axis)) for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if any(not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models):
        raise ValueError('ReactZyme exact source/decoder width family binding drift')
    if base.decoder.rnn.hidden_size != base.decoder.get_extra_state()['mup']['base_width']:
        raise ValueError('Actual ReactZyme base member width differs from declared base')
    return width_family(*ledgers)
