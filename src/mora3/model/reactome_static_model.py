"""Static Reactome role and observed stoichiometry reconstruction."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.admet import canonical, digest
from mora3.data.reactome_static_pathways import (
    ROLE_AXIS, ReactomeStaticAxes, ReactomeStaticCondition, ReactomeStaticPacket, ReactomeStaticSource,
    log_stoichiometry,
)
from .annotation_multihead import SharedAnnotationModel
from .annotation_mup import annotation_mup_parameter_roles
from .mixed_payload_projection import exact_wire
from .morphology_mup import _native_roles
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger
from .physical_trainer import SharedPhysicalModel
from .tokengt import TokenGT


_SHARED_TRANSFORMS = ('feature', 'role', 'graph_token', 'layers', 'final_norm')
_PAYLOAD_HEADER = dict(name='native_context', units=None, status='observed', value_type='str')
_LONGEST_KNOWN_ENTITY_KIND = 'native-known-compartment-membership'


def _stream_bound(axes):
    # One observed canonical JSON scalar at most. Complete byte coverage makes
    # its UTF-8 byte count a conservative upper bound on shortest-path tokens.
    return (len(canonical(dict(kind=_LONGEST_KNOWN_ENTITY_KIND, directed=None)))
            + 2 + len(canonical(_PAYLOAD_HEADER)) + axes.max_entity_utf8_bytes)


class NativeReactomeStaticModel(nn.Module):
    """Reuse shared payload and ordinary transforms with a full source node rank."""

    def __init__(self, shared, axes, *, max_attention_elements=16_000_000):
        super().__init__()
        if type(shared) not in (SharedAnnotationModel, SharedPhysicalModel) or type(axes) is not ReactomeStaticAxes:
            raise ValueError('Exact shared annotation and complete native Reactome axes required')
        axes.binding()
        shared.validate_parameterization_configuration()
        annotation = shared.get_extra_state()
        native = shared.backbone.configuration()
        if (
            native['schema'] != 'tokengt-native-configuration-v3' or native['attention'] != 'vanilla'
            or native['width'] != native['feature_width'] or shared.payload.max_tokens < _stream_bound(axes)
            or type(max_attention_elements) is not int
            or max_attention_elements < (axes.max_entities + 1) ** 2 * native['heads'] * native['layers']
        ):
            raise ValueError('Complete native Reactome payload and attention budget required')
        self.shared = shared
        self._shared_owner = (shared,)
        self.axes = axes
        self.width = native['width']
        self.base_width = annotation['mup']['base_width']
        self.readout_mult = annotation['mup']['output_mult']
        self.max_attention_elements = max_attention_elements
        self.graph = TokenGT.from_configuration(dict(native, id_width=axes.max_nodes))
        for name in _SHARED_TRANSFORMS:
            setattr(self.graph, name, getattr(shared.backbone, name))
        reference = shared.backbone.feature.weight
        self.graph.incidence.to(device=reference.device, dtype=reference.dtype)
        options = dict(device=reference.device, dtype=reference.dtype)
        self.role_output = nn.Linear(self.width, len(ROLE_AXIS), **options)
        self.stoichiometry_output = nn.Linear(self.width, 1, **options)
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            for head in (self.role_output, self.stoichiometry_output):
                head.weight.mul_(scale)
                head.bias.mul_(scale)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeReactomeStaticModel/v1', source=self.axes.binding(),
            shared=self.shared.get_extra_state(), graph=self.graph.configuration(),
            width=self.width, base_width=self.base_width, readout_mult=self.readout_mult,
            max_attention_elements=self.max_attention_elements, max_entity_stream_upper_bound=_stream_bound(self.axes),
            topology='complete known compartment/species/reaction nodes and membership/participant incidence',
            target='all withheld native role occurrences and source-supplied positive stoichiometric coefficients',
            objective='per-graph mean role CE plus supplied-only mean log-stoichiometry MSE, then mean across graphs',
            training_admission=False,
        )

    def get_extra_state(self):
        if (
            self.shared is not self._shared_owner[0]
            or set(self._modules) != {'shared', 'graph', 'role_output', 'stoichiometry_output'}
            or self._parameters or self._buffers or type(self.graph) is not TokenGT
            or self.graph.incidence is self.shared.backbone.incidence
            or any(getattr(self.graph, name) is not getattr(self.shared.backbone, name) for name in _SHARED_TRANSFORMS)
            or any(type(head) is not nn.Linear or head._modules or head._buffers or head.bias is None
                   or (head.in_features, head.out_features) != (self.width, size)
                   or tuple(head.weight.shape) != (size, self.width) or tuple(head.bias.shape) != (size,)
                   for head, size in ((self.role_output, len(ROLE_AXIS)), (self.stoichiometry_output, 1)))
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Current Reactome source or shared operator configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact current native Reactome checkpoint configuration required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native Reactome checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _prepare(self, conditions):
        self.get_extra_state()
        conditions = tuple(conditions)
        if not conditions or any(type(item) is not ReactomeStaticCondition for item in conditions):
            raise ValueError('Complete target-free native Reactome conditions required')
        native = self.graph.configuration()
        graphs = tuple(item.graph() for item in conditions)
        elements = 0
        for graph, _ in graphs:
            self.axes.require_graph(graph)
            elements += (len(graph.nodes) + len(graph.edges) + 1) ** 2 * native['heads'] * native['layers']
            if elements > self.max_attention_elements:
                raise ValueError('Complete Reactome attention resource bound exceeded')
        return graphs

    def _predict(self, graphs):
        reference = self.graph.feature.weight
        role_logits, quantities = [], []
        for record, slots in graphs:
            input_fields = {(kind, i, m.name) for kind, entities in (('node', record.nodes), ('edge', record.edges))
                            for i, entity in enumerate(entities) for m in entity.measurements}
            identifiers = torch.eye(len(record.nodes), self.axes.max_nodes, dtype=reference.dtype, device=reference.device)
            graph = self.shared.payload(record, input_fields=input_fields, identifiers=identifiers)
            hidden = self.graph([graph])[0][0, 1:][list(slots)]
            features = hidden * (self.readout_mult / (self.width / self.base_width))
            role_logits.append(self.role_output(features).double())
            quantities.append(self.stoichiometry_output(features).squeeze(-1).double())
        return tuple(role_logits), tuple(quantities)

    def forward(self, conditions):
        return self._predict(self._prepare(conditions))

    def _reference_objective(self, packets):
        packets = tuple(packets)
        if not packets or any(type(packet) is not ReactomeStaticPacket for packet in packets):
            raise ValueError('Complete native Reactome packets required')
        graphs = self._prepare(packet.condition for packet in packets)
        targets = []
        for packet, (_, slots) in zip(packets, graphs, strict=True):
            if (type(packet.targets) is not tuple or len(packet.targets) != len(slots)
                    or any(type(x) is not tuple or len(x) != 2 or x[0] not in ROLE_AXIS for x in packet.targets)):
                raise ValueError('Exact native Reactome withheld roles and stoichiometry required')
            targets.append((tuple(ROLE_AXIS.index(role) for role, _ in packet.targets),
                            tuple(log_stoichiometry(value) for _, value in packet.targets)))
        logits, quantities = self._predict(graphs)
        role_losses, quantity_losses = [], []
        role_count = quantity_count = 0
        for prediction, quantity, (roles, observations) in zip(logits, quantities, targets, strict=True):
            role_count += len(roles)
            role_losses.append(F.cross_entropy(prediction, torch.tensor(roles, device=prediction.device)))
            observed = [i for i, value in enumerate(observations) if value is not None]
            quantity_count += len(observed)
            if observed:
                reference = quantity.new_tensor([observations[i] for i in observed])
                quantity_losses.append((quantity[observed] - reference).square().mean())
            else:
                quantity_losses.append(quantity.new_zeros(()))
        role_loss = torch.stack(role_losses).mean()
        quantity_loss = torch.stack(quantity_losses).mean()
        return dict(
            loss=role_loss + quantity_loss, role_loss=role_loss, log_stoichiometry_loss=quantity_loss,
            role_positions=role_count, supplied_stoichiometry_positions=quantity_count,
            role_logits=logits, log_stoichiometry=quantities,
        )

    def objective(self, source, batch, *, split='train'):
        if type(source) is not ReactomeStaticSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact bound native Reactome source required')
        return self._reference_objective(source.packets(batch, split=split))


def reactome_static_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeReactomeStaticModel:
        raise ValueError('Exact native Reactome shared owner required')
    model.get_extra_state()
    shared = annotation_mup_parameter_roles(model.shared, identifier_axis=identifier_axis)
    node_axis = tuple(f'Reactome-native-node-position:{i}' for i in range(model.axes.max_nodes))
    graph = _native_roles(model.graph, node_axis)
    roles = {'shared.' + name: role for name, role in shared.items()}
    roles.update({'graph.' + name: role for name, role in graph.items()})
    hidden = Axis('width', 'hidden')
    for name, categories in (
        ('role_output', ROLE_AXIS),
        ('stoichiometry_output', ('log-native-SBML-stoichiometric-coefficient',)),
    ):
        axis = Axis('finite', digest(canonical(['Reactome98-native-withheld-output-v1', categories])))
        roles[name + '.weight'] = ParameterRole('output', (axis, hidden))
        roles[name + '.bias'] = ParameterRole('vector', (axis,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles
