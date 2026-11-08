"""Optional source-bound SCRI likelihoods on the existing shared TokenGT.

This component registers no primary-owner route and issues no training admission.
Complete conditions use the owner's existing, explicitly lossy positional payload
pooling. Native target values never enter that payload. Source axis digests bind
the declared representation, not the truth, rights, or admission of source rows.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.physical_trainer import SharedPhysicalModel
from mora3.model.annotation_mup import annotation_mup_parameter_roles
from mora3.model.mixed_payload_projection import exact_wire
from mora3.model.parameterization import Axis, ParameterRole, parameter_ledger, width_family
from mora3.model.mu_optimizer import _storage_signature
from .contracts import TransitionPair, canonical, compile_transition, digest, number, text


@dataclass(frozen=True)
class SCRITargetAxis:
    """Exact ordered native axes and likelihood; no unit or label conversion.

    source_sha256 identifies the source axis/protocol artifact. Per-observation
    artifact hashes remain in TransitionPair audit records and are not features.
    Gaussian applies to native continuous coordinates, Poisson to raw integer
    counts only, and categorical to explicit per-feature source category tuples.
    """
    source_sha256: str
    features: tuple[tuple[str, str, str], ...]
    likelihood: str
    categories: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        digest(self.source_sha256, 'source axis/protocol SHA256')
        if type(self.features) is not tuple or not self.features:
            raise ValueError('complete ordered native feature axis required')
        for feature in self.features:
            if type(feature) is not tuple or len(feature) != 3:
                raise ValueError('native modality, feature and unit tuple required')
            for value in feature:
                text(value, 'native feature coordinate')
        if len({x[:2] for x in self.features}) != len(self.features):
            raise ValueError('duplicate native feature identity')
        if self.likelihood not in ('gaussian', 'poisson', 'categorical'):
            raise ValueError('explicit supported native likelihood required')
        if self.likelihood == 'poisson' and any(x[2] != 'count' for x in self.features):
            raise ValueError('Poisson requires explicit native count units; no normalized expression')
        if type(self.categories) is not tuple:
            raise ValueError('immutable categorical axes required')
        if self.likelihood == 'categorical':
            if len(self.categories) != len(self.features):
                raise ValueError('one complete native category axis per feature required')
            for axis in self.categories:
                if type(axis) is not tuple or len(axis) < 2 or len(set(axis)) != len(axis):
                    raise ValueError('unique finite source categories required')
                for value in axis:
                    text(value, 'native category')
        elif self.categories:
            raise ValueError('numeric likelihood accepts no category axis')

    @property
    def output_axis(self):
        if self.likelihood == 'gaussian':
            return tuple((f, statistic) for statistic in ('mean', 'raw_scale') for f in self.features)
        if self.likelihood == 'poisson':
            return tuple((f, 'log_rate') for f in self.features)
        return tuple((f, category) for f, axis in zip(self.features, self.categories) for category in axis)


def masked_native_objective(parameters, values, observed, axis):
    """Mean NLL over explicitly observed native coordinates, in float64.

    Missing coordinates must be None and never participate in likelihood math.
    Empty supervision returns connected zero and observed_targets=0; callers
    must retain that denominator and cannot count it as an optimizer window.
    Gaussian scale is softplus(raw_scale)+1e-6 in the target's native unit.
    Poisson retains the log-factorial constant and performs no library scaling.
    """
    if type(axis) is not SCRITargetAxis:
        raise ValueError('exact source target axis required')
    axis.__post_init__()
    if (not isinstance(parameters, torch.Tensor) or not parameters.is_floating_point()
            or parameters.ndim != 2 or parameters.shape[1] != len(axis.output_axis)
            or not torch.isfinite(parameters).all().item()):
        raise ValueError('finite complete native prediction parameters required')
    batch, dimension = parameters.shape[0], len(axis.features)
    if (type(values) not in (tuple, list) or type(observed) not in (tuple, list)
            or len(values) != batch or len(observed) != batch):
        raise ValueError('complete row-aligned native targets and masks required')
    pieces = []
    for row in range(batch):
        if (type(values[row]) not in (tuple, list) or type(observed[row]) not in (tuple, list)
                or len(values[row]) != dimension or len(observed[row]) != dimension):
            raise ValueError('complete native target feature axis required')
        offset = 0
        for col in range(dimension):
            value, present = values[row][col], observed[row][col]
            if type(present) is not bool or present != (value is not None):
                raise ValueError('explicit missing None and Boolean observation mask required')
            classes = len(axis.categories[col]) if axis.likelihood == 'categorical' else 0
            if present:
                if axis.likelihood == 'categorical':
                    if type(value) is not str or value not in axis.categories[col]:
                        raise ValueError('unknown native category; no implicit class coercion')
                    logits = parameters[row, offset:offset+classes].double()
                    pieces.append(-F.log_softmax(logits, dim=0)[axis.categories[col].index(value)])
                else:
                    number(value, 'observed native target')
                    target = parameters.new_tensor(value, dtype=torch.float64)
                    if axis.likelihood == 'poisson':
                        if value < 0 or int(value) != value or value > 2**53:
                            raise ValueError('exact nonnegative integer native count required')
                        log_rate = parameters[row, col].double()
                        pieces.append(log_rate.exp() - target*log_rate + torch.lgamma(target+1))
                    else:
                        mean = parameters[row, col].double()
                        scale = F.softplus(parameters[row, dimension+col].double()) + 1e-6
                        pieces.append(0.5*((target-mean)/scale).square() + scale.log()
                                      + 0.5*math.log(2*math.pi))
            offset += classes
    loss = torch.stack(pieces).mean() if pieces else parameters.double().sum()*0
    if not torch.isfinite(loss).item():
        raise ValueError('nonfinite native likelihood; no clipping or substituted targets')
    return {'loss': loss, 'observed_targets': len(pieces), 'training_admitted': False}


def _condition_graph(condition):
    raw = canonical(condition).decode('utf-8')
    record = GraphRecord('scri-condition', (Node('cell-state', (
        Measurement('condition', raw, None, 'observed', True),
        Measurement('requested_output', None, None, 'missing', False))),), (),
        (Provenance('native-model-input', 'scri-condition-v1', 'input', '0'*64,
                    'source-admission-external'),))
    task = ConditionalTask(((('node', 0, 'condition'), 'condition'),
                            (('node', 0, 'requested_output'), 'target')), topology_known=True)
    return record, task


class SharedSCRIModel(nn.Module):
    """Borrow the actual shared owner by identity; add one finite native head."""
    def __init__(self, shared, axis, *, max_batch_size=16, max_condition_bytes=65536,
                 max_spatial_cells=128):
        super().__init__()
        if type(shared) not in (SharedAnnotationModel, SharedPhysicalModel) or type(axis) is not SCRITargetAxis:
            raise ValueError('exact existing annotation/physical shared owner and source axis required')
        shared.validate_parameterization_configuration()
        axis.__post_init__()
        if (any(type(x) is not int or x <= 0 for x in
                (max_batch_size, max_condition_bytes, max_spatial_cells)) or max_spatial_cells > 128):
            raise ValueError('positive explicit complete-condition resource bounds required')
        self.shared, self.axis = shared, axis
        self.max_batch_size, self.max_condition_bytes = max_batch_size, max_condition_bytes
        self.max_spatial_cells = max_spatial_cells
        self._owner_reference = (shared, axis)
        cfg = shared.get_extra_state()
        self.output = nn.Linear(cfg['width'], len(axis.output_axis))
        # Same fresh SP-to-base-variance conversion as the native annotation head.
        with torch.no_grad():
            factor = math.sqrt(cfg['width']/cfg['mup']['base_width'])
            self.output.weight.mul_(factor)
            self.output.bias.mul_(factor)
        owner_parameter = next(shared.parameters())
        self.output.to(device=owner_parameter.device, dtype=owner_parameter.dtype)
        self._configuration = self._local_state()

    def _local_state(self):
        shared = self.shared.get_extra_state()
        return dict(schema='Mora3.SharedSCRIModel/v1', axis=asdict(self.axis),
                    width=shared['width'], base_width=shared['mup']['base_width'],
                    output_mult=shared['mup']['output_mult'],
                    condition='compile_transition.condition-only;shared-positional-payload-is-lossy',
                    readout='fresh-SP-times-sqrt-width-over-base;input-output-mult-over-width-ratio',
                    gaussian_scale='softplus(raw_scale)+1e-6-native-unit',
                    objective='mean-NLL-over-observed-native-coordinates;float64;empty-count-zero',
                    max_batch_size=self.max_batch_size, max_condition_bytes=self.max_condition_bytes,
                    spatial=dict(policy='typed-baseline-neighborhood;query-node-zero;native-radius-adjacency',
                                 identifiers='exact-identity-zero-padded;node-count-must-not-exceed-existing-rank',
                                 identifier_capacity=self.shared.backbone.incidence.in_features//2,
                                 max_cells=self.max_spatial_cells),
                    source_admission_issued=False)

    def get_extra_state(self):
        if (self.shared is not self._owner_reference[0] or self.axis is not self._owner_reference[1]
                or self._parameters or self._buffers or set(self._modules) != {'shared', 'output'}
                or type(self.output) is not nn.Linear or self.output._modules or self.output._buffers
                or set(self.output._parameters) != {'weight', 'bias'}
                or (self.output.in_features, self.output.out_features) !=
                   (self.shared._configuration['width'], len(self.axis.output_axis))
                or tuple(self.output.weight.shape) != (len(self.axis.output_axis), self.shared._configuration['width'])
                or tuple(self.output.bias.shape) != (len(self.axis.output_axis),)
                or not exact_wire(self._local_state(), self._configuration)):
            raise ValueError('SCRI shared ownership, native axis, head or configuration drift')
        return dict(**copy.deepcopy(self._configuration), shared=self.shared.get_extra_state())

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('SCRI source/axis/shared checkpoint mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Preflight every child before PyTorch can partially copy the shared core.
        expected = self.state_dict()
        incoming = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if set(incoming) != set(expected):
            raise ValueError('complete exact SCRI checkpoint keys required')
        for key, reference in expected.items():
            value = incoming[key]
            if isinstance(reference, torch.Tensor):
                if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                        or value.dtype != reference.dtype or not torch.isfinite(value).all().item()):
                    raise ValueError('SCRI checkpoint tensor shape, dtype or finiteness mismatch')
            elif not exact_wire(value, reference):
                raise ValueError('SCRI checkpoint source/axis/shared configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _tasks(self, pairs):
        self.get_extra_state()
        if type(pairs) not in (tuple, list) or not 0 < len(pairs) <= self.max_batch_size:
            raise ValueError('bounded nonempty explicit transition batch required')
        tasks = []
        for pair in pairs:
            if type(pair) is not TransitionPair:
                raise ValueError('typed source transition pair required')
            task = compile_transition(pair)
            if tuple(tuple(x) for x in task['target']['axis']) != self.axis.features:
                raise ValueError('complete native source target axis/order/unit mismatch')
            if len(canonical(task['condition'])) > self.max_condition_bytes:
                raise ValueError('complete SCRI condition exceeds byte bound; no truncation')
            tasks.append(task)
        return tasks

    def forward(self, pairs):
        graphs = []
        for task in self._tasks(pairs):
            record, query = _condition_graph(task['condition'])
            graphs.append(self.shared.payload.forward_task(
                record, task=query, identifiers=self.shared.sequence_identifiers()))
        latent = self.shared.backbone(graphs)[0][:, 0]
        # Apply the shared native owner's readout scaling exactly once.
        return self.output(self.shared._readout_features(latent))

    def objective(self, parameters, pairs):
        tasks = self._tasks(pairs)
        return masked_native_objective(parameters, [t['target']['values'] for t in tasks],
                                       [t['target']['observed'] for t in tasks], self.axis)

    def forward_spatial(self, queries):
        """Predict query-node targets from a typed baseline neighborhood graph.

        Uses the exact same payload, backbone and head as transition prediction.
        No target truth is accepted by the query builder. Native radius adjacency
        has symmetric directed encoding for the existing payload implementation.
        The existing backbone fixes identifier capacity: a rank-two primary owner
        can accept at most two cells here. Larger graphs fail closed; this method
        never truncates cells, changes the owner, or fabricates low-rank node IDs.
        """
        from .spatial import SpatialQuery, validate_spatial_query
        self.get_extra_state()
        if type(queries) not in (tuple, list) or not 0 < len(queries) <= self.max_batch_size:
            raise ValueError('bounded nonempty spatial query batch required')
        rank = self.shared.backbone.incidence.in_features // 2
        weight = self.shared.backbone.feature.weight
        graphs = []
        for query in queries:
            if type(query) is not SpatialQuery:
                raise ValueError('exact typed native spatial query required')
            validate_spatial_query(query)
            if not exact_wire(query.target_axis, self.axis.features):
                raise ValueError('spatial target source axis/order/unit mismatch')
            count = len(query.record.nodes)
            if not 0 < count <= min(self.max_spatial_cells, rank):
                raise ValueError(f'complete spatial cells={count} exceed existing exact identifier capacity={rank} '
                                 f'or node bound={self.max_spatial_cells}')
            if query.query_node_index != 0:
                raise ValueError('native spatial query must be node zero')
            if len(canonical(asdict(query.record))) + len(query.task.canonical_bytes()) > self.max_condition_bytes:
                raise ValueError('complete spatial graph exceeds byte bound; no truncation')
            identifiers = torch.eye(count, rank, device=weight.device, dtype=weight.dtype)
            graphs.append(self.shared.payload.forward_task(
                query.record, task=query.task, identifiers=identifiers))
        latent = self.shared.backbone(graphs)[0][:, 0]
        return self.output(self.shared._readout_features(latent))


def scri_parameter_roles(model, *, identifier_axis):
    """Complete local μP roles; does not modify full_owner_mup's operator registry."""
    if type(model) is not SharedSCRIModel:
        raise ValueError('exact optional SCRI wrapper required')
    model.get_extra_state()
    roles = {'shared.'+name: role for name, role in
             annotation_mup_parameter_roles(model.shared, identifier_axis=identifier_axis).items()}
    finite = Axis('finite', sha256(canonical(['SCRI-native-output-v1', asdict(model.axis),
                                              model.axis.output_axis])).hexdigest())
    roles.update({'output.weight': ParameterRole('output', (finite, Axis('width', 'hidden'))),
                  'output.bias': ParameterRole('vector', (finite,))})
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def scri_width_family(base, delta, target, *, identifier_axis):
    """Component width family only; no whole-owner, corpus or resource readiness."""
    from mora3.model.full_owner_mup import _without_member_dimensions
    models = (base, delta, target)
    ledgers = [parameter_ledger(m, scri_parameter_roles(m, identifier_axis=identifier_axis)) for m in models]
    reference = _without_member_dimensions(base.get_extra_state())
    if any(not exact_wire(_without_member_dimensions(m.get_extra_state()), reference) for m in models):
        raise ValueError('SCRI finite source/axis/policy or shared configuration drift across widths')
    if base._configuration['width'] != base._configuration['base_width']:
        raise ValueError('actual SCRI base width must equal its declaration')
    return width_family(*ledgers)
