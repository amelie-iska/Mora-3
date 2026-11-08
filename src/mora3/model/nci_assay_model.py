"""Native NCI assay likelihoods and connectivity generation with the shared encoder."""

import copy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.nci_current_assays import (
    CONCLUSION, SMILES, NciAssayPartition, NciAssaySourceAxes, NciCurrentAssaySource, NciCurrentAssayTask, nci_known_condition,
)
from mora3.data.provenance import Provenance
from mora3.data.schema import GraphRecord, Measurement, Node
from mora3.data.tasks import ConditionalTask
from .mixed_payload_projection import exact_wire
from .morphology_mup import _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger, width_family
from .pubchem_decoder import PubChemDecoder
from .sequence_mup import sequence_mup_parameter_roles
from .source_byte_objective import source_byte_objective, validate_byte_targets


@dataclass(frozen=True)
class NciPredictionResult:
    partition: NciAssayPartition
    indices: tuple[int, ...]
    loss: torch.Tensor
    logits: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    scale: torch.Tensor | None = None


def _readout_axis(partition):
    coordinates = partition.coordinate_axis
    if partition.family == CONCLUSION:
        return coordinates
    return tuple((statistic, coordinate) for statistic in ('mean', 'raw_positive_scale') for coordinate in coordinates)


class NativeNciAssayModel(nn.Module):
    """Predict literal source summaries and generate full source connectivity strings."""

    def __init__(self, decoder, axes, *, max_batch_size=128, max_total_target_bytes=None):
        super().__init__()
        if type(decoder) is not PubChemDecoder or type(axes) is not NciAssaySourceAxes:
            raise ValueError('Exact shared byte decoder and complete native NCI source axes required')
        axes.binding()
        if max_total_target_bytes is None and type(max_batch_size) is int:
            max_total_target_bytes = max_batch_size * axes.max_target_utf8_bytes
        if (
            decoder.get_extra_state()['version'] != 2
            or type(max_batch_size) is not int or max_batch_size < 1
            or type(max_total_target_bytes) is not int or max_total_target_bytes < axes.max_target_utf8_bytes
        ):
            raise ValueError('Explicit native decoder and complete bounded NCI byte target capacity required')
        self.decoder = decoder
        self._decoder_reference = (decoder,)
        self.axes = axes
        self.max_batch_size = max_batch_size
        self.max_total_target_bytes = max_total_target_bytes
        cfg = decoder.shared.get_extra_state()
        self.width = cfg['width']
        self.base_width = cfg['mup']['base_width']
        self.readout_mult = cfg['mup']['output_mult']
        self.heads = nn.ModuleDict({
            partition.identity: nn.Linear(self.width, len(_readout_axis(partition)))
            for partition in self.axes.partitions
        })
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            for head in self.heads.values():
                head.weight.mul_(scale)
                head.bias.mul_(scale)
        reference = decoder.embedding.weight
        self.heads.to(device=reference.device, dtype=reference.dtype)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeNciAssayModel/v1', axes=self.axes.binding(),
            width=self.width, base_width=self.base_width, readout_mult=self.readout_mult,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            classification='three literal source conclusion classes',
            numeric='independent Gaussian densities on complete native log10 source means; separate M/u partitions',
            generation='source connectivity UTF8 and EOS via exact shared byte decoder',
            semantics='source summaries and desired priors only; no clinical/stereo/ratio or cross-unit inference',
            decoder=self.decoder.get_extra_state(),
        )

    def get_extra_state(self):
        if (
            set(self._modules) != {'decoder', 'heads'} or self.decoder is not self._decoder_reference[0]
            or type(self.heads) is not nn.ModuleDict
            or tuple(self.heads) != tuple(partition.identity for partition in self.axes.partitions)
            or any(not _valid_head(self.heads[partition.identity], self.width, len(_readout_axis(partition)))
                   for partition in self.axes.partitions)
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Native NCI shared byte owner, source axis or readout policy drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact native NCI source/owner checkpoint configuration required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native NCI source checkpoint configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def condition(self, queries, *, direction):
        self.get_extra_state()
        queries = tuple(queries)
        if not 1 <= len(queries) <= self.max_batch_size:
            raise ValueError('Complete bounded native NCI known queries required')
        known = []
        for partition, condition in queries:
            if partition not in self.axes.partitions:
                raise ValueError('Native NCI query axis outside the bound assay partitions')
            body = nci_known_condition(partition, direction, condition)
            if len(body) > self.axes.max_condition_utf8_bytes:
                raise ValueError('Complete NCI condition exceeds its bound source bytes')
            known.append(body)
        shared = self.decoder.shared
        graphs = []
        for body in known:
            record = GraphRecord('known-nci-assay-query', (Node('nci_source_assay', (
                Measurement('condition', body.decode('utf-8'), None, 'observed', True),
                Measurement('requested_output', None, None, 'missing', False),
            )),), (), (Provenance('native-model-input', 'nci-source-v1', 'input',
                                  '0' * 64, 'source-admission-external'),))
            task = ConditionalTask(((('node', 0, 'condition'), 'condition'),
                                    (('node', 0, 'requested_output'), 'target')), topology_known=True)
            graphs.append(shared.payload.forward_task(record, task=task,
                                                      identifiers=shared.sequence_identifiers()))
        return shared.backbone(graphs)[0][:, 0]

    def _source_tasks(self, source, batch, split, direction):
        if type(source) is not NciCurrentAssaySource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact native NCI batch source and model assay axes required')
        if split in ('validation', 'test') and torch.is_grad_enabled():
            raise ValueError('Original NCI validation/test roles require evaluation-only use')
        return source.tasks(batch, split=split, direction=direction)

    def _validate_tasks(self, tasks, direction):
        tasks = tuple(tasks)
        if not 1 <= len(tasks) <= self.max_batch_size or any(
            type(task) is not NciCurrentAssayTask or task.direction != direction or task.partition not in self.axes.partitions
            for task in tasks
        ):
            raise ValueError('Complete bounded native NCI tasks in one exact direction required')
        return tasks

    def _prediction_objectives(self, tasks):
        tasks = self._validate_tasks(tasks, 'prediction')
        partitions = {}
        targets = []
        for i, task in enumerate(tasks):
            partition = task.partition
            if partition.family == CONCLUSION:
                if task.target not in partition.coordinate_axis:
                    raise ValueError('Literal native NCI conclusion target outside its class axis')
                targets.append(partition.coordinate_axis.index(task.target))
            else:
                if (
                    type(task.target) is not tuple or len(task.target) != len(partition.coordinate_axis)
                    or any(type(value) is not Decimal or not math.isfinite(value) for value in task.target)
                ):
                    raise ValueError('Every exact native NCI endpoint coordinate required')
                targets.append(tuple(float(value) for value in task.target))
            partitions.setdefault(partition, []).append(i)
        context = self.condition(((task.partition, task.inputs()) for task in tasks), direction='prediction')
        context = context * (self.readout_mult / (self.width / self.base_width))
        results = []
        for partition, indices in partitions.items():
            raw = self.heads[partition.identity](context[indices])
            if not bool(torch.isfinite(raw).all()):
                raise ValueError('Finite native NCI assay predictions required')
            values = [targets[i] for i in indices]
            if partition.family == CONCLUSION:
                labels = torch.tensor(values, dtype=torch.long, device=raw.device)
                results.append(NciPredictionResult(partition, tuple(indices), F.cross_entropy(raw, labels), logits=raw))
            else:
                mean, raw_scale = raw.double().chunk(2, dim=-1)
                scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).eps
                observed = mean.new_tensor(values)
                loss = -torch.distributions.Normal(mean, scale).log_prob(observed).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Finite native NCI source coordinate likelihood required')
                results.append(NciPredictionResult(partition, tuple(indices), loss, mean=mean, scale=scale))
        return tuple(results)

    def prediction_objectives(self, source, batch, *, split):
        """Preserve separate loss coordinates for different native assay/unit partitions."""
        return self._prediction_objectives(self._source_tasks(source, batch, split, 'prediction'))

    def _generation_objective(self, tasks, *, chunk_size, recompute):
        tasks = self._validate_tasks(tasks, 'generation')
        if any(type(task.target) is not bytes or not 1 <= len(task.target) <= self.axes.max_target_utf8_bytes
               for task in tasks):
            raise ValueError('Complete native NCI target connectivity bytes required')
        targets = validate_byte_targets(
            (task.target for task in tasks), max_batch_size=self.max_batch_size,
            max_total_target_bytes=self.max_total_target_bytes, chunk_size=chunk_size, recompute=recompute,
        )
        context = self.condition(((task.partition, task.inputs()) for task in tasks), direction='generation')
        return source_byte_objective(
            self.decoder, context, targets, chunk_size=chunk_size, recompute=recompute,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
        )

    def generation_objective(self, source, batch, *, split, chunk_size=1024, recompute=False):
        return self._generation_objective(
            self._source_tasks(source, batch, split, 'generation'), chunk_size=chunk_size, recompute=recompute,
        )

    @torch.no_grad()
    def generate(self, queries, *, max_new_tokens):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.axes.max_target_utf8_bytes + 1:
            raise ValueError('Explicit bounded NCI connectivity generation token budget required')
        queries = tuple(queries)
        context = self.condition(queries, direction='generation')
        result = self.decoder._generate_from_context(context, max_new_tokens=max_new_tokens)
        result['target_fields'] = (SMILES,) * len(queries)
        return result


def _valid_head(layer, width, channels):
    return (
        type(layer) is nn.Linear and not layer._modules and not layer._buffers
        and set(layer._parameters) == {'weight', 'bias'} and layer.bias is not None
        and (layer.in_features, layer.out_features) == (width, channels)
        and tuple(layer.weight.shape) == (channels, width) and tuple(layer.bias.shape) == (channels,)
    )


def nci_assay_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeNciAssayModel:
        raise ValueError('Exact native NCI assay owner required')
    model.get_extra_state()
    roles = {f'decoder.{name}': role for name, role in
             sequence_mup_parameter_roles(model.decoder, identifier_axis=identifier_axis).items()}
    hidden = Axis('width', 'hidden')
    for partition in model.axes.partitions:
        name = f'heads.{partition.identity}'
        raw = json.dumps(['native-nci-assay-readout-v1', partition.binding(), _readout_axis(partition)],
                         separators=(',', ':')).encode()
        finite = Axis('finite', hashlib.sha256(raw).hexdigest())
        roles[name + '.weight'] = ParameterRole('output', (finite, hidden))
        roles[name + '.bias'] = ParameterRole('vector', (finite,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def nci_assay_width_family(base, delta, target, *, identifier_axis):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, nci_assay_parameter_roles(model, identifier_axis=identifier_axis))
               for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if base.width != base.base_width or any(
        not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models
    ):
        raise ValueError('Native NCI assay family source or shared decoder policy drift')
    return width_family(*ledgers)
