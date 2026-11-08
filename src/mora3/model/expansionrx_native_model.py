"""Exact ExpansionRx native measurement likelihoods and source connectivity inverse."""

import copy
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.admet import (
    DesiredProperty, NativeCell, NativeEntity, admet_prediction_view, canonical, desired_admet_inputs,
)
from mora3.data.expansionrx_current import (
    ExpansionRxCurrentSource, ExpansionRxNativeTask, ExpansionRxPartition, ExpansionRxSourceAxes,
)
from .admet_model import SharedAdmetModel
from .admet_mup import admet_mup_parameter_roles
from .admet_objectives import AdmetObjectiveContract, gaussian_censored_nll
from .mixed_payload_projection import exact_wire
from .morphology_mup import _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger, width_family
from .pubchem_decoder import PubChemDecoder
from .sequence_mup import sequence_mup_parameter_roles
from .source_byte_objective import source_byte_objective, validate_byte_targets


@dataclass(frozen=True)
class ExpansionRxPredictionResult:
    partition: ExpansionRxPartition
    indices: tuple[int, ...]
    loss: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    censor_relations: tuple[str, ...]


def expansionrx_official_contracts(axes):
    """Reuse exact declared-unit SharedAdmetModel heads for eight source partitions."""
    if type(axes) is not ExpansionRxSourceAxes:
        raise ValueError('Exact complete native ExpansionRx axes required')
    axes.binding()
    return tuple(AdmetObjectiveContract(partition.definition, partition.context, 'small_molecule')
                 for partition in axes.partitions if not partition.nominal)


class ExpansionRxNativeModel(nn.Module):
    """Keep known units and nominal source coordinates in separate readouts."""

    def __init__(self, admet, decoder, axes, *, max_batch_size=128, max_total_target_bytes=None):
        super().__init__()
        if (
            type(admet) is not SharedAdmetModel or type(decoder) is not PubChemDecoder
            or type(axes) is not ExpansionRxSourceAxes or admet.shared is not decoder.shared
            or decoder.get_extra_state()['version'] != 2
        ):
            raise ValueError('Exact shared ADMET, byte decoder and current native ExpansionRx axes required')
        admet.validate_parameterization_configuration()
        if max_total_target_bytes is None and type(max_batch_size) is int:
            max_total_target_bytes = max_batch_size * axes.max_connectivity_bytes
        if (
            type(max_batch_size) is not int or max_batch_size < 1
            or type(max_total_target_bytes) is not int or max_total_target_bytes < axes.max_connectivity_bytes
        ):
            raise ValueError('Complete bounded native ExpansionRx tasks and connectivity byte resource required')
        contracts = expansionrx_official_contracts(axes)
        if any(admet.contracts.get(contract.partition_key) != contract for contract in contracts):
            raise ValueError('All eight actual official-unit ExpansionRx ADMET heads must be registered')
        self.admet = admet
        self.decoder = decoder
        self._owners = (admet, decoder)
        self.axes = axes
        self._registry = axes.registry()
        self._official = {contract.definition.key: contract for contract in contracts}
        self.max_batch_size = max_batch_size
        self.max_total_target_bytes = max_total_target_bytes
        config = admet.shared.get_extra_state()
        self.width = config['width']
        self.base_width = config['mup']['base_width']
        self.nominal_heads = nn.ModuleDict({
            partition.identity: nn.Linear(self.width, 2) for partition in axes.partitions if partition.nominal
        })
        with torch.no_grad():
            ratio = math.sqrt(self.width / self.base_width)
            for head in self.nominal_heads.values():
                head.weight.mul_(ratio)
                head.bias.mul_(ratio)
        reference = decoder.embedding.weight
        self.nominal_heads.to(device=reference.device, dtype=reference.dtype)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.ExpansionRxNativeModel/v1', axes=self.axes.binding(),
            admet=self.admet.get_extra_state(), decoder=self.decoder.get_extra_state(),
            width=self.width, base_width=self.base_width,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            regression='separate native source partition FP64 Gaussian equality/censor likelihood',
            nominal='LogD/Efflux source-reported numeric coordinates with explicitly unresolved units',
            inverse='original native SMILES UTF8+EOS and exact invariant non-representation entity fields',
        )

    def get_extra_state(self):
        if (
            set(self._modules) != {'admet', 'decoder', 'nominal_heads'}
            or self.admet is not self._owners[0] or self.decoder is not self._owners[1]
            or self.admet.shared is not self.decoder.shared or type(self.nominal_heads) is not nn.ModuleDict
            or self._registry != self.axes.registry()
            or self._official != {c.definition.key: c for c in expansionrx_official_contracts(self.axes)}
            or any(self.admet.contracts.get(c.partition_key) != c for c in self._official.values())
            or tuple(self.nominal_heads) != tuple(p.identity for p in self.axes.partitions if p.nominal)
            or any(not _valid_nominal_head(layer, self.width) for layer in self.nominal_heads.values())
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('Current ExpansionRx ADMET/byte aliases, source scope or readout policy drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Exact native ExpansionRx source checkpoint configuration required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Native ExpansionRx checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _source_tasks(self, source, batch, split, direction):
        if type(source) is not ExpansionRxCurrentSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact current source-bound ExpansionRx assay batch required')
        if split == 'test' and torch.is_grad_enabled():
            raise ValueError('Original ExpansionRx source test requires evaluation-only use')
        return source.tasks(batch, split=split, direction=direction)

    def _validate_tasks(self, tasks, direction):
        tasks = tuple(tasks)
        if not 1 <= len(tasks) <= self.max_batch_size or any(
            type(task) is not ExpansionRxNativeTask or task.direction != direction
            or task.partition not in self.axes.partitions for task in tasks
        ):
            raise ValueError('Complete bounded ExpansionRx native source tasks in one direction required')
        return tasks

    def _official_view_and_contract(self, task):
        view = admet_prediction_view(task.observation, self._registry)
        contract = self._official[task.partition.definition.key]
        self.admet._validate((view,), (contract,))
        return view, contract

    def _prediction_objectives(self, tasks):
        self.get_extra_state()
        tasks = self._validate_tasks(tasks, 'entity_to_measurement')
        views, coordinates, groups = [], [], {}
        for i, task in enumerate(tasks):
            partition = task.partition
            view = (admet_prediction_view(task.observation, self._registry) if partition.nominal
                    else self._official_view_and_contract(task)[0])
            if not task.observation.experimental_measurement_candidate:
                raise ValueError('Source-observed native ExpansionRx measurement or bound required')
            views.append(view)
            coordinates.append(_native_coordinate(task.observation.cell))
            groups.setdefault(partition, []).append(i)
        context = self.admet._readout_features(self.admet.condition(view.inputs() for view in views))
        results = []
        for partition, indices in groups.items():
            contract = self._official.get(partition.definition.key)
            layer = (self.nominal_heads[partition.identity] if partition.nominal
                     else self.admet.heads[contract.partition_key])
            mean, raw_scale = layer(context[indices]).double().unbind(-1)
            scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).eps
            native = [coordinates[i] for i in indices]
            active = torch.ones_like(mean, dtype=torch.bool)
            native_args = dict(
                relation=tuple(row[0] for row in native), value=mean.new_tensor([row[1] for row in native]),
                lower=mean.new_tensor([row[2] for row in native]), upper=mean.new_tensor([row[3] for row in native]),
                observed=active, applicable=active, applicability_known=active,
            )
            loss = (gaussian_censored_nll(mean, scale, **native_args) if partition.nominal
                    else contract.censored_loss(mean, scale, native_unit=contract.unit, **native_args))
            results.append(ExpansionRxPredictionResult(
                partition, tuple(indices), loss, mean, scale,
                tuple(tasks[i].observation.cell.relation for i in indices),
            ))
        return tuple(results)

    def prediction_objectives(self, source, batch, *, split):
        return self._prediction_objectives(self._source_tasks(source, batch, split, 'entity_to_measurement'))

    def inverse_condition(self, queries):
        self.get_extra_state()
        queries = tuple(queries)
        if not 1 <= len(queries) <= self.max_batch_size:
            raise ValueError('Complete bounded native ExpansionRx inverse queries required')
        inputs = []
        for partition, cell in queries:
            if type(partition) is not ExpansionRxPartition or partition not in self.axes.partitions or type(cell) is not NativeCell:
                raise ValueError('Exact bound native ExpansionRx endpoint and desired measurement required')
            _native_coordinate(cell)
            request = DesiredProperty(partition.definition.key, cell, partition.context, b'{}')
            inputs.append(desired_admet_inputs('small_molecule', (request,), self._registry))
        return self.admet.condition(inputs)

    def _generation_objective(self, tasks, *, chunk_size, recompute):
        tasks = self._validate_tasks(tasks, 'measurement_to_entity')
        raw = tuple(task.native_connectivity_bytes() for task in tasks)
        if any(not 1 <= len(value) <= self.axes.max_connectivity_bytes for value in raw):
            raise ValueError('Every complete native ExpansionRx target connectivity byte required')
        targets = validate_byte_targets(
            raw, max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            chunk_size=chunk_size, recompute=recompute,
        )
        context = self.inverse_condition((task.partition, task.observation.cell) for task in tasks)
        return source_byte_objective(
            self.decoder, context, targets, chunk_size=chunk_size, recompute=recompute,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
        )

    def generation_objective(self, source, batch, *, split, chunk_size=1024, recompute=False):
        return self._generation_objective(
            self._source_tasks(source, batch, split, 'measurement_to_entity'), chunk_size=chunk_size, recompute=recompute,
        )

    @torch.no_grad()
    def generate(self, queries, *, max_new_tokens):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.axes.max_connectivity_bytes + 1:
            raise ValueError('Explicit bounded native ExpansionRx connectivity token budget required')
        context = self.inverse_condition(queries)
        result = self.decoder._generate_from_context(context, max_new_tokens=max_new_tokens)
        template = json.loads(self.axes.native_entity_template)
        result['native_entities'] = tuple(
            NativeEntity.from_mapping(dict(template, representation=text)).as_mapping() if done and text else None
            for done, text in zip(result['terminated'], result['text'])
        )
        return result


def _native_coordinate(cell):
    if type(cell) is not NativeCell or cell.numeric_loss_eligible is not True:
        raise ValueError('Complete native source equality or censor-bound literal required')
    kind = cell.censor_kind
    if kind == 'interval':
        lower = _source_float(Decimal(cell.lower_decimal_literal))
        upper = _source_float(Decimal(cell.upper_decimal_literal))
        if not lower < upper:
            raise ValueError('Native ExpansionRx interval collapsed at FP64 precision')
        return kind, 0., lower, upper
    if kind not in ('equal', 'left', 'right'):
        raise ValueError('Exact native ExpansionRx equality/left/right/interval required')
    value = _source_float(cell.decimal_value())
    return kind, value, value, value


def _source_float(value):
    result = float(value)
    if not math.isfinite(result) or value != 0 and result == 0:
        raise ValueError('Native ExpansionRx scalar cannot be represented at FP64 precision')
    return result


def _valid_nominal_head(layer, width):
    return (
        type(layer) is nn.Linear and not layer._modules and not layer._buffers
        and set(layer._parameters) == {'weight', 'bias'} and layer.bias is not None
        and (layer.in_features, layer.out_features) == (width, 2)
        and tuple(layer.weight.shape) == (2, width) and tuple(layer.bias.shape) == (2,)
    )


def expansionrx_parameter_roles(model, *, identifier_axis):
    if type(model) is not ExpansionRxNativeModel:
        raise ValueError('Exact current source ExpansionRx shared owner required')
    model.get_extra_state()
    roles = {
        **{'admet.' + n: r for n, r in admet_mup_parameter_roles(model.admet, identifier_axis=identifier_axis).items()},
        **{'decoder.' + n: r for n, r in sequence_mup_parameter_roles(model.decoder, identifier_axis=identifier_axis).items()},
    }
    hidden = Axis('width', 'hidden')
    for partition in model.axes.partitions:
        if partition.nominal:
            name = 'nominal_heads.' + partition.identity
            raw = canonical(['ExpansionRx-nominal-source-coordinate-v1', partition.binding(), ['mean', 'raw_scale']])
            finite = Axis('finite', hashlib.sha256(raw).hexdigest())
            roles[name + '.weight'] = ParameterRole('output', (finite, hidden))
            roles[name + '.bias'] = ParameterRole('vector', (finite,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def expansionrx_width_family(base, delta, target, *, identifier_axis):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, expansionrx_parameter_roles(model, identifier_axis=identifier_axis))
               for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if base.width != base.base_width or any(
        not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models
    ):
        raise ValueError('Native ExpansionRx source/official/nominal family binding drift')
    return width_family(*ledgers)
