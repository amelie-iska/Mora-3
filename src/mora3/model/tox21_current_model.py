"""Original Tox21 forward calls and inverse molecules with shared native owners."""

import copy

import torch
from torch import nn

from mora3.data.tox21 import FORWARD, INVERSE, Tox21View
from mora3.data.tox21_current import Tox21CurrentAxes, Tox21CurrentSource, Tox21DesiredCallPattern
from .assay_mup import assay_mup_parameter_roles
from .mixed_payload_projection import exact_wire
from .morphology_mup import _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import parameter_ledger, width_family
from .pubchem_decoder import PubChemDecoder
from .sequence_mup import sequence_mup_parameter_roles
from .source_byte_objective import source_byte_objective, validate_byte_targets
from .tox21 import SharedTox21Model


class NativeTox21SourceModel(nn.Module):
    """Share observed-call masks, numeric input projection, and full byte recurrence."""

    def __init__(self, tox21, decoder, axes, *, max_batch_size=128, max_total_target_bytes=None):
        super().__init__()
        if (
            type(tox21) is not SharedTox21Model or type(decoder) is not PubChemDecoder
            or type(axes) is not Tox21CurrentAxes or tox21.shared is not decoder.shared
            or decoder.get_extra_state()['version'] != 2
        ):
            raise ValueError('Exact shared native Tox21 and byte decoder owners required')
        tox21.validate_parameterization_configuration()
        axes.binding()
        if max_total_target_bytes is None and type(max_batch_size) is int:
            max_total_target_bytes = max_batch_size * axes.max_full_smiles_bytes
        if (
            type(max_batch_size) is not int or max_batch_size < 1
            or type(max_total_target_bytes) is not int or max_total_target_bytes < axes.max_full_smiles_bytes
        ):
            raise ValueError('Complete bounded native Tox21 batch and full molecule byte resource required')
        self.tox21 = tox21
        self.decoder = decoder
        self._owners = (tox21, decoder)
        self.axes = axes
        self.max_batch_size = max_batch_size
        self.max_total_target_bytes = max_total_target_bytes
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.NativeTox21CurrentSourceModel/v1', axes=self.axes.binding(),
            tox21=self.tox21.get_extra_state(), decoder=self.decoder.get_extra_state(),
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            forward='existing observed-call mask BCE; no unknown negatives',
            inverse='existing full 12-call value+mask input and complete native source SMILES+EOS',
            semantics='observed source call priors; no clinical safety or activity guarantee',
        )

    def get_extra_state(self):
        if (
            set(self._modules) != {'tox21', 'decoder'} or self._parameters or self._buffers
            or self.tox21 is not self._owners[0] or self.decoder is not self._owners[1]
            or self.tox21.shared is not self.decoder.shared
            or not exact_wire(self._configuration, self._current_configuration())
        ):
            raise ValueError('Exact Tox21 source/model/call-mask owner binding drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('Current source-bound Tox21 checkpoint configuration differs')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('Current source-bound Tox21 checkpoint configuration differs')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _views(self, source, batch, split, direction):
        self.get_extra_state()
        if type(source) is not Tox21CurrentSource or not exact_wire(source.axes.binding(), self.axes.binding()):
            raise ValueError('Exact bound native Tox21 source task batch required')
        if split in ('validation', 'test') and torch.is_grad_enabled():
            raise ValueError('Original Tox21 validation and test roles are evaluation only')
        views = source.views(batch, split=split, direction=direction)
        if len(views) > self.max_batch_size:
            raise ValueError('Native Tox21 source batch exceeds configured bound')
        return views

    def prediction_objective(self, source, batch, *, split, return_logits=False):
        if type(return_logits) is not bool:
            raise ValueError('Explicit Tox21 reporting policy required')
        views = self._views(source, batch, split, FORWARD)
        logits = self.tox21(views)
        loss = self.tox21.objective(logits, views)
        return dict(loss=loss, logits=logits) if return_logits else loss

    def inverse_condition(self, patterns):
        self.get_extra_state()
        patterns = tuple(patterns)
        if not 1 <= len(patterns) <= self.max_batch_size or any(type(p) is not Tox21DesiredCallPattern for p in patterns):
            raise ValueError('Complete bounded exact native Tox21 desired call patterns required')
        return self.tox21._condition_inputs(tuple((INVERSE, pattern.inputs()) for pattern in patterns))

    def _reference_inverse_objective(self, views, *, chunk_size, recompute):
        views = tuple(views)
        if not 1 <= len(views) <= self.max_batch_size or any(type(v) is not Tox21View or v.direction != INVERSE for v in views):
            raise ValueError('Complete bounded native Tox21 inverse source views required')
        patterns, targets = [], []
        for view in views:
            pattern = Tox21DesiredCallPattern.from_inputs(view.inputs())
            if pattern.values != view.values or pattern.observed != view.observed_mask:
                raise ValueError('Tox21 original desired calls and observation mask differ')
            target = view.targets()
            if set(target) != {'canonical_isomeric_smiles'} or type(target['canonical_isomeric_smiles']) is not str:
                raise ValueError('Exact original native Tox21 connectivity string required')
            raw = target['canonical_isomeric_smiles'].encode('utf-8')
            if not 1 <= len(raw) <= self.axes.max_full_smiles_bytes:
                raise ValueError('Complete original native Tox21 molecule outside source byte bounds')
            patterns.append(pattern)
            targets.append(raw)
        targets = validate_byte_targets(
            targets, max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            chunk_size=chunk_size, recompute=recompute,
        )
        context = self.inverse_condition(patterns)
        return source_byte_objective(
            self.decoder, context, targets, chunk_size=chunk_size, recompute=recompute,
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
        )

    def generation_objective(self, source, batch, *, split, chunk_size=1024, recompute=False):
        views = self._views(source, batch, split, INVERSE)
        return self._reference_inverse_objective(views, chunk_size=chunk_size, recompute=recompute)

    @torch.no_grad()
    def generate(self, patterns, *, max_new_tokens):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.axes.max_full_smiles_bytes + 1:
            raise ValueError('Explicit full native Tox21 generation budget required')
        patterns = tuple(patterns)
        result = self.decoder._generate_from_context(self.inverse_condition(patterns), max_new_tokens=max_new_tokens)
        result['target_fields'] = ('canonical_isomeric_smiles',) * len(patterns)
        return result


def tox21_current_model_parameter_roles(model, *, identifier_axis):
    if type(model) is not NativeTox21SourceModel:
        raise ValueError('Exact current native Tox21 shared-owner scope required')
    model.get_extra_state()
    roles = {
        **{'tox21.' + n: r for n, r in assay_mup_parameter_roles(model.tox21, identifier_axis=identifier_axis).items()},
        **{'decoder.' + n: r for n, r in sequence_mup_parameter_roles(model.decoder, identifier_axis=identifier_axis).items()},
    }
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def tox21_current_model_width_family(base, delta, target, *, identifier_axis):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, tox21_current_model_parameter_roles(model, identifier_axis=identifier_axis))
               for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if any(not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models):
        raise ValueError('Current Tox21 source/owner width family binding drift')
    if base.decoder.rnn.hidden_size != base.decoder.get_extra_state()['mup']['base_width']:
        raise ValueError('Actual Tox21 base member width differs from declared base')
    return width_family(*ledgers)
