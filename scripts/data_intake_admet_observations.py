"""Lossless source-bound ADMET observations and explicitly gated task projections.

This module never reviews a source license, assigns a split, converts a unit or
canonicalizes chemistry. Those decisions require independent corpus evidence.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re

from data_intake_corpus import CorpusTask, canonical

CLASSES = frozenset(('small_molecule', 'peptide', 'protein', 'RNA', 'DNA'))
_CONTEXT = frozenset(('species', 'strain', 'cell_line', 'tissue', 'matrix',
    'temperature', 'pH', 'exposure_time', 'dose', 'route', 'formulation',
    'carrier', 'assay_system', 'assay_direction', 'source_measurement_coordinate'))
_IDENTITY = frozenset(('molecule_class', 'representation_type', 'representation',
    'native_modifications', 'construct', 'source_identity_missingness'))
_ENDPOINT = frozenset(('endpoint_key', 'source_endpoint', 'description', 'readout',
    'unit', 'unit_basis', 'scale', 'measurement_type', 'assay_protocol_context'))
_GATES = ('source_rights', 'exact_identity', 'endpoint_definition',
          'task_units_or_declared_native_coordinate', 'split_group_review')
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_CELL = re.compile(r'^\s*(<=|>=|<|>|=|≤|≥)?\s*(' + _NUMBER + r')\s*$')
_RELATION = {'≤': '<=', '≥': '>='}


def native_cell(literal, *, missing_literals=('',)):
    """Preserve a CSV literal; decimal/censor parsing adds no implicit units."""
    if literal is not None and not isinstance(literal, str):
        raise TypeError('native CSV cell must be a literal string or explicit null')
    if literal is None or literal in missing_literals:
        return dict(native_literal=literal, status='missing', decimal_literal=None,
                    relation=None, observation_present=False, numeric_loss_eligible=False)
    match = _CELL.fullmatch(literal)
    if match:
        relation = _RELATION.get(match[1], match[1]) or '='
        try:
            number = Decimal(match[2])
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite():
            return dict(native_literal=literal,
                status='exact' if relation == '=' else 'censored_bound',
                decimal_literal=match[2], relation=relation,
                observation_present=True, numeric_loss_eligible=True)
    return dict(native_literal=literal, status='unparsed', decimal_literal=None,
                relation=None, observation_present=True, numeric_loss_eligible=False)


def observation(*, entity, endpoint, cell, known_context, source_ref,
                native_source_record, evidence_role, source_split, admission):
    """No unknown native field is dropped: the complete record stays in audit."""
    if set(entity) - _IDENTITY or entity.get('molecule_class') not in CLASSES:
        raise ValueError('explicit molecule class and allowlisted native identity required')
    if not isinstance(entity.get('representation'), str) or not entity['representation']:
        raise ValueError('nonempty exact source identity representation required')
    if not isinstance(entity.get('representation_type'), str):
        raise ValueError('source-native representation type is required')
    if set(endpoint) - _ENDPOINT or not endpoint.get('source_endpoint'):
        raise ValueError('allowlisted literal endpoint definition required')
    if endpoint.get('unit_basis') not in ('source_literal', 'official_source_schema', 'unresolved'):
        raise ValueError('literal, official-schema and unresolved units stay distinct')
    if endpoint.get('unit_basis') == 'unresolved' and endpoint.get('unit') is not None:
        raise ValueError('unresolved physical unit cannot be silently supplied')
    if set(known_context) - _CONTEXT:
        raise ValueError('source identifiers, names and labels are not assay predictors')
    if evidence_role not in ('experimental_measurement', 'source_curated_measurement',
                             'computed', 'predicted', 'model_generated', 'desired'):
        raise ValueError('an explicit evidence role is required')
    if native_cell(cell['native_literal'], missing_literals=('',)) != cell:
        # A source-specific missing lexicon may be used, but must be auditable.
        if cell['status'] != 'missing' or cell.get('missing_lexicon_source') is None:
            raise ValueError('cell must preserve native parser semantics or explicit missing lexicon')
    result = dict(schema='mora3.ADMETNativeObservation/v1', entity=deepcopy(entity),
        endpoint=deepcopy(endpoint), cell=deepcopy(cell), known_context=deepcopy(known_context),
        source_ref=deepcopy(source_ref), native_source_record=deepcopy(native_source_record),
        evidence_role=evidence_role, source_split=deepcopy(source_split),
        admission=deepcopy(admission), absence_is_not_negative=True,
        native_fields_not_projected_remain_in_source_record=True,
        inferred_unit_conversions=[], source_native_axis_cell_multiplicity=1,
        independent_experimental_measurement_count=None)
    canonical(result)  # Reject non-JSON/NaN values, rather than coerce source data.
    return result


def task_view(record, *, direction):
    """Project only after task-specific admission; this is not model support."""
    if record.get('schema') != 'mora3.ADMETNativeObservation/v1':
        raise ValueError('source-native observation is required')
    a = record['admission']
    if not all(a.get(g) is True for g in _GATES):
        raise ValueError('all task-specific source/identity/endpoint/units/split gates required')
    cell = record['cell']
    if not cell['observation_present'] or not cell['numeric_loss_eligible']:
        raise ValueError('missing or unparsed source cell is not an observed numeric target')
    if record['evidence_role'] not in ('experimental_measurement', 'source_curated_measurement'):
        if a.get('explicit_nonexperimental_objective') is not True:
            raise ValueError('computed/predicted labels require a separate explicit objective')
    entity = deepcopy(record['entity'])
    context = dict(endpoint=deepcopy(record['endpoint']),
                   assay_context=deepcopy(record['known_context']),
                   evidence_role=record['evidence_role'])
    measurement = {k: deepcopy(cell[k]) for k in
        ('native_literal', 'status', 'decimal_literal', 'relation')}
    if direction == 'entity_to_measurement':
        conditions = dict(native_entity=entity, **context)
        targets = dict(native_measurement=measurement)
    elif direction == 'measurement_to_entity':
        conditions = dict(desired_native_measurement=measurement,
                          molecule_class=entity['molecule_class'], **context)
        targets = dict(native_entity=entity)
    else:
        raise ValueError('explicit ADMET direction required')
    return CorpusTask('source_native_ADMET_' + direction, conditions, targets,
        dict(source_observation=deepcopy(record), native_model_adapter_supported=False,
             directions_reuse_one_source_observation=True,
             evidence_is_not_clinical_or_causal_validation=True,
             excluded_predictors=['source/entity/study/assay IDs', 'names/citations/audit',
                 'hidden target native values/identity/length/hashes',
                 'forward target availability mask and unrequested endpoints']),
        [k for k in sorted(_CONTEXT) if k not in record['known_context']]
        + (['physical_unit'] if record['endpoint']['unit_basis'] == 'unresolved' else []))
