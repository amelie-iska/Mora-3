"""Typed original UniProt annotation directions for the current train-only view."""
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .finite_union_native_rows import FiniteUnionNativeView


UNIPROT_NATIVE_INVERSE_TASK = 'source_UniProt_annotation_conditioned_canonical_sequence'
UNIPROT_NATIVE_FORWARD_TASK = 'sequence_to_curated_function_annotations'
UNIPROT_FORWARD_TARGET_FIELDS = (
    'source_curated_function', 'source_curated_catalytic_activity',
    'source_protein_name', 'source_GO_identifiers',
)
_FORWARD_SOURCE_FIELDS = {
    'source_curated_function': 'Function [CC]',
    'source_curated_catalytic_activity': 'Catalytic activity',
    'source_protein_name': 'Protein names',
}
_INVERSE_FIELDS = frozenset({
    'organism_taxon_id', 'source_experimentally_supported_function_texts',
    'source_experimental_associated_GO_ids', 'source_explicit_experimental_NOT_GO_ids',
    'source_experimentally_supported_EC_ids', 'source_kinetic_conditions',
})
_AA_SYMBOLS = frozenset('ABCDEFGHIKLMNOPQRSTUVWXYZ')
_GO = re.compile(r'GO:[0-9]{7}')
_EC = re.compile(r'[0-9]+(?:\.(?:[0-9]+|-)){3}|[0-9]+(?:\.[0-9]+){2}\.n[0-9]+')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_FORWARD_EXPORT_SHA256 = '74ba74f0adeeecc5415b514118967fc521af3a26f30d04d512fea75099d4bd64'
_EFFECTIVE_V2_SHA256 = '5d60c917c5592959ea7ca909e8ddfb9cbd06221aa13af8ddd2c754c103fbae3e'
_EFFECTIVE_V2_EVIDENCE_SHA256 = 'f768c6edb7610793320fe6430d66ce2999548c20362c45bb8108c2a5945086f3'
_INVERSE_GUARD_PROPOSAL_SHA256 = 'c9ef860d0695222c03fce33568f29e243c2efbb19d4c80d3c215f4b5496f685d'
_INVERSE_GUARD_TASK = '33438b8022abc2069790c67081d1bfd2a93d448cae39d3678566106e27c597d4'
_INVERSE_GUARD_SHARD = ('outputs/data-intake/20261031-uniprot-full-annotations-v1/'
                        'admitted-corpus-v4/shards/train/02530.parquet')
_INVERSE_GUARD_ROW = 25


@dataclass(frozen=True)
class UniProtAnnotationOrigin:
    source_shard_path: str
    source_shard_sha256: str
    source_row_ordinal: int
    task_id: str
    accession: str
    source_revision: str
    native_payload_sha256: str


@dataclass(frozen=True)
class UniProtNativeKineticCondition:
    parameter_type: str
    source_value: float
    source_unit: str
    substrate_label: str | None

    def model_fields(self):
        return {
            'parameter_type': self.parameter_type,
            'source_value': self.source_value,
            'source_unit': self.source_unit,
            'substrate_label': self.substrate_label,
            'assay_pH': None,
            'assay_temperature': None,
        }


@dataclass(frozen=True)
class UniProtInverseAnnotationRow:
    """A canonical sequence target and its original allowed evidence fields."""
    organism_taxon_id: int
    source_experimentally_supported_function_texts: tuple[str, ...]
    source_experimental_associated_GO_ids: tuple[str, ...]
    source_explicit_experimental_NOT_GO_ids: tuple[str, ...]
    source_experimentally_supported_EC_ids: tuple[str, ...]
    source_kinetic_conditions: tuple[UniProtNativeKineticCondition, ...]
    canonical_sequence: str
    origin: UniProtAnnotationOrigin

    def model_condition_fields(self):
        return {
            'organism_taxon_id': self.organism_taxon_id,
            'source_experimentally_supported_function_texts': self.source_experimentally_supported_function_texts,
            'source_experimental_associated_GO_ids': self.source_experimental_associated_GO_ids,
            'source_explicit_experimental_NOT_GO_ids': self.source_explicit_experimental_NOT_GO_ids,
            'source_experimentally_supported_EC_ids': self.source_experimentally_supported_EC_ids,
            'source_kinetic_conditions': tuple(k.model_fields() for k in self.source_kinetic_conditions),
        }


@dataclass(frozen=True)
class UniProtForwardFunctionRow:
    """One native input and separate observed annotation target fields."""
    amino_acid_sequence: str
    organism_taxon: int
    source_curated_function: str | None
    source_curated_catalytic_activity: str | None
    source_protein_name: str | None
    source_GO_identifiers: tuple[str, ...] | None
    origin: UniProtAnnotationOrigin

    def model_condition_fields(self):
        return {'amino_acid_sequence': self.amino_acid_sequence, 'organism_taxon': self.organism_taxon}

    def target_bytes(self, field):
        """Return literal UTF-8 or an ordered fixed-width 10-byte GO ID axis."""
        if field not in UNIPROT_FORWARD_TARGET_FIELDS:
            raise ValueError('unknown native UniProt forward target field')
        value = getattr(self, field)
        if value is None:
            return None
        if field == 'source_GO_identifiers':
            return ''.join(value).encode('ascii')
        return value.encode('utf-8')


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_taxon(value):
    if type(value) is not int or value <= 0:
        raise ValueError('positive original UniProt taxon integer required')
    return value


def _sequence(value):
    if type(value) is not str or not value or not _AA_SYMBOLS.issuperset(value):
        raise ValueError('complete literal UniProt amino-acid sequence required')
    return value


def _native_text(value):
    if type(value) is not str or not value:
        raise ValueError('nonempty native UniProt text required')
    value.encode('utf-8')
    return value


def _native_list(value, pattern=None):
    if type(value) is not list:
        raise ValueError('ordered original UniProt annotation list required')
    values = tuple(_native_text(item) for item in value)
    if pattern is not None and any(pattern.fullmatch(item) is None for item in values):
        raise ValueError('unsupported original UniProt identifier spelling')
    return values


def _kinetic(value):
    keys = {'parameter_type', 'source_value', 'source_unit', 'substrate_label', 'assay_pH', 'assay_temperature'}
    if type(value) is not dict or set(value) != keys:
        raise ValueError('explicit original UniProt kinetic fields required')
    parameter, number = value['parameter_type'], value['source_value']
    if (parameter not in ('MichaelisConstant', 'maximumVelocity') or type(number) is not float
            or not math.isfinite(number) or number < 0
            or value['assay_pH'] is not None or value['assay_temperature'] is not None):
        raise ValueError('unsupported original UniProt kinetic value or assay context')
    substrate = value['substrate_label']
    if parameter == 'MichaelisConstant':
        if type(substrate) is not str:
            raise ValueError('original MichaelisConstant substrate string required')
        substrate.encode('utf-8')
    elif substrate is not None:
        raise ValueError('original maximumVelocity substrate is null in this release view')
    return UniProtNativeKineticCondition(parameter, number, _native_text(value['source_unit']), substrate)


def _read_envelope(row, family, direction):
    raw = row['native_payload']
    if (type(raw) is not bytes or _sha(raw) != row['native_payload_sha256']
            or row['task_name'] != direction or row['source_family'] != family
            or row['native_schema'] != 'mora3.CorpusTask/v1' or row['split'] != 'train'
            or row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False):
        raise ValueError('exact original UniProt source envelope/role required')
    task = json.loads(raw)
    if (type(task) is not dict or set(task) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
            or task['schema'] != row['native_schema'] or task['task'] != direction or raw != _canonical(task)
            or any(type(task[k]) is not dict for k in ('conditions', 'targets', 'audit'))
            or type(task['missing']) is not list or json.loads(row['missing_context_json']) != task['missing']):
        raise ValueError('exact native UniProt task schema/payload required')
    receipt = json.loads(row['admission_receipt_json'])
    if receipt.get('reviewed') is not True or receipt.get('annotation_score') not in (4, 5):
        raise ValueError('exact reviewed 2026_03 source annotation scope required')
    return task, receipt


def _source_origin(row, shard, ordinal):
    return UniProtAnnotationOrigin(shard['path'], shard['sha256'], ordinal, row['task_id'],
                                   row['record_id'], row['source_revision'], row['native_payload_sha256'])


def _inverse_row(row, shard, ordinal):
    if (row['task_id'] == _INVERSE_GUARD_TASK
            or (shard['path'], ordinal) == (_INVERSE_GUARD_SHARD, _INVERSE_GUARD_ROW)):
        raise ValueError('exact original UniProt inverse literal-target-fragment exclusion')
    task, receipt = _read_envelope(row, 'UniProt-full-record-annotations', UNIPROT_NATIVE_INVERSE_TASK)
    conditions, targets = task['conditions'], task['targets']
    if set(conditions) != _INVERSE_FIELDS or set(targets) != {'canonical_sequence'}:
        raise ValueError('only exact native inverse evidence fields and sequence target required')
    sequence = _sequence(targets['canonical_sequence'])
    ref, identity = task['audit']['source_ref'], task['audit']['identity']
    entry_version, sequence_version = ref['entry_version'], ref['sequence_version']
    expected_revision = f'UniProt2026_03;entryVersion{entry_version};sequenceVersion{sequence_version}'
    if (receipt.get('source_ref') != ref or receipt.get('license') != 'CC-BY-4.0'
            or receipt.get('scope_safe_generation_v2') is not True
            or receipt.get('canonical_sequence_exact_baseline') is not True
            or ref['source_release'] != '2026_03' or ref['raw_file_sha256'] != row['source_sha256']
            or _SHA256.fullmatch(row['source_sha256']) is None or row['source_revision'] != expected_revision
            or any(type(v) is not int or v <= 0 for v in (entry_version, sequence_version))
            or identity['primary_accession'] != row['record_id'] or identity['sequence_sha256'] != _sha(sequence.encode('ascii'))
            or identity['sequence_length'] != len(sequence) or row['full_axis_values'] != len(sequence)
            or identity['entry_version'] != entry_version or identity['sequence_version'] != sequence_version
            or identity['annotation_score'] != receipt['annotation_score']):
        raise ValueError('exact canonical UniProt inverse source/revision proof differs')
    functions = _native_list(conditions['source_experimentally_supported_function_texts'])
    associated = _native_list(conditions['source_experimental_associated_GO_ids'], _GO)
    explicit_not = _native_list(conditions['source_explicit_experimental_NOT_GO_ids'], _GO)
    ec = _native_list(conditions['source_experimentally_supported_EC_ids'], _EC)
    values = conditions['source_kinetic_conditions']
    if type(values) is not list or not any((functions, associated, explicit_not, ec, values)):
        raise ValueError('nonempty observed native inverse evidence required')
    return UniProtInverseAnnotationRow(_source_taxon(conditions['organism_taxon_id']), functions, associated,
        explicit_not, ec, tuple(_kinetic(item) for item in values), sequence, _source_origin(row, shard, ordinal))


def _forward_row(row, shard, ordinal):
    task, receipt = _read_envelope(row, 'UniProt-function-annotations', UNIPROT_NATIVE_FORWARD_TASK)
    conditions, targets = task['conditions'], task['targets']
    if (set(conditions) != {'amino_acid_sequence', 'organism_taxon'}
            or not targets or not set(targets) <= set(UNIPROT_FORWARD_TARGET_FIELDS)
            or not {'source_curated_function', 'source_curated_catalytic_activity', 'source_GO_identifiers'} & set(targets)):
        raise ValueError('exact original forward sequence/context and observed fields required')
    sequence = _sequence(conditions['amino_acid_sequence'])
    taxon = _source_taxon(conditions['organism_taxon'])
    native = task['audit']['original_source_row']
    if (row['source_revision'] != 'UniProt2026_03' or row['source_sha256'] != _FORWARD_EXPORT_SHA256
            or receipt.get('release') != '2026_03' or receipt.get('exact_old_sequence_match') is not True
            or receipt.get('source_only_annotation_scope') is not True
            or native['Reviewed'] != 'reviewed' or native['Entry'] != row['record_id']
            or native['Sequence'] != sequence or int(native['Organism (ID)']) != taxon
            or row['full_axis_values'] != len(sequence)):
        raise ValueError('exact original UniProt forward TSV source/revision proof differs')
    literal = {key: _native_text(native[source]) for key, source in _FORWARD_SOURCE_FIELDS.items()
               if native[source].strip()}
    go = [item.strip() for item in native['Gene Ontology IDs'].split(';') if item.strip()]
    if go:
        literal['source_GO_identifiers'] = go
    if targets != literal:
        raise ValueError('complete original UniProt forward field bytes/order required')
    return UniProtForwardFunctionRow(sequence, taxon, literal.get('source_curated_function'),
        literal.get('source_curated_catalytic_activity'), literal.get('source_protein_name'),
        _native_list(go, _GO) if go else None, _source_origin(row, shard, ordinal))


def _require_current_view(view, batch, split):
    if type(view) is not FiniteUnionNativeView or split != 'train':
        raise ValueError('exact current finite native view and original train role required')
    effective = getattr(view, 'effective_filter', {})
    if type(effective) is not dict or effective.get('source_role') != 'train':
        raise ValueError('current reviewed UniProt native source filter required')
    schema = effective.get('schema')
    if schema == 'Mora3.AdditionalFiniteNativeStudyOrdinalFilter/v1':
        expected = 227
        evidence = effective.get('source_task_consumption_manifest', {}).get('sha256')
        expected_evidence = _EFFECTIVE_V2_EVIDENCE_SHA256
    elif schema == 'Mora3.FiniteNativeInverseLiteralTargetOrdinalFilter/v1':
        expected = 228
        evidence = effective.get('parent_effective_filter', {}).get('sha256')
        expected_evidence = _EFFECTIVE_V2_SHA256
        guard = effective.get('source_inverse_literal_guard_proposal', {})
        if guard.get('sha256') != _INVERSE_GUARD_PROPOSAL_SHA256:
            raise ValueError('exact original UniProt inverse exclusion evidence required')
    else:
        raise ValueError('current reviewed UniProt native source filter required')
    if (evidence != expected_evidence or effective.get('combined_excluded_taskviews') != expected
            or len(view._ordinal_exclusions) != expected):
        raise ValueError('exact existing source exclusions required')
    view.require_native_rows(batch, split=split)


def _rows(view, batch, split, projector):
    _require_current_view(view, batch, split)
    try:
        return tuple(projector(row, batch.source_shard, ordinal)
            for ordinal, row in zip(batch.source_row_ordinals, batch.native_rows.to_pylist(), strict=True))
    except (AttributeError, KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('malformed original UniProt annotation row') from exc


def uniprot_native_inverse_rows(view, batch, *, split='train'):
    """Authenticate original generation rows, refusing the known fragment witness."""
    return _rows(view, batch, split, _inverse_row)


def uniprot_native_forward_rows(view, batch, *, split='train'):
    """Authenticate original forward rows and expose only their typed native fields."""
    return _rows(view, batch, split, _forward_row)
