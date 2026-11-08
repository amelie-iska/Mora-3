"""Full native-string priors authenticated against a finite original-row view.

These mappings describe source task semantics, not new scientific admission.
Every original field is re-read by ``require_native_rows`` before extraction.
Native strings are UTF-8 encoded without normalization, alphabet repair, crop,
chemical standardization or a numeric/JSON fallback. Historical model-support
flags and all source identifiers remain unchanged in the audit sidecar.
"""
from dataclasses import dataclass
import hashlib
import json

from .finite_union_native_rows import FiniteUnionNativeView


_NUCLEOTIDE_TASK = 'source_native_nucleotide_sequence_prior'
_PUBCHEM_REVISION = 'f83219601635a0a80fc99c13a9ca37f99ef34f0a'
# family, native task -> route, target datatype, target field, exact revision,
# known source prior domain (None denotes an explicit representation condition).
_ROUTES = {
    ('DNA-native-reference-prior', _NUCLEOTIDE_TASK): (
        'dna', 'DNA', 'source_native_sequence',
        '4e6860536e67a2799c206bb66b0ccce4931e78f1', 'source_DNA_sequence'),
    ('Rfam15.1-native-reference-prior', _NUCLEOTIDE_TASK): (
        'rfam_rna', 'RNA', 'source_native_sequence',
        'Rfam15.1;January2026;originalCC0FASTA', 'RNA_family_reference_native_symbols'),
    ('RNAcentral-reference-priors', _NUCLEOTIDE_TASK): (
        'rnacentral_rna', 'RNA', 'source_native_sequence',
        'officialRNAcentral28.0active+inactive', 'RNA_reference_native_symbols'),
    ('UniProt', 'protein_sequence_modeling'): (
        'protein', 'protein', 'amino_acid_sequence', 'UniProt2026_03',
        'protein_reference_native_amino_acids'),
    ('PubChem-native-SMILES-reference-prior', 'source_native_molecular_SMILES_reference_prior'): (
        'native_smiles', 'native_SMILES', 'native_smiles', _PUBCHEM_REVISION,
        'molecular_reference_native_SMILES'),
    ('PubChem-standardSELFIES-codec', 'computed_standard_SELFIES_smiles_to_selfies'): (
        'smiles_to_selfies', 'standard_SELFIES', 'standard_selfies', _PUBCHEM_REVISION, None),
    ('PubChem-standardSELFIES-codec', 'computed_standard_SELFIES_selfies_to_smiles'): (
        'selfies_to_smiles', 'native_SMILES', 'source_canonical_isomeric_smiles', _PUBCHEM_REVISION, None),
}
SUPPORTED_FAMILIES = frozenset(family for family, task in _ROUTES)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate native JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite native JSON constant')
    if type(raw) not in (str, bytes):
        raise ValueError('serialized native JSON required')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class NativePriorView:
    """Immutable full-string projection; construction alone is not authentication.

    Model entry points must call ``source_native_prior_views`` on the original
    finite view and batch, rather than trust a caller-constructed instance.
    ``datatype`` describes the target, including each codec's direction.
    """
    route: str
    datatype: str
    target_field: str
    target_bytes: bytes
    condition_bytes: bytes
    audit_bytes: bytes

    def condition_inputs(self):
        return _json(self.condition_bytes)

    def inputs(self):
        return self.condition_inputs()

    def provenance(self):
        return _json(self.audit_bytes)


def _conditions(task, route, domain):
    conditions = task['conditions']
    if type(conditions) is not dict:
        raise ValueError('explicit target-free native conditions required')
    if route in ('dna', 'rfam_rna', 'rnacentral_rna'):
        if (set(conditions) != {'prior_domain', 'source_organism'}
                or conditions['prior_domain'] != domain
                or conditions['source_organism'] is not None and
                    (type(conditions['source_organism']) is not str or not conditions['source_organism'])
                or route == 'rnacentral_rna' and conditions['source_organism'] is not None):
            raise ValueError('native nucleotide condition domain/organism allowlist differs')
        return conditions
    if route == 'protein':
        if conditions:
            raise ValueError('source protein prior has no native condition fields')
        # Explicit task semantics, not an accession/family/dataset identifier.
        return {'prior_domain': domain}
    if route == 'native_smiles':
        if conditions != {'prior_domain': domain}:
            raise ValueError('native molecular prior domain allowlist differs')
        return conditions
    source_field = 'native_smiles' if route == 'smiles_to_selfies' else 'standard_selfies'
    if (set(conditions) != {'representation_task', source_field}
            or conditions['representation_task'] != 'standard_SELFIES_codec'
            or type(conditions[source_field]) is not str or not conditions[source_field]):
        raise ValueError('exact native codec source string/direction required')
    return conditions


def _project(row, *, source_shard, ordinal, split):
    raw = row['native_payload']
    if type(raw) is not bytes or _sha(raw) != row['native_payload_sha256']:
        raise ValueError('native task payload byte integrity differs')
    task = _json(raw)
    route_spec = _ROUTES.get((row['source_family'], row['task_name']))
    if (route_spec is None or row['split'] != split or row['admitted_for_task'] is not True
            or row['native_schema'] != 'mora3.CorpusTask/v1' or type(task) is not dict
            or set(task) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
            or task['schema'] != row['native_schema'] or task['task'] != row['task_name']):
        raise ValueError('explicit native source prior route/schema/role required')
    route, datatype, target_field, revision, domain = route_spec
    if row['source_revision'] != revision:
        raise ValueError('source prior revision differs from reviewed task semantics')
    conditions = _conditions(task, route, domain)
    targets = task['targets']
    if (type(targets) is not dict or set(targets) != {target_field}
            or type(targets[target_field]) is not str or not targets[target_field]):
        raise ValueError('one complete literal native string target required')
    if (type(task['audit']) is not dict or type(task['missing']) is not list
            or any(type(value) is not str for value in task['missing'])
            or _canonical(_json(row['missing_context_json'])) != _canonical(task['missing'])):
        raise ValueError('native missingness/audit sidecar differs')
    groups = _json(row['group_ids_json'])
    if (type(groups) is not list or not groups or any(type(g) is not str or not g for g in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError('exact original source guard declarations required')
    target = targets[target_field].encode('utf8')
    inputs = _canonical(conditions)
    provenance = dict(schema='mora3.FiniteUnionNativePriorView/v1',
        original_source_shard=source_shard, original_source_row_ordinal=ordinal,
        original_split=split, original_envelope={k: v for k, v in row.items() if k != 'native_payload'},
        original_task=task, original_native_payload_sha256=_sha(raw),
        original_declared_groups=groups, target_native_utf8_sha256=_sha(target),
        condition_sha256=_sha(inputs), full_target_retained=True,
        condition_mapping=('explicit source protein task prior domain; original conditions empty'
            if route == 'protein' else 'exact original native conditions'),
        source_native_row_authentication='FiniteUnionNativeView.require_native_rows',
        new_scientific_admission=False, paired_assay_inferred=False,
        full_project_training_ready=False)
    return NativePriorView(route, datatype, target_field, target, inputs, _canonical(provenance))


def source_native_prior_views(view, batch, *, split):
    """Authenticate every original column/ordinal/guard and map seven priors.

    Explicit source-path selection belongs to ``iter_native_task_batches``;
    this function never scans other shards or changes the finite selection.
    All supported native priors currently have original train-only roles.
    Unsupported alphabets remain literal bytes here; a neural symbol codec
    must reject its unsupported values explicitly rather than repair them.
    """
    if type(view) is not FiniteUnionNativeView or split != 'train':
        raise ValueError('exact finite original-row view and original train prior role required')
    view.require_native_rows(batch, split=split)
    try:
        return tuple(_project(row, source_shard=batch.source_shard, ordinal=ordinal, split=split)
            for ordinal, row in zip(batch.source_row_ordinals, batch.native_rows.to_pylist()))
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('malformed native prior task/conditions/full string') from exc
