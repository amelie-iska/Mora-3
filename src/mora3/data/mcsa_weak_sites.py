"""Sparse MCSA source-annotation-list reconstruction on current references.

These are source claims, not physical catalytic truth. Only an exact, checked
source position selects a loss location. Unlisted positions have no target.
Role lists retain source order and repetitions; EOS means the observed source
list ended. It never labels absent biological roles or unlisted residues.
This candidate reader cannot itself admit a corpus or activate training.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re

from .mcsa_sequence_context import check_residue_mapping

ROLE_FIELDS = ('emo', 'function', 'function_type', 'group_function')


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _role_key(role):
    if (type(role) is not dict or set(role) != set(ROLE_FIELDS)
            or any(type(role[k]) is not str for k in ROLE_FIELDS)):
        raise ValueError('exact native four-string role tuple required')
    return tuple(role[k] for k in ROLE_FIELDS)


@dataclass(frozen=True)
class WeakRoleVocabulary:
    """Exact frozen source tuple axis, followed by EOS; BOS is input-only."""
    roles: tuple

    def __post_init__(self):
        if (type(self.roles) is not tuple or not self.roles
                or any(type(r) is not tuple or len(r) != 4 or any(type(x) is not str for x in r) for r in self.roles)
                or tuple(sorted(set(self.roles))) != self.roles):
            raise ValueError('nonempty unique lexically ordered exact role tuple axis required')

    @classmethod
    def from_document(cls, document):
        if (type(document) is not dict or document.get('schema') != 'Mora3.MCSAExactSourceRoleListVocabulary/v1'
                or document.get('role_fields') != list(ROLE_FIELDS) or type(document.get('roles')) is not list):
            raise ValueError('frozen MCSA source role vocabulary document required')
        roles = []
        for i, role in enumerate(document['roles']):
            if type(role) is not dict or type(role.get('id')) is not int or role['id'] != i:
                raise ValueError('consecutive frozen role IDs required')
            roles.append(_role_key({k: role[k] for k in ROLE_FIELDS} if set(role) == {'id', *ROLE_FIELDS} else {}))
        result = cls(tuple(roles))
        if (any(type(document.get(k)) is not int for k in ('EOS_target_id', 'BOS_input_only_id', 'prediction_class_count', 'decoder_input_class_count'))
                or document.get('EOS_target_id') != result.eos_id or document.get('BOS_input_only_id') != result.bos_id
                or document.get('prediction_class_count') != result.prediction_class_count
                or document.get('decoder_input_class_count') != result.decoder_input_class_count):
            raise ValueError('frozen termination/input axis does not match roles')
        return result

    @property
    def axis_sha256(self):
        return hashlib.sha256(json.dumps([ROLE_FIELDS, self.roles, 'ordered-native-source-list/EOS/BOS-v1'], separators=(',', ':')).encode()).hexdigest()

    @property
    def eos_id(self):
        return len(self.roles)

    @property
    def bos_id(self):
        return len(self.roles) + 1

    @property
    def prediction_class_count(self):
        return len(self.roles) + 1

    @property
    def decoder_input_class_count(self):
        return len(self.roles) + 2

    def encode(self, native_roles):
        if type(native_roles) is not list:
            raise ValueError('observed source role list required')
        lookup = {key: i for i, key in enumerate(self.roles)}
        result = []
        for role in native_roles:
            key = _role_key(role)
            if key not in lookup:
                raise ValueError('unknown source role cannot silently expand frozen axis')
            result.append(lookup[key])
        return tuple(result)


@dataclass(frozen=True)
class WeakSiteSourceList:
    position_0based: int
    amino_acid: str
    role_ids: tuple
    source_residue_ordinal: int
    source_mapping_ordinal: int
    native_annotation: dict


@dataclass(frozen=True)
class WeakSiteCandidate:
    canonical_sequence: str
    source_lists: tuple
    vocabulary_axis_sha256: str
    audit: dict

    def condition_inputs(self):
        # Never expose source IDs, known-positive positions, target role lists,
        # group/split metadata, chains, or annotations as predictor conditions.
        return {'canonical_sequence': self.canonical_sequence}


@dataclass(frozen=True)
class SourceListSupervision:
    record_index: int
    position_0based: int
    decoder_input_ids: tuple
    target_ids: tuple
    source_residue_ordinal: int
    source_mapping_ordinal: int


@dataclass(frozen=True)
class WeakSiteBatch:
    canonical_sequences: tuple
    source_list_supervision: tuple
    audit: tuple
    prediction_class_count: int
    decoder_input_class_count: int
    vocabulary_axis_sha256: str
    task_admitted: bool = False

    def condition_inputs(self):
        return {'canonical_sequences': self.canonical_sequences}


def parse_weak_site_candidate(record, vocabulary):
    if (type(record) is not dict or record.get('schema') != 'Mora3.MCSAWeakReferenceSiteCandidate/v1'
            or not isinstance(vocabulary, WeakRoleVocabulary)):
        raise ValueError('MCSA weak source-list candidate and frozen role vocabulary required')
    conditions = record.get('conditions')
    if type(conditions) is not dict or set(conditions) != {'canonical_sequence'}:
        raise ValueError('only canonical sequence is a predictor condition')
    sequence = conditions['canonical_sequence']
    if type(sequence) is not str or re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+', sequence) is None:
        raise ValueError('complete current canonical protein sequence required')
    audit = record.get('audit')
    if type(audit) is not dict:
        raise ValueError('separate source/version/group audit context required')
    if (not _integer(audit.get('source_mcsa_id'), 1)
            or record.get('record_id') != 'mcsa-weak-reference:' + str(audit['source_mcsa_id'])):
        raise ValueError('exact source candidate identity required')
    provenance = audit.get('current_uniprot_provenance')
    if (type(provenance) is not dict
            or provenance.get('canonical_sequence_sha256') != hashlib.sha256(sequence.encode()).hexdigest()
            or provenance.get('primary_accession') != audit.get('reference_uniprot_id')):
        raise ValueError('exact accession and current sequence binding required')
    if (record.get('unlisted_residue_loss_mask') is not False
            or record.get('unlisted_residues_are_negative') is not False
            or record.get('source_annotation_not_physical_or_experimental_truth') is not True
            or record.get('source_version_unknown_is_explicit_weak_evidence_not_label_repair') is not True):
        raise ValueError('sparse source-claim and sequence-version uncertainty semantics required')
    if record.get('task_admission') is not False or record.get('training') is not False:
        raise ValueError('candidate reader cannot grant corpus admission or activate training')
    if (type(record.get('proposed_group')) is not str or not record['proposed_group']
            or record.get('proposed_split') not in ('train', 'validation', 'test')
            or type(record.get('group_key_evidence')) is not list):
        raise ValueError('nonapplying source grouping and original proposed role context required')
    targets = record.get('targets')
    if (type(targets) is not dict or set(targets) != {'source_annotated_site_roles'}
            or type(targets['source_annotated_site_roles']) is not list or not targets['source_annotated_site_roles']):
        raise ValueError('nonempty sparse source annotation target list required')
    sites = []
    source_coordinates = set()
    for site in targets['source_annotated_site_roles']:
        if (type(site) is not dict or not _integer(site.get('source_residue_ordinal'))
                or not _integer(site.get('source_mapping_ordinal'))
                or not _integer(site.get('position_1based'), 1)
                or site.get('evidence_role') != 'weak_source_curated_annotation_on_current_accession_reference'
                or site.get('experimental_catalytic_truth') is not False
                or site.get('physical_original_sequence_identity_proven') is not False):
            raise ValueError('exact sparse source coordinates and weak evidence role required')
        coordinate = (site['source_residue_ordinal'], site['source_mapping_ordinal'])
        if coordinate in source_coordinates:
            raise ValueError('duplicate source mapping view')
        source_coordinates.add(coordinate)
        check = check_residue_mapping(site.get('source_mapping'), sequence, audit.get('reference_uniprot_id'))
        if (check['disposition'] != 'POSITION_CODE_AGREES_CURRENT_SEQUENCE'
                or check['source_position'] != site['position_1based']):
            raise ValueError('source position and amino acid must agree current sequence')
        annotation = site.get('native_annotation')
        if type(annotation) is not dict or type(annotation.get('roles')) is not list:
            raise ValueError('observed native annotation role list required')
        native_mappings = annotation.get('residue_sequences')
        if (type(native_mappings) is not list or site['source_mapping_ordinal'] >= len(native_mappings)
                or native_mappings[site['source_mapping_ordinal']] != site['source_mapping']
                or annotation.get('mcsa_id') != audit['source_mcsa_id']):
            raise ValueError('exact native annotation and mapping target pointer required')
        sites.append(WeakSiteSourceList(site['position_1based'] - 1,
            check['current_sequence_residue'], vocabulary.encode(annotation['roles']),
            *coordinate, copy.deepcopy(annotation)))
    if record.get('loss_positions_1based') != [site.position_0based + 1 for site in sites]:
        raise ValueError('sparse source loss positions must exactly match retained mapping views')
    metadata = copy.deepcopy(audit)
    metadata.update(record_id=record.get('record_id'), proposed_group=record['proposed_group'],
        proposed_split=record['proposed_split'], group_key_evidence=copy.deepcopy(record['group_key_evidence']),
        task_admitted=False, objective='ordered_source_annotation_role_list_reconstruction_with_EOS',
        source_list_complements_are_not_biological_negatives=True,
        unlisted_residue_targets_created=False)
    return WeakSiteCandidate(sequence, tuple(sites), vocabulary.axis_sha256, metadata)


def collate_weak_site_candidates(records, vocabulary):
    if not isinstance(vocabulary, WeakRoleVocabulary):
        raise ValueError('frozen role vocabulary required')
    records = tuple(records)
    if not records or any(not isinstance(r, WeakSiteCandidate) for r in records):
        raise ValueError('nonempty validated source-specific candidate records required')
    supervision = []
    for record_index, record in enumerate(records):
        if record.vocabulary_axis_sha256 != vocabulary.axis_sha256:
            raise ValueError('candidate and batch role axis identities must agree')
        for site in record.source_lists:
            if any(not _integer(i) or i >= len(vocabulary.roles) for i in site.role_ids):
                raise ValueError('site role IDs do not match supplied frozen axis')
            supervision.append(SourceListSupervision(record_index, site.position_0based,
                (vocabulary.bos_id, *site.role_ids), (*site.role_ids, vocabulary.eos_id),
                site.source_residue_ordinal, site.source_mapping_ordinal))
    return WeakSiteBatch(tuple(r.canonical_sequence for r in records), tuple(supervision),
        tuple(copy.deepcopy(r.audit) for r in records), vocabulary.prediction_class_count,
        vocabulary.decoder_input_class_count, vocabulary.axis_sha256)


def read_weak_site_candidates(path, expected_sha256, vocabulary):
    """Read a whole pinned small projection; no source download or admission."""
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('exact candidate-file SHA256 required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('candidate projection SHA256 mismatch')
    result = []
    seen = set()
    for line in raw.splitlines():
        record = json.loads(line)
        if record.get('record_id') in seen:
            raise ValueError('duplicate source candidate record ID')
        seen.add(record.get('record_id'))
        result.append(parse_weak_site_candidate(record, vocabulary))
    return tuple(result)
