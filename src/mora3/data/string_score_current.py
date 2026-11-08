"""Current STRING v12.0 source score profiles and same-release sequence joins."""

from dataclasses import dataclass
import gzip
from types import MappingProxyType

from .admet import _json, canonical, digest
from .finite_union_native_rows import FiniteUnionNativeView, full9, read_exact


FAMILY = 'STRING-human-score-prior'
REVISION = 'STRING-v12.0-human-physical-detailed'
TASK = 'source_integrated_association_score_profile'
SCHEMA = 'mora3.CorpusTask/v1'
SCOPE = 'STRING-physical-association-not-direct-binding'
TAXON = 9606
UNIT = 'source_confidence_integer_0_to_1000'
SCORE_AXIS = ('experimental_channel', 'database_channel', 'textmining_channel', 'combined_score')
NATIVE_LINK_HEADER = ('protein1', 'protein2', 'experimental', 'database', 'textmining', 'combined_score')


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _provider_id(value):
    return (type(value) is str and value.startswith(f'{TAXON}.') and len(value) > 5
            and value.isascii() and not any(c.isspace() for c in value))


def valid_score_profile(values):
    return (type(values) is tuple and len(values) == len(SCORE_AXIS)
            and all(type(value) is int and 0 <= value <= 1000 for value in values))


@dataclass(frozen=True)
class StringScoreCondition:
    sequences: tuple[str, str]


@dataclass(frozen=True)
class StringScorePacket:
    condition: StringScoreCondition
    scores: tuple[int, int, int, int]
    provider_pair: tuple[str, str]
    oriented_source_lines: tuple[int, ...]
    source_pair_group: str
    source_coordinate: tuple[str, int, str]


def project_string_score_row(row, path, ordinal, sequences, link_archive_sha256):
    """Retain native endpoint identities in provenance and exact sequence-only inputs."""
    try:
        if (
            (row['source_family'], row['source_revision'], row['task_name'], row['native_schema'], row['split'])
            != (FAMILY, REVISION, TASK, SCHEMA, 'train')
            or row['source_sha256'] != link_archive_sha256
            or row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False
            or row['evidence_class'] != 'source_integrated_computed_association_confidence_prior_not_experimental_binding'
            or row['condition_context_json'] != '{}'
        ):
            raise ValueError('Exact retained original STRING score-profile source train envelope required')
        raw = row['native_payload']
        identity = digest(raw)
        if identity != row['native_payload_sha256']:
            raise ValueError('Original native STRING score profile checksum differs')
        doc = _json(raw)
        conditions, targets, audit = (doc[key] for key in ('conditions', 'targets', 'audit'))
        if (
            set(doc) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
            or (doc['schema'], doc['task']) != (SCHEMA, TASK)
            or set(conditions) != {'protein1_source_id', 'protein2_source_id', 'taxon', 'association_scope'}
            or type(conditions['taxon']) is not int or conditions['taxon'] != TAXON
            or conditions['association_scope'] != SCOPE
            or set(targets) != set(SCORE_AXIS)
            or set(audit) != {'source_oriented_rows', 'source_unordered_pair_group', 'experimental_channel_supported',
                             'experimental_truth', 'negative_label', 'direct_binding_demonstrated'}
            or audit['experimental_truth'] is not False or audit['negative_label'] is not None
            or audit['direct_binding_demonstrated'] is not None
        ):
            raise ValueError('Exact source confidence evidence context and typed coordinates required')
        pair = (conditions['protein1_source_id'], conditions['protein2_source_id'])
        if not all(_provider_id(value) for value in pair) or pair[0] > pair[1]:
            raise ValueError('Exact source canonical provider-pair ordering required')
        lines = audit['source_oriented_rows']
        group = audit['source_unordered_pair_group']
        if (
            type(lines) is not list or len(lines) != (1 if pair[0] == pair[1] else 2)
            or any(type(line) is not int or line < 2 for line in lines) or lines != sorted(set(lines))
            or type(group) is not str or not group
        ):
            raise ValueError('Complete original oriented link line coordinates required')
        score = []
        for field in SCORE_AXIS:
            item = targets[field]
            if type(item) is not dict or set(item) != {'unit', 'value'} or item['unit'] != UNIT:
                raise ValueError('Exact native STRING 0..1000 score unit required')
            score.append(item['value'])
        score = tuple(score)
        if not valid_score_profile(score) or audit['experimental_channel_supported'] is not (score[0] > 0):
            raise ValueError('Exact original observed source score vector required')
        native_sequences = tuple(sequences[key] for key in pair)
        if any(type(sequence) is not str or not sequence for sequence in native_sequences):
            raise ValueError('Complete same-release native STRING sequences required')
        return StringScorePacket(StringScoreCondition(native_sequences), score, pair, tuple(lines), group,
                                 (path, ordinal, identity))
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('Malformed current native STRING score profile or sequence join') from exc


def _read_fasta(root, reference):
    """Read the complete pinned same-release source FASTA without cross-release joins."""
    path = (root / reference['path']).resolve()
    path.relative_to(root)
    before = full9(path)
    raw = read_exact(reference, root=root)
    output = {}
    identifier, chunks = None, []
    body = gzip.decompress(raw).decode('ascii')
    for line in body.splitlines():
        if line.startswith('>'):
            if identifier is not None:
                if not chunks or identifier in output:
                    raise ValueError('Empty or duplicate original native STRING FASTA record')
                output[identifier] = ''.join(chunks)
            header = line[1:].split()
            if not header or not _provider_id(header[0]):
                raise ValueError('Invalid original native STRING FASTA provider ID')
            identifier, chunks = header[0], []
        elif identifier is None or not line or not line.isascii() or not line.isalpha() or not line.isupper():
            raise ValueError('Invalid original native STRING FASTA sequence text')
        else:
            chunks.append(line)
    if not chunks or identifier in output:
        raise ValueError('Empty or duplicate original native STRING FASTA record')
    output[identifier] = ''.join(chunks)
    if full9(path) != before:
        raise ValueError('Original STRING FASTA identity changed during source binding')
    return MappingProxyType(output)


@dataclass(frozen=True)
class StringScoreAxes:
    census_sha256: str
    native_view_policy_sha256: str
    link_archive_sha256: str
    sequence_archive_sha256: str
    original_train_pairs: int
    max_sequence_utf8_bytes: int
    native_sequence_alphabet: str

    def binding(self):
        alphabet = self.native_sequence_alphabet
        if (
            not all(_sha(value) for value in (self.census_sha256, self.native_view_policy_sha256,
                                             self.link_archive_sha256, self.sequence_archive_sha256))
            or type(self.original_train_pairs) is not int or self.original_train_pairs < 1
            or type(self.max_sequence_utf8_bytes) is not int or self.max_sequence_utf8_bytes < 1
            or type(alphabet) is not str or not alphabet or not alphabet.isascii() or not alphabet.isalpha()
            or not alphabet.isupper() or alphabet != ''.join(sorted(set(alphabet)))
        ):
            raise ValueError('Exact same-release STRING score and sequence source axes required')
        return dict(
            schema='Mora3.StringScoreAxes/v1', census_sha256=self.census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256, link_archive_sha256=self.link_archive_sha256,
            sequence_archive_sha256=self.sequence_archive_sha256, original_train_pairs=self.original_train_pairs,
            max_sequence_utf8_bytes=self.max_sequence_utf8_bytes, native_sequence_alphabet=alphabet,
            native_score_axis=SCORE_AXIS, unit=UNIT, taxon=TAXON, association_scope=SCOPE,
        )

    def require_condition(self, condition):
        if (
            type(condition) is not StringScoreCondition or type(condition.sequences) is not tuple
            or len(condition.sequences) != 2
            or any(type(sequence) is not str or not 1 <= len(sequence) <= self.max_sequence_utf8_bytes
                   or set(sequence) - set(self.native_sequence_alphabet) for sequence in condition.sequences)
        ):
            raise ValueError('Two complete same-release native STRING sequence contexts required')


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current finite native STRING score view required')
    return digest(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def source_paths_and_counts(view):
    paths = {}
    for descriptor in view.descriptors:
        matches = [entry for entry in descriptor['source_task_schema_counts'] if entry['source_family'] == FAMILY
                   and entry['eligible_view_disposition'] in (0, 1, 2)]
        if matches:
            if any((row['task_name'], row['native_schema'], row['split']) != (TASK, SCHEMA, 'train') for row in matches):
                raise ValueError('Unexpected current STRING score profile schema or original role')
            paths[descriptor['source_shard']['path']] = sum(row['rows'] for row in matches)
    if not paths:
        raise ValueError('No current retained original STRING score-profile source paths')
    return paths


class StringScoreCurrentSource:
    def __init__(self, view, census_path, expected_sha256):
        policy = _view_policy(view)
        path = (view.root / census_path).resolve()
        path.relative_to(view.root)
        raw = path.read_bytes()
        if not _sha(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError('Pinned current STRING score-profile source census checksum differs')
        proof = _json(raw)
        paths = source_paths_and_counts(view)
        if (
            proof['schema'] != 'Mora3.StringCurrentScoreProfileCensus/v1'
            or proof['native_view_policy_sha256'] != policy or proof['original_source_path_counts'] != paths
            or proof['original_train_pairs'] != sum(paths.values())
            or tuple(proof['native_score_axis']) != SCORE_AXIS
            or proof['all_selected_oriented_source_rows_exact'] is not True
            or proof['all_retained_endpoints_resolve_exact_same_release_sequence'] is not True
        ):
            raise ValueError('Exact complete original retained STRING source census required')
        fasta, links = proof['sequence_archive'], proof['link_archive']
        self.axes = StringScoreAxes(expected_sha256, policy, links['sha256'], fasta['sha256'],
                                    proof['original_train_pairs'], proof['max_native_sequence_utf8_bytes'],
                                    proof['native_sequence_alphabet'])
        self.axes.binding()
        self.source_paths = tuple(paths)
        self._sequences = _read_fasta(view.root, fasta)
        if len(self._sequences) != proof['all_source_Fasta_ids']:
            raise ValueError('Original STRING same-release sequence population differs')
        self._view = view

    def packets(self, batch, *, split):
        if split != 'train' or _view_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact current STRING score source and original train role required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('Current STRING score path outside the bound original source census')
        return tuple(project_string_score_row(row, path, ordinal, self._sequences, self.axes.link_archive_sha256)
                     for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals, strict=True))
