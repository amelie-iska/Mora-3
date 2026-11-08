"""Current original BioGRID chemical/gene annotation occurrences."""

from dataclasses import dataclass

from .admet import _json, canonical, digest
from .finite_union_native_rows import FiniteUnionNativeView


FAMILY = 'BioGRID'
REVISION = 'BioGRID5.0.262'
TASK = 'source_annotated_chemical_gene_action'
SCHEMA = 'mora3.CorpusTask/v1'
KNOWN_FIELDS = ('chemical_identifier', 'chemical_name', 'gene_identifier', 'organism_taxon')
TARGET_FIELDS = ('action', 'interaction_type')


def _literal(value):
    if type(value) is not str or not value:
        raise ValueError('Exact nonempty source BioGRID text required')
    return value.encode('utf-8')


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class BioGRIDChemicalCondition:
    chemical_identifier: str
    chemical_name: str
    gene_identifier: str
    organism_taxon: str

    def inputs(self):
        values = {field: getattr(self, field) for field in KNOWN_FIELDS}
        for value in values.values():
            _literal(value)
        return values


@dataclass(frozen=True)
class BioGRIDChemicalPacket:
    condition: BioGRIDChemicalCondition
    target: tuple[str, str]
    source_coordinate: tuple[str, int, str]


def project_biogrid_chemical_row(row, path, ordinal):
    """Verify the exact current native projection against its original source-row audit."""
    try:
        if (
            (row['source_family'], row['source_revision'], row['task_name'], row['native_schema'], row['split'])
            != (FAMILY, REVISION, TASK, SCHEMA, 'train')
            or row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False
            or row['evidence_class'] != 'source_curated_publication_supported_annotation'
            or row['condition_context_json'] != '{}'
        ):
            raise ValueError('Exact retained original BioGRID chemical annotation train envelope required')
        raw = row['native_payload']
        identity = digest(raw)
        if identity != row['native_payload_sha256']:
            raise ValueError('Original BioGRID native payload checksum differs')
        doc = _json(raw)
        if (set(doc) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
                or (doc['schema'], doc['task']) != (SCHEMA, TASK)
                or set(doc['conditions']) != set(KNOWN_FIELDS) or set(doc['targets']) != set(TARGET_FIELDS)
                or doc['audit']['evidence'] != 'original BIOGRID curated'):
            raise ValueError('Exact native BioGRID chemical condition and target fields required')
        original = doc['audit']['source_row']
        target = tuple(doc['targets'][field] for field in TARGET_FIELDS)
        for label in target:
            _literal(label)
        expected = dict(
            chemical_identifier=original['Chemical Source'].upper() + ':' + original['Chemical Source ID'],
            chemical_name=original['Chemical Name'],
            gene_identifier=original['Entrez Gene ID'],
            organism_taxon=original['Organism ID'],
        )
        condition = BioGRIDChemicalCondition(**doc['conditions'])
        if condition.inputs() != expected or target != (original['Action'], original['Interaction Type']):
            raise ValueError('Native BioGRID source audit and literal fields disagree')
        return BioGRIDChemicalPacket(condition, target, (path, ordinal, identity))
    except (KeyError, AttributeError, TypeError, UnicodeError) as exc:
        raise ValueError('Malformed native BioGRID chemical annotation') from exc


@dataclass(frozen=True)
class BioGRIDChemicalAxes:
    census_sha256: str
    native_view_policy_sha256: str
    original_train_rows: int
    target_pairs: tuple[tuple[str, str], ...]
    known_field_max_utf8_bytes: tuple[int, ...]

    def binding(self):
        pairs = self.target_pairs
        bounds = self.known_field_max_utf8_bytes
        if (
            not _sha(self.census_sha256) or not _sha(self.native_view_policy_sha256)
            or type(self.original_train_rows) is not int or self.original_train_rows < 1
            or type(pairs) is not tuple or not 1 <= len(pairs) <= self.original_train_rows
            or any(type(pair) is not tuple or len(pair) != 2 for pair in pairs)
            or type(bounds) is not tuple or len(bounds) != len(KNOWN_FIELDS)
            or any(type(v) is not int or v < 1 for v in bounds)
        ):
            raise ValueError('Complete exact BioGRID chemical source axes and capacity required')
        order = lambda pair: tuple(_literal(value) for value in pair)
        if pairs != tuple(sorted(set(pairs), key=order)):
            raise ValueError('BioGRID native joint source classes must be unique and UTF-8 lexically ordered')
        return dict(
            schema='Mora3.BioGRIDChemicalAxes/v1', source_release=REVISION, census_sha256=self.census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256, original_train_rows=self.original_train_rows,
            full_literal_target_axis=pairs, known_field_axis=KNOWN_FIELDS, known_field_max_utf8_bytes=bounds,
        )

    def require_condition(self, condition):
        if type(condition) is not BioGRIDChemicalCondition or any(
            len(_literal(value)) > bound
            for value, bound in zip(condition.inputs().values(), self.known_field_max_utf8_bytes, strict=True)
        ):
            raise ValueError('Complete target-free native BioGRID condition outside the source envelope')


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current finite native BioGRID view required')
    return digest(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def source_paths_and_counts(view):
    paths = {}
    for descriptor in view.descriptors:
        matches = [entry for entry in descriptor['source_task_schema_counts'] if entry['source_family'] == FAMILY
                   and entry['task_name'] == TASK and entry['eligible_view_disposition'] in (0, 1, 2)]
        if matches:
            if any((row['native_schema'], row['split']) != (SCHEMA, 'train') for row in matches):
                raise ValueError('Unexpected current BioGRID chemical annotation schema or original role')
            paths[descriptor['source_shard']['path']] = sum(row['rows'] for row in matches)
    if not paths:
        raise ValueError('No current retained original BioGRID chemical task paths')
    return paths


class BioGRIDChemicalCurrentSource:
    def __init__(self, view, census_path, expected_sha256):
        policy = _view_policy(view)
        path = (view.root / census_path).resolve()
        path.relative_to(view.root)
        raw = path.read_bytes()
        if not _sha(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError('Pinned current BioGRID chemical census checksum differs')
        proof = _json(raw)
        paths = source_paths_and_counts(view)
        if (
            proof['schema'] != 'Mora3.BioGRIDCurrentChemicalAnnotationCensus/v1'
            or proof['native_view_policy_sha256'] != policy or proof['original_source_path_counts'] != paths
            or proof['original_train_rows'] != sum(paths.values())
            or tuple(proof['known_field_axis']) != KNOWN_FIELDS
            or proof['all_retained_source_audit_rows_match_original_release'] is not True
        ):
            raise ValueError('Exact complete BioGRID chemical source census required')
        self.axes = BioGRIDChemicalAxes(expected_sha256, policy, proof['original_train_rows'],
                                        tuple(tuple(x) for x in proof['full_literal_target_axis']),
                                        tuple(proof['known_field_max_utf8_bytes']))
        self.axes.binding()
        self.source_paths = tuple(paths)
        self._view = view

    def packets(self, batch, *, split):
        if split != 'train' or _view_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact current BioGRID chemical annotation source and original train role required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('BioGRID chemical source path outside the bound source census')
        packets = tuple(project_biogrid_chemical_row(row, path, ordinal) for row, ordinal in zip(
            batch.native_rows.to_pylist(), batch.source_row_ordinals, strict=True,
        ))
        for packet in packets:
            self.axes.require_condition(packet.condition)
            if packet.target not in self.axes.target_pairs:
                raise ValueError('BioGRID chemical annotation outside the exact native joint target axis')
        return packets
