"""Typed current Reactome source graph and withheld SBML participant labels."""

from dataclasses import dataclass
import json
import math

from .admet import canonical, digest, _decimal, _json
from .finite_union_native_rows import FiniteUnionNativeView
from .provenance import Provenance
from .schema import Edge, GraphRecord, Measurement, Node


FAMILY = 'Reactome98-static-pathway-prior'
TASK = 'source_static_pathway_masked_participant_roles'
SCHEMA = 'mora3.CorpusTask/v1'
ROLE_AXIS = ('reactant', 'product', 'modifier')
_FIELD = 'native_context'
_PROVENANCE = (Provenance('reactome-static-known-input', 'Reactome98', 'input', '0' * 64, 'source-audit-external'),)


def _sha256(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _object(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('Exact native Reactome object fields required')


def _array(value):
    if type(value) is not list:
        raise ValueError('Exact native Reactome array required')
    return value


def _string(value, *, optional=False):
    if value is None and optional:
        return
    if type(value) is not str:
        raise ValueError('Exact native Reactome text required')
    value.encode('utf-8')


def _references(value, qualifiers):
    if type(value) is not dict or set(value) - set(qualifiers):
        raise ValueError('Exact native qualified Reactome reference context required')
    for references in value.values():
        for reference in _array(references):
            _string(reference)


def log_stoichiometry(value):
    if value is None:
        return None
    number = float(_decimal(value))
    if not math.isfinite(number) or number <= 0:
        raise ValueError('Supplied native stoichiometry must be positive and finite')
    return math.log(number)


def _check_role(role, stoichiometry):
    if role not in ROLE_AXIS:
        raise ValueError('Exact native source participant role required')
    log_stoichiometry(stoichiometry)


def _measurement(value):
    return (Measurement(_FIELD, canonical(value).decode('utf-8'), None, 'observed', True),)


def _known_graph(value):
    _object(value, ('compartments', 'species', 'reactions', 'known_participant_incidence'))
    compartments, species, reactions = (_array(value[k]) for k in ('compartments', 'species', 'reactions'))
    if value['known_participant_incidence'] is not True or not species or not reactions:
        raise ValueError('Complete known native Reactome pathway topology required')
    nodes, edges, withheld = [], [], []
    for compartment in compartments:
        _object(compartment, ('name', 'reference_qualifiers'))
        _string(compartment['name'])
        _references(compartment['reference_qualifiers'], ('is',))
        nodes.append(Node('native-compartment', _measurement(compartment)))
    for index, item in enumerate(species, start=len(compartments)):
        _object(item, ('name', 'sbo_term', 'compartment_index', 'qualified_reference_context'))
        _string(item['name'])
        _string(item['sbo_term'], optional=True)
        _references(item['qualified_reference_context'], ('is', 'hasPart', 'isHomologTo', 'hasVersion'))
        compartment = item['compartment_index']
        if compartment is not None:
            if type(compartment) is not int or not 0 <= compartment < len(compartments):
                raise ValueError('Native species compartment index outside complete source axis')
            edges.append(Edge(index, compartment, 'native-known-compartment-membership', True))
        nodes.append(Node('native-species-state', _measurement({k: v for k, v in item.items() if k != 'compartment_index'})))
    first_reaction = len(nodes)
    for index, reaction in enumerate(reactions, start=first_reaction):
        _object(reaction, ('reversible_source', 'participants'))
        _string(reaction['reversible_source'], optional=True)
        nodes.append(Node('native-reaction', _measurement({'reversible_source': reaction['reversible_source']})))
        for participant in _array(reaction['participants']):
            _object(participant, ('species_index', 'role', 'stoichiometry_source'))
            species_index = participant['species_index']
            if type(species_index) is not int or not 0 <= species_index < len(species):
                raise ValueError('Native participant index outside complete species axis')
            role, quantity = participant['role'], participant['stoichiometry_source']
            if role is None:
                if quantity is not None:
                    raise ValueError('Withheld native role cannot expose its stoichiometry')
                withheld.append(len(edges))
                fields = ()
            else:
                _check_role(role, quantity)
                fields = _measurement(dict(role=role, stoichiometry_source=quantity))
            # Endpoint order is fixed species-to-reaction for every role. It
            # carries incidence and cannot disclose the masked native category.
            edges.append(Edge(len(compartments) + species_index, index, 'native-participant-incidence', True, fields))
    if not withheld:
        raise ValueError('At least one explicitly withheld native participant is required')
    record = GraphRecord('reactome-static-known-input', tuple(nodes), tuple(edges), _PROVENANCE)
    return record, tuple(len(nodes) + edge for edge in withheld)


@dataclass(frozen=True)
class ReactomeStaticCondition:
    native_condition_bytes: bytes

    def graph(self):
        if type(self.native_condition_bytes) is not bytes:
            raise ValueError('Exact canonical native Reactome condition bytes required')
        doc = _json(self.native_condition_bytes)
        if canonical(doc) != self.native_condition_bytes:
            raise ValueError('Exact canonical native Reactome condition bytes required')
        return _known_graph(doc)


@dataclass(frozen=True)
class ReactomeStaticPacket:
    condition: ReactomeStaticCondition
    targets: tuple[tuple[str, str | None], ...]
    source_coordinate: tuple[str, int, str]


def max_entity_utf8_bytes(graph):
    """Upper bound with the live full-byte-coverage payload tokenizer."""
    return max(len(m.value.encode('utf-8')) for entity in (*graph.nodes, *graph.edges)
               for m in entity.measurements)


def project_reactome_row(row, path, ordinal):
    """Preserve every original masked occurrence in native display order."""
    try:
        if (
            (row['source_family'], row['task_name'], row['native_schema'], row['source_revision'], row['split'])
            != (FAMILY, TASK, SCHEMA, 'Reactome98', 'train')
            or row['admitted_for_task'] is not True or row['chemical_holdout_review_pass'] is not True
            or row['source_annotation_optimization_admitted'] is not True
            or row['final_full_mixture_admission'] is not False or row['model_adapter_supported'] is not False
            or row['condition_context_json'] != '{}'
        ):
            raise ValueError('Exact current source-reviewed Reactome static train envelope required')
        raw = row['native_payload']
        identity = digest(raw)
        if identity != row['native_payload_sha256']:
            raise ValueError('Original native Reactome payload checksum differs')
        doc = _json(raw)
        _object(doc, ('schema', 'task', 'missing', 'conditions', 'targets', 'audit'))
        if (doc['schema'], doc['task']) != (SCHEMA, TASK):
            raise ValueError('Exact original native Reactome task required')
        conditions = doc['conditions']
        _, slots = _known_graph(conditions)
        _object(doc['targets'], ('masked_reaction_participants',))
        values = _array(doc['targets']['masked_reaction_participants'])
        reactions = conditions['reactions']
        if len(values) != len(reactions):
            raise ValueError('Native reaction and masked participant axes differ')
        selected = doc['audit']['selected_reaction_indices']
        if (type(selected) is not list or any(type(i) is not int for i in selected)
                or selected != [i for i, value in enumerate(values) if value is not None]):
            raise ValueError('Original source mask and target reaction axes differ')
        targets = []
        for reaction, value in zip(reactions, values, strict=True):
            participants = reaction['participants']
            if value is None:
                if any(p['role'] is None for p in participants):
                    raise ValueError('Unknown unselected role in original source pathway')
                continue
            value = _array(value)
            if len(value) != len(participants) or any(p['role'] is not None for p in participants):
                raise ValueError('Masked roles must match all original participant occurrences exactly')
            for item in value:
                _object(item, ('role', 'stoichiometry_source'))
                pair = item['role'], item['stoichiometry_source']
                _check_role(*pair)
                targets.append(pair)
        if len(targets) != len(slots):
            raise ValueError('Native masked participant and target counts differ')
        return ReactomeStaticPacket(ReactomeStaticCondition(canonical(conditions)), tuple(targets), (path, ordinal, identity))
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('Malformed native Reactome source task') from exc


@dataclass(frozen=True)
class ReactomeStaticAxes:
    census_sha256: str
    native_view_policy_sha256: str
    original_train_graphs: int
    max_nodes: int
    max_entities: int
    max_entity_utf8_bytes: int

    def binding(self):
        if (
            not _sha256(self.census_sha256) or not _sha256(self.native_view_policy_sha256)
            or any(type(x) is not int or x < 1 for x in (
                self.original_train_graphs, self.max_nodes, self.max_entities, self.max_entity_utf8_bytes,
            )) or self.max_nodes > self.max_entities
        ):
            raise ValueError('Exact complete native Reactome graph envelope required')
        return dict(
            schema='Mora3.ReactomeStaticAxes/v1', census_sha256=self.census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256, original_train_graphs=self.original_train_graphs,
            max_nodes=self.max_nodes, max_entities=self.max_entities, max_entity_utf8_bytes=self.max_entity_utf8_bytes,
            native_role_axis=ROLE_AXIS, source_release='Reactome98', native_unit='SBML stoichiometric coefficient',
        )

    def require_graph(self, graph):
        if (len(graph.nodes) > self.max_nodes or len(graph.nodes) + len(graph.edges) > self.max_entities
                or max_entity_utf8_bytes(graph) > self.max_entity_utf8_bytes):
            raise ValueError('Complete native Reactome graph outside the source envelope')


def native_view_policy_sha256(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current finite native Reactome view required')
    return digest(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def source_paths_and_counts(view):
    paths = {}
    for descriptor in view.descriptors:
        rows = tuple(row for row in descriptor['source_task_schema_counts'] if row['source_family'] == FAMILY
                     and row['eligible_view_disposition'] in (0, 1, 2))
        if rows:
            if any((r['task_name'], r['native_schema'], r['split']) != (TASK, SCHEMA, 'train') for r in rows):
                raise ValueError('Unexpected retained native Reactome task or role')
            paths[descriptor['source_shard']['path']] = sum(r['rows'] for r in rows)
    if not paths:
        raise ValueError('No retained native Reactome pathway sources')
    return paths


class ReactomeStaticSource:
    def __init__(self, view, census_path, expected_sha256):
        policy = native_view_policy_sha256(view)
        path = (view.root / census_path).resolve()
        path.relative_to(view.root)
        raw = path.read_bytes()
        if not _sha256(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError('Pinned current native Reactome census differs')
        proof = _json(raw)
        paths = source_paths_and_counts(view)
        if (
            proof['schema'] != 'Mora3.CurrentNativeReactomeStaticCensus/v1'
            or proof['native_view_policy_sha256'] != policy
            or proof['original_source_path_counts'] != paths
            or proof['original_train_graphs'] != sum(paths.values())
            or tuple(proof['native_role_axis']) != ROLE_AXIS
        ):
            raise ValueError('Complete current native Reactome census/source scope required')
        self.axes = ReactomeStaticAxes(expected_sha256, policy, proof['original_train_graphs'],
                                      proof['max_nodes'], proof['max_entities'], proof['max_entity_utf8_bytes'])
        self.axes.binding()
        self.source_paths = tuple(paths)
        self._view = view

    def packets(self, batch, *, split):
        if split != 'train' or native_view_policy_sha256(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact current native Reactome view and original train role required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('Original Reactome shard outside complete retained source')
        return tuple(project_reactome_row(row, path, ordinal) for row, ordinal in zip(
            batch.native_rows.to_pylist(), batch.source_row_ordinals, strict=True,
        ))
