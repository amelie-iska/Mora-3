"""Native Mora radius graphs from one explicit baseline specimen.

This component builds a query, not supervision or source admission. Geometry
comes exclusively from baseline coordinates. Symmetric adjacency is encoded
as reciprocal directed edges for the existing Mora payload encoder; direction
does not assert a biological or causal relationship.
"""
from dataclasses import asdict, dataclass
import json
import math

from mora3.data.provenance import Provenance
from mora3.data.schema import Edge, GraphRecord, Measurement, Node
from mora3.data.tasks import ConditionalTask
from .contracts import StateRecord, canonical, number, text, validate_record


def _axis(value):
    if type(value) is not tuple or not value:
        raise ValueError('nonempty exact ordered native target_axis tuple required')
    for item in value:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError('target axis entries must be (modality, feature_id, native_unit)')
        for part in item:
            text(part, 'target axis component')
    if len({item[:2] for item in value}) != len(value):
        raise ValueError('duplicate native target feature')
    return value


@dataclass(frozen=True)
class SpatialQuery:
    record: GraphRecord
    task: ConditionalTask
    target_axis: tuple[tuple[str, str, str], ...]
    audit: dict

    @property
    def query_node_index(self):
        return 0


def validate_spatial_query(query):
    if type(query) is not SpatialQuery or type(query.record) is not GraphRecord or type(query.task) is not ConditionalTask:
        raise ValueError('typed native SpatialQuery, GraphRecord and ConditionalTask required')
    _axis(query.target_axis)
    if not 1 <= len(query.record.nodes) <= 128 or not query.task.topology_known:
        raise ValueError('bounded baseline topology required')
    query.task.input_fields(query.record)
    fields = query.task._fields(query.record)
    roles = dict(query.task.roles)
    targets = {ref for ref, role in query.task.roles if role == 'target'}
    expected = {('node', 0, f'target:{i}') for i in range(len(query.target_axis))}
    if targets != expected:
        raise ValueError('exact ordered query target placeholders required')
    for i, axis in enumerate(query.target_axis):
        value = fields[('node', 0, f'target:{i}')]
        if value.value is not None or value.available or value.status != 'missing' or value.units != axis[2]:
            raise ValueError('query targets must remain missing in exact native units')
    coordinates = []
    coordinate_unit = None
    shared_context = None
    native_units = {}
    target_native_units = {(modality, feature): unit
                           for modality, feature, unit in query.target_axis}
    for i, node in enumerate(query.record.nodes):
        if node.kind != 'scri-cell':
            raise ValueError('exact native spatial node kind required')
        named = {m.name: m for m in node.measurements}
        fixed = {'is_query', 'organism', 'cell_type', 'baseline_time', 'interventions'}
        if i == 0:
            fixed |= {'requested_axis', 'radius', *(f'target:{j}' for j in range(len(query.target_axis)))}
        if not fixed <= set(named):
            raise ValueError('complete declared baseline node fields required')
        native, observed, axes = set(), set(), set()
        for name, measurement in named.items():
            expected_role = ('target' if name in fixed and name.startswith('target:')
                             else 'condition' if measurement.available else 'unused')
            if roles[('node', i, name)] != expected_role:
                raise ValueError('exact baseline condition, missingness and target roles required')
            if name in fixed:
                continue
            if name in ('coordinate:0', 'coordinate:1', 'coordinate:2'):
                axes.add(name)
                continue
            if name.startswith('native:') or name.startswith('observed:'):
                prefix, key = name.split(':', 1)
                try:
                    pair = json.loads(key)
                except (ValueError, TypeError) as exc:
                    raise ValueError('exact native modality-feature field identity required') from exc
                if (type(pair) is not list or len(pair) != 2
                        or any(type(x) is not str or not x.strip() for x in pair)
                        or canonical(pair).decode() != key):
                    raise ValueError('exact native modality-feature field identity required')
                (native if prefix == 'native' else observed).add(key)
                continue
            raise ValueError('undeclared predictor field; audit and future targets must stay outside conditions')
        if not native or native != observed:
            raise ValueError('paired native measurement and explicit missingness fields required')
        for key in native:
            value, mask = named['native:' + key], named['observed:' + key]
            if (value.units is None or mask.value is not value.available
                    or mask.status != 'observed' or not mask.available or mask.units != 'dimensionless'):
                raise ValueError('native measurement missingness or units changed')
            feature = tuple(json.loads(key))
            # The exact source feature must not acquire a new physical unit in
            # a neighbor or in the query's redacted target-coordinate field.
            if (feature in native_units and native_units[feature] != value.units
                    or feature in target_native_units and target_native_units[feature] != value.units):
                raise ValueError('native feature unit differs across cells or from supervised target axis')
            native_units[feature] = value.units
            # Spatial query supervision cannot be an observed input at its own
            # query node. Neighbors retain their original measured features.
            if i == 0 and feature in target_native_units:
                if value.value is not None or value.available or value.status != 'missing' or mask.value is not False:
                    raise ValueError('query target feature must be withheld from conditions')
        if axes not in ({'coordinate:0', 'coordinate:1'},
                        {'coordinate:0', 'coordinate:1', 'coordinate:2'}):
            raise ValueError('complete native 2D or 3D coordinate axis required')
        point = tuple(named[f'coordinate:{j}'] for j in range(len(axes)))
        for value in point:
            number(value.value, 'baseline coordinate')
            if value.status != 'observed' or value.units is None:
                raise ValueError('observed coordinates with explicit native units required')
            if coordinate_unit is None:
                coordinate_unit = value.units
            elif value.units != coordinate_unit:
                raise ValueError('mixed coordinate units')
        if coordinates and len(point) != len(coordinates[0]):
            raise ValueError('mixed coordinate dimensions')
        coordinates.append(tuple(m.value for m in point))
        context = tuple(named[name] for name in ('organism', 'baseline_time', 'interventions'))
        if any(not m.available or m.status != 'observed' for m in context):
            raise ValueError('known baseline context required')
        if shared_context is None:
            shared_context = context
        elif context != shared_context:
            raise ValueError('mixed predictive baseline context')
        marker = next((m for m in node.measurements if m.name == 'is_query'), None)
        if (marker is None or marker.value is not (i == 0) or not marker.available
                or marker.status != 'observed' or marker.units != 'dimensionless'):
            raise ValueError('exactly node zero must carry the query indicator')
    requested = fields[('node', 0, 'requested_axis')]
    if requested.value != canonical(query.target_axis).decode():
        raise ValueError('requested axis condition must match the exact ordered target axis')
    radius_field = fields[('node', 0, 'radius')]
    number(radius_field.value, 'native radius')
    if radius_field.value <= 0 or radius_field.units != coordinate_unit:
        raise ValueError('positive radius in exact native coordinate units required')
    if any(math.dist(coordinates[0], point) > radius_field.value for point in coordinates):
        raise ValueError('node lies outside the requested baseline neighborhood')
    edges = {(e.source, e.target): e for e in query.record.edges}
    if len(edges) != len(query.record.edges):
        raise ValueError('duplicate radius edges')
    for (i, j), edge in edges.items():
        reverse = edges.get((j, i))
        if (i == j or not edge.directed or edge.kind != 'baseline-radius-neighbor'
                or reverse is None or reverse.measurements != edge.measurements):
            raise ValueError('radius adjacency must be complete reciprocal noncausal edges')
        expected_distance = Measurement('distance', math.dist(coordinates[i], coordinates[j]),
                                        coordinate_unit, 'observed', True)
        if edge.measurements != (expected_distance,):
            raise ValueError('edge distance must derive exclusively from baseline coordinates')
    expected_edges = {(i, j) for i in range(len(coordinates)) for j in range(len(coordinates))
                      if i != j and math.dist(coordinates[i], coordinates[j]) <= radius_field.value}
    if set(edges) != expected_edges or any(roles[('edge', i, 'distance')] != 'condition'
                                          for i in range(len(query.record.edges))):
        raise ValueError('exact complete baseline radius topology and condition roles required')
    return query


def _cohort(row):
    if row.coordinates is None or row.time_value is None or row.source.donor_id is None:
        raise ValueError('known baseline coordinates, time and donor are required')
    for value in (row.coordinate_frame, row.coordinate_unit, row.time_unit,
                  row.time_reference, row.source.sample_id, row.source.study_id,
                  row.source.experiment_id, row.source.donor_id, row.organism):
        text(value, 'native cohort context')
        if value.strip().lower() in ('unknown', 'unspecified', 'none', 'na', 'n/a'):
            raise ValueError('unknown native cohort context cannot establish a neighborhood')
    # An unreviewed mixture of source licenses cannot become one predictive
    # spatial specimen simply because study/donor/geometry happen to match.
    # Distinct file locators within an otherwise uniform rights stratum remain
    # eligible; independent source-license reconciliation belongs upstream.
    return (row.source.dataset_id, row.source.revision, row.source.license_id,
            row.source.study_id, row.source.experiment_id, row.source.donor_id,
            row.source.sample_id, row.source.evidence_kind, row.organism,
            row.coordinate_frame, row.coordinate_unit, len(row.coordinates),
            row.time_value, row.time_unit, row.time_reference,
            row.intervention_ids, row.official_split)


def _biological_key(row):
    # Never order nodes by record IDs, donor IDs, accession strings, or hashes.
    # Ties with identical biological inputs are predictively interchangeable.
    return (row.coordinates, canonical({
        'cell_type': row.cell_type, 'measurements': [asdict(v) for v in row.measurements],
    }))


def build_neighborhood_query(baseline_records, *, query_record_id, target_axis,
                             radius, max_cells=128):
    """Select the complete closed-radius baseline neighborhood around a cell.

    ``sample_id`` is the explicit specimen identity. All supplied records must
    describe one source, donor, specimen, baseline time, frame and intervention
    cohort, including records outside the radius. Features on the requested
    target axis are withheld from the query cell's predictive measurements;
    original source rows and neighbor conditions are not modified. No
    nearest-neighbor fallback, normalization, unit conversion or truncation.
    """
    if type(baseline_records) not in (tuple, list) or not baseline_records:
        raise ValueError('explicit nonempty bounded baseline record collection required')
    if type(max_cells) is not int or not 1 <= max_cells <= 128:
        raise ValueError('max_cells must be an exact integer in 1..128')
    number(radius, 'native radius')
    if radius <= 0:
        raise ValueError('native radius must be positive')
    text(query_record_id, 'query_record_id')
    target_axis = _axis(target_axis)
    rows = tuple(validate_record(row) for row in baseline_records)
    by_id = {row.record_id: row for row in rows}
    if len(by_id) != len(rows) or query_record_id not in by_id:
        raise ValueError('unique baseline IDs and an explicit present query cell required')
    query = by_id[query_record_id]
    cohort = _cohort(query)
    if any(_cohort(row) != cohort for row in rows):
        raise ValueError('one coherent source license, study, experiment, donor, specimen, organism, frame, unit, time and intervention cohort required')
    native_units = {}
    for row in rows:
        for item in row.measurements:
            key = (item.modality, item.feature_id)
            if key in native_units and native_units[key] != item.unit:
                raise ValueError('one native unit per measured feature required; no implicit conversion')
            native_units[key] = item.unit
    for modality, feature, unit in target_axis:
        if (modality, feature) in native_units and native_units[(modality, feature)] != unit:
            raise ValueError('requested target unit conflicts with its native baseline feature')
    distances = {row.record_id: math.dist(query.coordinates, row.coordinates) for row in rows}
    for distance in distances.values():
        number(distance, 'baseline distance')
    selected = [row for row in rows if distances[row.record_id] <= radius and row is not query]
    selected.sort(key=_biological_key)
    selected = (query, *selected)
    if len(selected) > max_cells:
        raise ValueError('complete radius neighborhood exceeds max_cells; truncation is forbidden')

    nodes = []
    roles = []
    withheld_query_features = {(modality, feature) for modality, feature, _ in target_axis}
    for i, row in enumerate(selected):
        values = [Measurement('is_query', i == 0, 'dimensionless', 'observed', True),
                  Measurement('organism', row.organism, None, 'observed', True),
                  Measurement('cell_type', row.cell_type, None,
                              'observed' if row.cell_type is not None else 'missing', row.cell_type is not None),
                  Measurement('baseline_time', row.time_value, row.time_unit, 'observed', True),
                  Measurement('interventions', canonical(row.intervention_ids).decode(), None, 'observed', True)]
        values.extend(Measurement(f'coordinate:{j}', value, row.coordinate_unit, 'observed', True)
                      for j, value in enumerate(row.coordinates))
        for item in row.measurements:
            key = canonical((item.modality, item.feature_id)).decode()
            # Keep all original source records intact, but redact the exact
            # supervised query coordinate from predictive spatial inputs.
            withheld = i == 0 and (item.modality, item.feature_id) in withheld_query_features
            available = item.observed and not withheld
            status = 'predicted' if row.source.evidence_kind == 'model_prediction' else 'observed'
            values.append(Measurement('native:' + key, None if withheld else item.value, item.unit,
                                      status if available else 'missing', available))
            values.append(Measurement('observed:' + key, available, 'dimensionless', 'observed', True))
        if i == 0:
            values.append(Measurement('requested_axis', canonical(target_axis).decode(), None, 'observed', True))
            values.append(Measurement('radius', radius, row.coordinate_unit, 'observed', True))
            values.extend(Measurement(f'target:{j}', None, unit, 'missing', False)
                          for j, (_, _, unit) in enumerate(target_axis))
        nodes.append(Node('scri-cell', tuple(values)))
        roles.extend((('node', i, m.name), 'target' if m.name.startswith('target:')
                      else 'condition' if m.available else 'unused') for m in values)

    edges = []
    for i, left in enumerate(selected):
        for j in range(i + 1, len(selected)):
            distance = math.dist(left.coordinates, selected[j].coordinates)
            number(distance, 'baseline edge distance')
            if distance <= radius:
                measurements = (Measurement('distance', distance, query.coordinate_unit, 'observed', True),)
                for source, target in ((i, j), (j, i)):
                    edges.append(Edge(source, target, 'baseline-radius-neighbor', True, measurements))
                    roles.append((('edge', len(edges) - 1, 'distance'), 'condition'))
    sources = tuple(dict.fromkeys(Provenance(
        row.source.dataset_id, row.source.revision, row.source.source_locator,
        row.source.artifact_sha256, row.source.license_id) for row in selected))
    record = GraphRecord('scri-neighborhood-query', tuple(nodes), tuple(edges), sources)
    task = ConditionalTask(tuple(roles), topology_known=True)
    audit = {
        'schema': 'Mora3.SCRISpatialQuery/v1', 'query_record_id': query_record_id,
        'node_record_ids': tuple(row.record_id for row in selected),
        'sources': tuple(asdict(row.source) for row in selected),
        'candidate_record_count': len(rows), 'selected_record_count': len(selected),
        'coordinate_frame': query.coordinate_frame, 'coordinate_unit': query.coordinate_unit,
        'time_reference': query.time_reference, 'radius': radius,
        'adjacency': 'baseline_closed_radius_reciprocal_edges_not_causal',
        'training_admitted': False, 'source_admission': 'external',
    }
    return validate_spatial_query(SpatialQuery(record, task, target_axis, audit))
