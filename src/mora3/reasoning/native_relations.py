"""Native condition-only directed relations and categorical naturality checks.

GraphRecord topology and relation kinds must already be legitimate conditioning
information. A signed regulatory relation is kept as a relation label; it is not
converted into a negative metric or inferred physical3D location. A source hash
binds provenance, not experimental causality. No2-cell is invented from a graph
cycle: caller-supplied squares carry separate immutable evidence.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import re

from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from .directed_topology import Arrow, Square, DirectedComplex, DirectedFunctor


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit immutable evidence digest required')


def relation_complex(record, task, *, source_sha256, cost_field, cost_unit, squares=()):
    """Project only known directed topology and observed native edge costs.

    This adds no transitive shortcut edge, converse relation, commutative square,
    chemical join or causal assumption. Vertices/edges use stable local positions;
    source-record IDs and unavailable response targets cannot influence the graph.
    """
    if type(record) is not GraphRecord or type(task) is not ConditionalTask:
        raise ValueError('Exact native record and explicit task required')
    _sha(source_sha256)
    if source_sha256 not in {p.sha256 for p in record.provenance}:
        raise ValueError('Directed presentation source does not match record provenance')
    if any(type(x) is not str or not x.strip() for x in (cost_field, cost_unit)):
        raise ValueError('Explicit native cost field and unit required')
    fields = task._fields(record)
    task.input_fields(record)
    roles = dict(task.roles)
    arrows = []
    for i, edge in enumerate(record.edges):
        if not edge.directed:
            raise ValueError('Undirected relations need a separately declared directed encoding')
        ref = ('edge', i, cost_field)
        if ref not in fields or roles[ref] != 'condition':
            raise ValueError('Native relation cost must be an explicit condition, never a target')
        cost = fields[ref]
        if cost.status != 'observed' or not cost.available or cost.units != cost_unit:
            raise ValueError('Complete observed cost in the declared native unit required')
        arrows.append(Arrow(f'edge:{i}', f'node:{edge.source}', f'node:{edge.target}',
                            edge.kind, cost.value, source_sha256))
    if type(squares) is not tuple or any(type(s) is not Square for s in squares):
        raise ValueError('Explicit immutable evidence-bound squares required')
    result = DirectedComplex(tuple(f'node:{i}' for i in range(len(record.nodes))),
                             tuple(arrows), squares, cost_unit)
    wire = json.dumps(dict(presentation=asdict(result), node_kinds=[n.kind for n in record.nodes]),
                      sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return result, dict(schema='Mora3.NativeDirectedCondition/v1', condition_sha256=sha256(wire).hexdigest(),
        source_sha256=source_sha256, causal_authority_issued=False,
        coordinate_geometry_inferred=False, automatically_filled_cycles=0,
        topology_role='known input; not evidence for unknown-topology generation')


def verify_natural_transformation(first, second, components, evidence_sha256, *, max_states=20_000):
    """Check eta_y o F(a) = G(a) o eta_x on every source generator.

    For checked presentation functors this extends by path composition. Grandis
    identifies categorical directed homotopies with natural transformations;
    this verifier does not claim realization as a homotopy of biological spaces.
    A noninvertible eta is allowed; a natural isomorphism is NOT asserted.
    """
    _sha(evidence_sha256)
    if (type(first) is not DirectedFunctor or type(second) is not DirectedFunctor
            or first.source != second.source or first.target != second.target
            or type(components) is not dict or set(components) != set(first.source.vertices)):
        raise ValueError('Two checked functors with common source/target and complete components required')
    statuses = [first.verify(max_states=max_states)['status'], second.verify(max_states=max_states)['status']]
    target = first.target
    for vertex, path in components.items():
        if target.endpoint(first.objects[vertex], path) != second.objects[vertex]:
            raise ValueError('Naturality component has incorrect endpoints')
    proofs = []
    for arrow in first.source.arrows:
        left = target.compose(first.objects[arrow.source], first.arrows[arrow.name], components[arrow.target])
        right = target.compose(first.objects[arrow.source], components[arrow.source], second.arrows[arrow.name])
        proof = target.equivalent(first.objects[arrow.source], left, right, max_states=max_states)
        proofs.append(dict(generator=arrow.name, **asdict(proof)))
        statuses.append('proved' if proof.status == 'proved' else 'unknown' if proof.status == 'unknown-budget' else 'refused')
    status = 'refused' if 'refused' in statuses else 'unknown' if 'unknown' in statuses else 'proved'
    return dict(status=status, generator_naturality_proofs=tuple(proofs), evidence_sha256=evidence_sha256,
        natural_isomorphism_proved=False, continuous_homotopy_realization_proved=False,
        biological_intervention_equivalence_proved=False)
