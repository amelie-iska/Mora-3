"""Geometry-aware analogy objectives over the actual existing shared encoder.

No learned parameters, optimizer or alternative backbone are introduced. Physical
length units and normalization are explicit source-bound choices. Geometric
losses are rigid-invariant; the existing scalar payload encoder itself is NOT
thereby made SE(3)-equivariant. Distance-only topology cannot distinguish mirrors;
ordered signed-volume constraints are provided separately for chiral contexts.

Directed regulatory relations use typed transports/directed presentations, not
invented3D coordinates or indiscriminate Laplacian smoothing. The sheaf-inspired
energy below accepts externally declared restriction maps; it does not claim to
implement the learned Neural Sheaf Diffusion architecture.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
import torch

from .persistent_topology import topological_metric_loss, checked_distances


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit immutable source/scale evidence digest required')


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Finite strictly positive native scale required')


def _points(value):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or not 1 <= value.shape[0] <= 128
            or not 1 <= value.shape[1] <= 4096 or not torch.isfinite(value).all()):
        raise ValueError('Complete bounded finite real coordinate/embedding matrix required')


def physical_distances(points, *, scale):
    """Known-unit Euclidean distances divided by a declared frozen length scale."""
    _points(points); _positive(scale)
    delta = points.double()[:, None, :] - points.double()[None, :, :]
    value = torch.linalg.vector_norm(delta, dim=-1) / scale
    if not torch.isfinite(value).all():
        raise ValueError('Unrepresentable geometric distances')
    return value


def torus_distances(angles, *, periods):
    """Product geodesic distance on explicitly periodic native coordinates.

    Periods have the same units as the corresponding coordinates. The result
    is piecewise differentiable; wrap cut loci are not smooth. No observed
    molecular torsion identity or conformer correspondence is inferred.
    """
    _points(angles)
    if type(periods) is not tuple or len(periods) != angles.shape[1]:
        raise ValueError('One native period per ordered angle coordinate required')
    for p in periods:
        _positive(p)
    period = angles.new_tensor(periods, dtype=torch.float64)
    delta = angles.double()[:, None, :] - angles.double()[None, :, :]
    wrapped = torch.remainder(delta + period/2, period) - period/2
    return torch.linalg.vector_norm(wrapped, dim=-1)


def signed_volume(points, ordered_indices):
    """Oriented tetrahedron volume in native cubic units, preserving chirality.

    Atom/landmark order is part of the source contract; odd relabeling flips the
    sign. This is not a canonical chirality classifier for an unlabeled molecule.
    """
    _points(points)
    if (points.shape[1] != 3 or type(ordered_indices) is not tuple
            or len(ordered_indices) != 4 or len(set(ordered_indices)) != 4
            or any(type(i) is not int or not 0 <= i < len(points) for i in ordered_indices)):
        raise ValueError('Four distinct ordered native3D anchors required')
    a, b, c, d = points.double()[list(ordered_indices)]
    return torch.dot(b-a, torch.linalg.cross(c-a, d-a)) / 6


@dataclass(frozen=True)
class GeometryPolicy:
    source_sha256: str
    domain: str
    coordinate_names: tuple[str, ...]
    coordinate_unit: str
    frame_field: str
    coordinate_frame: str
    length_scale: float
    scale_evidence_sha256: str
    max_nodes: int = 64
    max_simplices: int = 50_000
    max_entities: int = 4097
    max_attention_elements: int = 68_000_000
    topology_weight: float = 1.
    metric_weight: float = 1.

    def __post_init__(self):
        _sha(self.source_sha256); _sha(self.scale_evidence_sha256)
        if self.domain not in ('molecular_structure', 'protein_structure', 'cell_spatial_geometry', 'morphology_landmarks'):
            raise ValueError('Only declared physical geometry domains; GRNs need directed relational structure')
        if (type(self.coordinate_names) is not tuple or len(self.coordinate_names) not in (2, 3)
                or len(set(self.coordinate_names)) != len(self.coordinate_names)):
            raise ValueError('Complete distinct2D/3D coordinate names required')
        for value in (*self.coordinate_names, self.coordinate_unit, self.frame_field, self.coordinate_frame):
            if type(value) is not str or not value.strip():
                raise ValueError('Explicit native coordinate/unit/frame fields required')
        if self.frame_field in self.coordinate_names:
            raise ValueError('Coordinate and reference-frame fields must be distinct')
        _positive(self.length_scale)
        for x in (self.max_nodes, self.max_simplices, self.max_entities, self.max_attention_elements):
            if type(x) is not int or x < 1:
                raise ValueError('Explicit positive integer geometry resource budgets required')
        if self.max_nodes > 128 or self.max_simplices > 100_000:
            raise ValueError('Exact geometry/topology reference implementation bound exceeded')
        for weight in (self.topology_weight, self.metric_weight):
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise ValueError('Frozen finite nonnegative objective weights required')
        if self.topology_weight + self.metric_weight == 0:
            raise ValueError('At least one explicit geometric objective must be active')

    def binding(self):
        self.__post_init__()
        return dict(schema='Mora3.GeometryAnalogyPolicy/v2', **asdict(self),
            loss_normalization='mean over native pairs/critical edges; latent RMS across width',
            pairings='detached piecewise Rips-F2-H0-H1',
            tie_policy='all-edges-at-persistence-critical-levels',
            adds_learned_parameters=False, biological_causality_claim=False)


def _observed_geometry(record, task, policy):
    from mora3.data.schema import GraphRecord
    from mora3.data.tasks import ConditionalTask
    if type(record) is not GraphRecord or type(task) is not ConditionalTask or type(policy) is not GeometryPolicy:
        raise ValueError('Exact native record/task and source-bound geometry policy required')
    policy.__post_init__()
    fields = task._fields(record)
    roles = dict(task.roles)
    task.input_fields(record)
    if policy.source_sha256 not in {p.sha256 for p in record.provenance}:
        raise ValueError('Geometry source does not match declared provenance')
    if len(record.nodes) > policy.max_nodes:
        raise ValueError('Complete geometry exceeds node budget; no landmark subsampling')
    rows = []
    for i in range(len(record.nodes)):
        names = (*policy.coordinate_names, policy.frame_field)
        selected = []
        for name in names:
            ref = ('node', i, name)
            if ref not in fields or roles.get(ref) != 'condition':
                raise ValueError('Geometric reference may use only explicit observed condition fields')
            m = fields[ref]
            if m.status != 'observed' or not m.available:
                raise ValueError('Predicted/unknown geometry cannot masquerade as observed reference')
            selected.append(m)
        *coordinates, frame = selected
        if frame.value != policy.coordinate_frame:
            raise ValueError('Mixed coordinate frames require an explicit source transform')
        if any(m.units != policy.coordinate_unit or type(m.value) not in (int, float) for m in coordinates):
            raise ValueError('Complete native coordinate units and numeric values required')
        rows.append(tuple(m.value for m in coordinates))
    condition = dict(nodes=[dict(kind=n.kind, measurements=[asdict(m) for m in n.measurements
                    if roles[('node', i, m.name)] in ('condition', 'desired')]) for i, n in enumerate(record.nodes)],
        edges=[dict(source=e.source, target=e.target, kind=e.kind, directed=e.directed,
                    measurements=[asdict(m) for m in e.measurements if roles[('edge', i, m.name)] in ('condition', 'desired')])
               for i, e in enumerate(record.edges)], policy=policy.binding())
    wire = json.dumps(condition, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return rows, sha256(wire).hexdigest()


def geometry_objective(conditioner, record, task, policy):
    """Diagnostic/auxiliary forward on the literal shared TokenGT node states.

    Admission belongs to the native training entry below. This utility performs
    no optimizer steps, initializes no weights, and adds no parameters. Physical
    geometry is encoded by the existing payload path; this function regularizes
    its node states and does not replace geometry with a text-only similarity.
    """
    from mora3.model.geometry_conditioning import GeometryConditioner
    if (type(conditioner) is not GeometryConditioner
            or conditioner.parameterization_policy != 'mup-geometry-v1'):
        raise ValueError('Exact explicitly parameterized native GeometryConditioner required; no legacy migration')
    conditioner.get_extra_state()
    rows, condition_sha = _observed_geometry(record, task, policy)
    count = len(rows)
    if count > conditioner.config['id_width']:
        raise ValueError('Geometry exceeds exact native identifier capacity')
    tokens = 1 + count + len(record.edges)
    native = conditioner.graph.configuration()
    if tokens > policy.max_entities or native['heads'] * native['layers'] * tokens * tokens > policy.max_attention_elements:
        raise ValueError('Full native graph exceeds declared attention/entity budget')
    weight = conditioner.graph.feature.weight
    positions = weight.new_tensor(rows, dtype=torch.float64)
    reference = physical_distances(positions, scale=policy.length_scale)
    # Refuse combinatorial overflow before neural work.
    from .persistent_topology import rips_persistence
    rips_persistence(reference.cpu().numpy(), max_vertices=policy.max_nodes, max_simplices=policy.max_simplices)
    ids = torch.eye(count, conditioner.config['id_width'], dtype=weight.dtype, device=weight.device)
    encoded = conditioner.payload.forward_task(record, task=task, identifiers=ids)
    node_states = conditioner.graph([encoded])[0][0, 1:1+count]
    latent = physical_distances(node_states, scale=math.sqrt(node_states.shape[1]))
    topological = topological_metric_loss(reference, latent,
        max_vertices=policy.max_nodes, max_simplices=policy.max_simplices)
    i, j = torch.triu_indices(count, count, offset=1, device=weight.device)
    metric = (latent[i, j] - reference[i, j]).square().mean() if len(i) else latent.sum()*0
    loss = policy.topology_weight * topological['loss'] + policy.metric_weight * metric
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite geometric auxiliary objective')
    return dict(loss=loss, topology_loss=topological['loss'], metric_loss=metric,
        node_embeddings=node_states, latent_metric='RMS-coordinate-distance',
        reference_persistence=topological['reference'], latent_persistence=topological['latent'],
        observed_pairs=len(i), condition_sha256=condition_sha,
        latent_variance=node_states.double().var(dim=0, unbiased=False).mean(),
        collapse_detected=bool(len(i) and not torch.any(latent[i, j] > 0)),
        new_learned_parameters=0, training_admission=False,
        se3_equivariant_encoder_claim=False, reconstruction_or_reasoning_quality_proved=False)


def relational_distortion(source_distances, target_distances, correspondence):
    """Partial fixed-correspondence metric distortion, NOT inferred GW transport.

    Explicit normalization/physical unit conversion must precede this call.
    Supports non-additive mappings: no global embedding-offset assumption.
    """
    for d in (source_distances, target_distances):
        if not isinstance(d, torch.Tensor) or not d.is_floating_point():
            raise ValueError('Native real distance tensors required')
        checked_distances(d.detach().cpu().numpy(), max_vertices=128)
    if source_distances.device != target_distances.device:
        raise ValueError('Distance devices differ')
    if (type(correspondence) is not tuple or len(correspondence) < 2
            or any(type(p) is not tuple or len(p) != 2 or any(type(i) is not int for i in p) for p in correspondence)):
        raise ValueError('At least two explicit ordered anchor pairs required')
    a, b = zip(*correspondence)
    if (len(set(a)) != len(a) or len(set(b)) != len(b)
            or min(a) < 0 or max(a) >= len(source_distances)
            or min(b) < 0 or max(b) >= len(target_distances)):
        raise ValueError('Injective known partial correspondence required')
    i, j = torch.triu_indices(len(a), len(a), offset=1, device=source_distances.device)
    da = source_distances[list(a)][:, list(a)].double()
    db = target_distances[list(b)][:, list(b)].double()
    return (da[i, j]-db[i, j]).square().mean()


def relation_transport_energy(states, edges, source_restrictions, target_restrictions):
    """Mean sheaf-style residual ||R_u z_u - R_v z_v||^2 per edge/coordinate.

    Maps are explicit caller-owned operators. A negative regulation sign belongs
    to the relation/map; ordinary smoothing z_u=z_v is not silently imposed.
    Learned maps, if later introduced, need their own complete μP qualification.
    """
    _points(states)
    n, width = states.shape
    if (type(edges) is not tuple or not edges or len(edges) > 512
            or any(type(e) is not tuple or len(e) != 2 or any(type(i) is not int or not 0 <= i < n for i in e) for e in edges)):
        raise ValueError('Complete bounded directed edge indices required')
    for r in (source_restrictions, target_restrictions):
        if (not isinstance(r, torch.Tensor) or r.shape != (len(edges), width, width)
                or r.dtype != states.dtype or r.device != states.device or not torch.isfinite(r).all()):
            raise ValueError('Matching finite source/target restriction maps required')
    u, v = zip(*edges)
    left = torch.bmm(source_restrictions.double(), states[list(u)].double().unsqueeze(-1))
    right = torch.bmm(target_restrictions.double(), states[list(v)].double().unsqueeze(-1))
    return (left-right).square().mean()


def geometry_regularized_native_loss(owner, native_objective, record, task, policy, *, geometry_module_path):
    """Opt-in composition with an existing admitted native objective.

    This is the training entry: it cannot grant readiness or create a run. The
    source schedule must separately freeze this loss/policy and its coefficients
    across proxy/target widths. Width normalization is not a proof of μTransfer.
    """
    from mora3.model.full_owner import SourceBoundOwner
    if type(owner) is not SourceBoundOwner:
        raise ValueError('Exact native source-bound owner required')
    owner.require_training_admission()
    if not callable(native_objective) or type(geometry_module_path) is not str:
        raise ValueError('Existing native objective and explicit registered geometry path required')
    modules = dict(owner.named_modules())
    if geometry_module_path not in modules:
        raise ValueError('Geometry branch is not registered in the primary owner')
    geometric = geometry_objective(modules[geometry_module_path], record, task, policy)
    native = native_objective(owner)
    if not isinstance(native, torch.Tensor) or native.ndim or not torch.isfinite(native):
        raise ValueError('Finite scalar native objective required')
    return dict(loss=native+geometric['loss'], native_loss=native, geometry=geometric)
