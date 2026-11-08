"""Bounded explicit-map rigid geometry; no chemistry/source/model qualification.

Coordinates are float64 in angstrom. Column-vector transform: y=R@x+t.
Weights follow the caller's pair order. R is proper SO(3); no scale or permutation.
Pose uses only the protein-anchor fit, never an independent ligand alignment.
"""
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

MAX_POINTS = 128
_BACKBONE = frozenset(("N", "CA", "C", "O", "OXT"))

def _text(value, name):
    if type(value) is not str or not value or value.isspace():
        raise ValueError(name + " must be an exact nonempty literal string")

def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " outside declared integer bound")

def _number(value, name, positive=False):
    if type(value) not in (int, float):
        raise ValueError(name + " requires a real scalar, not bool/type aliases")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + " outside finite float64 range") from None
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(name + " outside finite permitted range")
    return result

class Coverage(str, Enum):
    FULL = "full_supplied_axes"
    PARTIAL_REFERENCE = "explicit_partial_reference"

class Selection(str, Enum):
    ALL = "all_explicit_atoms"
    BACKBONE = "compatible_backbone"

class Status(str, Enum):
    PASS = "PASS"
    VIOLATED = "VIOLATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"

class Metric(str, Enum):
    TS_SHAPE = "TS_INTERNAL_SO3_RMSD"
    ENZYME_TEMPLATE = "ENZYME_TEMPLATE_SO3_RMSD"
    TS_POSE = "TS_POSE_AFTER_PROTEIN_ANCHOR_SO3"

@dataclass(frozen=True)
class AtomID:
    entity: str
    ordinal: int
    element: str
    atom_name: str
    residue_key: str | None = None
    residue_name: str | None = None
    isotope: int | None = None

    def __post_init__(self):
        _text(self.entity, "entity"); _integer(self.ordinal, "ordinal")
        _text(self.element, "element"); _text(self.atom_name, "atom_name")
        if (self.residue_key is None) != (self.residue_name is None):
            raise ValueError("residue identity must be supplied together")
        if self.residue_key is not None:
            _text(self.residue_key, "residue_key"); _text(self.residue_name, "residue_name")
        if self.isotope is not None:
            _integer(self.isotope, "isotope", 1)

    @property
    def key(self):
        return self.entity, self.ordinal

def _array(value, shape, name):
    if type(value) is not np.ndarray or value.dtype != np.dtype("float64") or value.shape != shape:
        raise ValueError(name + " requires exact float64 ndarray/shape")
    if not bool(np.isfinite(value).all()):
        raise ValueError(name + " contains nonfinite values")

@dataclass(frozen=True, eq=False)
class PointSet:
    ids: tuple[AtomID, ...]
    xyz: np.ndarray
    frame: str
    max_points: int
    units: str = "angstrom"

    def __post_init__(self):
        _integer(self.max_points, "max_points", 1, MAX_POINTS)
        if type(self.ids) is not tuple or not 1 <= len(self.ids) <= self.max_points:
            raise ValueError("complete nonempty point axis exceeds explicit budget")
        if any(type(a) is not AtomID for a in self.ids):
            raise ValueError("exact AtomID axis required")
        for a in self.ids:
            a.__post_init__()
        if len({a.key for a in self.ids}) != len(self.ids):
            raise ValueError("duplicate native atom identity")
        _text(self.frame, "frame")
        if type(self.units) is not str or self.units != "angstrom":
            raise ValueError("only exact angstrom units supported; no conversion")
        _array(self.xyz, (len(self.ids), 3), "xyz")
        # Immutable byte backing: external mutation and writeable-flag changes cannot
        # change this point snapshot; caller's array/order/flags remain untouched.
        frozen = np.frombuffer(self.xyz.tobytes(order="C"), dtype=np.float64).reshape((-1, 3))
        object.__setattr__(self, "xyz", frozen)

@dataclass(frozen=True)
class Pair:
    source: AtomID
    reference: AtomID

    def __post_init__(self):
        if type(self.source) is not AtomID or type(self.reference) is not AtomID:
            raise ValueError("explicit exact atom identities required")
        self.source.__post_init__(); self.reference.__post_init__()

@dataclass(frozen=True)
class Mapping:
    pairs: tuple[Pair, ...]
    coverage: Coverage

    def __post_init__(self):
        if type(self.coverage) is not Coverage:
            raise ValueError("explicit coverage mode required")
        if type(self.pairs) is not tuple or not 1 <= len(self.pairs) <= MAX_POINTS:
            raise ValueError("nonempty bounded ordered mapping required")
        if any(type(p) is not Pair for p in self.pairs):
            raise ValueError("exact Pair entries required")
        for p in self.pairs:
            p.__post_init__()
        if len({p.source.key for p in self.pairs}) != len(self.pairs) or len({p.reference.key for p in self.pairs}) != len(self.pairs):
            raise ValueError("ambiguous/duplicate correspondence")

@dataclass(frozen=True)
class Numerics:
    rank_rtol: float
    orthogonality_atol: float

    def __post_init__(self):
        r = _number(self.rank_rtol, "rank_rtol", True)
        o = _number(self.orthogonality_atol, "orthogonality_atol", True)
        if r >= 1 or o >= .01:
            raise ValueError("unsupported numerical guard; never RMSD acceptance slack")

@dataclass(frozen=True)
class MappedCoverage:
    source_total: int
    reference_total: int
    source_count: int
    reference_count: int
    source_selected: tuple[AtomID, ...]
    reference_selected: tuple[AtomID, ...]
    source_omitted: tuple[AtomID, ...]
    reference_omitted: tuple[AtomID, ...]
    full_source: bool
    scope: str

@dataclass(frozen=True)
class RigidTransform:
    rotation: tuple[tuple[float, ...], ...]
    translation: tuple[float, ...]
    source_frame: str
    reference_frame: str

@dataclass(frozen=True)
class FitResult:
    rmsd: float
    transform: RigidTransform
    rank: int
    rotation_unique: bool
    weights: tuple[float, ...]
    coverage: MappedCoverage
    singular_values: tuple[float, ...]
    singular_values_basis: str
    numerical_guards: Numerics

@dataclass(frozen=True)
class Evaluation:
    metric: Metric
    status: Status
    threshold: float | None
    rmsd: float | None
    transform: RigidTransform | None
    coverage: MappedCoverage | None
    weights: tuple[float, ...]
    reason: str
    fit: FitResult | None = None
    anchor_fit: FitResult | None = None
    units: str = "angstrom"
    source_or_model_authority: bool = field(default=False, init=False)

    def __post_init__(self):
        if type(self.metric) is not Metric or type(self.status) is not Status or self.units != "angstrom":
            raise ValueError("exact metric/status/units required")
        if self.source_or_model_authority is not False:
            raise ValueError("geometric residual cannot grant authority")
        _text(self.reason, "reason")
        if self.threshold is not None:
            _number(self.threshold, "threshold")
        if self.status is Status.NOT_EVALUABLE:
            if self.rmsd is not None or self.transform is not None:
                raise ValueError("unknown residual/pose must not appear evaluable")
        else:
            if self.rmsd is None or self.threshold is None or type(self.transform) is not RigidTransform or type(self.coverage) is not MappedCoverage:
                raise ValueError("evaluable result requires complete residual/transform/coverage")
            residual = _number(self.rmsd, "rmsd")
            expected = Status.PASS if residual <= self.threshold else Status.VIOLATED
            if self.status is not expected:
                raise ValueError("status must match strict unexpanded caller threshold")
            if self.metric is Metric.TS_POSE:
                if type(self.anchor_fit) is not FitResult or not self.anchor_fit.rotation_unique or self.anchor_fit.rank < 2:
                    raise ValueError("protein frame underdetermined")
            elif type(self.fit) is not FitResult:
                raise ValueError("shape/template needs explicit fit evidence")

    def require_source_or_model_use(self):
        raise RuntimeError("geometry-only evidence: native/source/model/training authority closed")

@dataclass(frozen=True)
class Requirement:
    result: Evaluation
    required: bool

    def __post_init__(self):
        if type(self.result) is not Evaluation or type(self.required) is not bool:
            raise ValueError("exact result and hard/optional boolean required")
        self.result.__post_init__()

@dataclass(frozen=True)
class OverallReport:
    status: Status
    requirements: tuple[Requirement, ...]
    source_or_model_authority: bool = field(default=False, init=False)

def overall_status(requirements):
    if type(requirements) is not tuple or len(requirements) > 128 or any(type(r) is not Requirement for r in requirements):
        raise ValueError("bounded literal requirement tuple required")
    for r in requirements:
        r.__post_init__()
    hard = [r.result.status for r in requirements if r.required]
    if Status.VIOLATED in hard:
        status = Status.VIOLATED
    elif not hard or Status.NOT_EVALUABLE in hard:
        status = Status.NOT_EVALUABLE
    else:
        status = Status.PASS
    return OverallReport(status, requirements)

def _point_contract(point):
    if type(point) is not PointSet:
        raise ValueError("exact PointSet required")
    _integer(point.max_points, "max_points", 1, MAX_POINTS)
    if not 1 <= len(point.ids) <= point.max_points or point.units != "angstrom":
        raise ValueError("point budget/units")
    _array(point.xyz, (len(point.ids), 3), "xyz")
    _text(point.frame, "frame")

def _prepare(source, reference, mapping, weights, numerics, selection):
    _point_contract(source); _point_contract(reference)
    if type(mapping) is not Mapping or type(numerics) is not Numerics or type(selection) is not Selection:
        raise ValueError("exact mapping/numerical/selection declarations required")
    mapping.__post_init__(); numerics.__post_init__()
    _array(weights, (len(mapping.pairs),), "weights")
    if not bool((weights > 0).all()):
        raise ValueError("weights must be finite and strictly positive")
    sx = {a.key: (a, i) for i, a in enumerate(source.ids)}
    ry = {a.key: (a, i) for i, a in enumerate(reference.ids)}
    xi, yi = [], []
    for pair in mapping.pairs:
        sa, ra = pair.source, pair.reference
        if sa.key not in sx or ra.key not in ry or sx[sa.key][0] != sa or ry[ra.key][0] != ra:
            raise ValueError("unbound exact native identity; no implicit intersection")
        if sa.element != ra.element or sa.isotope != ra.isotope:
            raise ValueError("incompatible explicit element/isotope comparison")
        if (sa.residue_name is None) != (ra.residue_name is None):
            raise ValueError("unknown residue compatibility")
        if sa.residue_name is not None:
            backbone = sa.atom_name == ra.atom_name and sa.atom_name in _BACKBONE
            if sa.residue_name != ra.residue_name and not backbone:
                raise ValueError("mutated sidechain comparison incompatible")
            if selection is Selection.BACKBONE and not backbone:
                raise ValueError("explicit backbone mapping required")
        elif selection is Selection.BACKBONE:
            raise ValueError("backbone selection lacks residue evidence")
        xi.append(sx[sa.key][1]); yi.append(ry[ra.key][1])
    sids = tuple(p.source for p in mapping.pairs)
    rids = tuple(p.reference for p in mapping.pairs)
    skeys, rkeys = {a.key for a in sids}, {a.key for a in rids}
    if rkeys != set(ry) or (mapping.coverage is Coverage.FULL and skeys != set(sx)):
        raise ValueError("coverage mode requires every reference and declared full source")
    cov = MappedCoverage(len(source.ids), len(reference.ids), len(sids), len(rids), sids, rids,
        tuple(a for a in source.ids if a.key not in skeys),
        tuple(a for a in reference.ids if a.key not in rkeys),
        len(skeys) == len(sx), mapping.coverage.value)
    raw = tuple(float(w) for w in weights)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        q = weights / float(np.max(weights))
        if bool((q <= 0).any()):
            raise ValueError("weight dynamic range cannot be represented without dropping atoms")
        q = q / float(np.sum(q))
        if bool((q <= 0).any()):
            raise ValueError("weight normalization would drop mapped atoms")
    return source.xyz[xi].copy(), reference.xyz[yi].copy(), q, raw, cov

def _finite(array, name):
    if not bool(np.isfinite(array).all()):
        raise ValueError(name + " exceeds supported finite numerical range")

def _scaled(array):
    scale = float(np.max(np.abs(array)))
    return array if scale == 0 else array / scale

def _rank(matrix, numerics):
    singular = np.linalg.svd(_scaled(matrix), compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0
    return int(np.count_nonzero(singular > singular[0] * numerics.rank_rtol))

def _rmsd(residual, q):
    _finite(residual, "residual")
    scale = float(np.max(np.abs(residual)))
    if scale == 0:
        return 0.
    norm = residual / scale
    mean = float(np.sum(q * np.sum(norm * norm, axis=1)))
    if not math.isfinite(mean) or mean < 0:
        raise ValueError("weighted residual outside numerical range")
    result = scale * math.sqrt(mean)
    if not math.isfinite(result) or (result == 0 and bool((residual != 0).any())):
        raise ValueError("residual under/overflow; cannot report false zero")
    return result

def _transform(rotation, translation, source_frame, reference_frame):
    return RigidTransform(tuple(tuple(float(v) for v in row) for row in rotation),
        tuple(float(v) for v in translation), source_frame, reference_frame)

def proper_fit(source, reference, mapping, weights, *, numerics, selection=Selection.ALL):
    """Weighted proper Procrustes on explicit pairs; singular values are scale-normalized.

    Positive coordinate-scale normalization in covariance is only a numerical
    operation and cannot change R or introduce a geometric scaling degree of freedom.
    The final residual is evaluated in the original supplied angstrom coordinates.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            x, y, q, raw, cov = _prepare(source, reference, mapping, weights, numerics, selection)
            cx, cy = q @ x, q @ y
            a, b = x - cx, y - cy
            _finite(a, "centered source"); _finite(b, "centered reference")
            an, bn = _scaled(a), _scaled(b)
            root = np.sqrt(q)[:, None]
            h = an.T @ (q[:, None] * bn)
            u, s, vt = np.linalg.svd(h)
            determinant = float(np.linalg.det(vt.T @ u.T))
            d = np.eye(3, dtype=np.float64)
            if determinant < 0:
                d[2, 2] = -1.
            rotation = vt.T @ d @ u.T
            rank = min(_rank(an * root, numerics), _rank(bn * root, numerics), _rank(h, numerics))
            nonunique_reflection = determinant < 0 and float(s[1] - s[2]) <= float(s[0]) * numerics.rank_rtol
            unique = rank >= 2 and not nonunique_reflection
            # Exact constant translation has a canonical identity representative.
            # This is exact equality, not residual snapping/tolerance acceptance.
            delta = y - x
            _finite(delta, "point differences")
            if bool(np.all(delta == delta[0])):
                rotation = np.eye(3, dtype=np.float64)
                translation = delta[0].copy()
            else:
                translation = cy - rotation @ cx
            _finite(rotation, "rotation"); _finite(translation, "translation")
            if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=numerics.orthogonality_atol) or abs(float(np.linalg.det(rotation)) - 1.) > numerics.orthogonality_atol:
                raise ValueError("proper rotation numerical verification failed")
            residual = x @ rotation.T + translation - y
            rmsd = _rmsd(residual, q)
            return FitResult(rmsd, _transform(rotation, translation, source.frame, reference.frame),
                rank, bool(unique), raw, cov, tuple(float(v) for v in s),
                "weighted_covariance_of_separately_maxabs_centered_axes", numerics)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("unsupported finite-precision geometry: " + type(exc).__name__) from None

def _threshold(value):
    return None if value is None else _number(value, "caller RMSD threshold")

def _not_evaluable(metric, threshold, reason, *, anchor_fit=None, coverage=None, weights=()):
    return Evaluation(metric, Status.NOT_EVALUABLE, threshold, None, None, coverage, weights,
        reason, anchor_fit=anchor_fit)

def _shape(metric, source, reference, mapping, weights, threshold, numerics, selection):
    _point_contract(source)
    if type(numerics) is not Numerics:
        raise ValueError("numerics required")
    numerics.__post_init__()
    limit = _threshold(threshold)
    if reference is None:
        return _not_evaluable(metric, limit, "reference_unavailable")
    fit = proper_fit(source, reference, mapping, weights, numerics=numerics, selection=selection)
    if limit is None:
        return _not_evaluable(metric, None, "caller_threshold_unavailable", coverage=fit.coverage, weights=fit.weights)
    status = Status.PASS if fit.rmsd <= limit else Status.VIOLATED
    return Evaluation(metric, status, limit, fit.rmsd, fit.transform, fit.coverage, fit.weights,
        "explicit_mapped_subset_only" if not fit.coverage.full_source else "complete_supplied_axes_only", fit=fit)

def evaluate_shape(source, reference, mapping, weights, threshold, *, numerics, selection=Selection.ALL):
    return _shape(Metric.TS_SHAPE, source, reference, mapping, weights, threshold, numerics, selection)

def evaluate_template(source, reference, mapping, weights, threshold, *, numerics, selection=Selection.ALL):
    return _shape(Metric.ENZYME_TEMPLATE, source, reference, mapping, weights, threshold, numerics, selection)

def evaluate_pose(ligand, reference_ligand, source_anchors, reference_anchors,
                  ligand_mapping, anchor_mapping, ligand_weights, anchor_weights,
                  threshold, *, numerics, anchor_selection=Selection.ALL):
    """Docking RMSD after the protein-anchor transform only; no ligand fit."""
    _point_contract(ligand); _point_contract(source_anchors)
    limit = _threshold(threshold)
    if type(numerics) is not Numerics:
        raise ValueError("numerics required")
    numerics.__post_init__()
    if ligand.frame != source_anchors.frame:
        raise ValueError("ligand and source protein anchors require the same declared frame")
    if reference_ligand is None or reference_anchors is None:
        return _not_evaluable(Metric.TS_POSE, limit, "pose_or_anchor_reference_unavailable")
    _point_contract(reference_ligand); _point_contract(reference_anchors)
    if reference_ligand.frame != reference_anchors.frame:
        raise ValueError("reference ligand/protein frame mismatch")
    anchor_fit = proper_fit(source_anchors, reference_anchors, anchor_mapping, anchor_weights,
        numerics=numerics, selection=anchor_selection)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            x, y, q, raw, cov = _prepare(ligand, reference_ligand, ligand_mapping, ligand_weights, numerics, Selection.ALL)
            if anchor_fit.rank < 2 or not anchor_fit.rotation_unique:
                return _not_evaluable(Metric.TS_POSE, limit, "protein_anchor_transform_underdetermined",
                    anchor_fit=anchor_fit, coverage=cov, weights=raw)
            if limit is None:
                return _not_evaluable(Metric.TS_POSE, None, "caller_threshold_unavailable",
                    anchor_fit=anchor_fit, coverage=cov, weights=raw)
            rotation = np.array(anchor_fit.transform.rotation, dtype=np.float64)
            translation = np.array(anchor_fit.transform.translation, dtype=np.float64)
            rmsd = _rmsd(x @ rotation.T + translation - y, q)
            status = Status.PASS if rmsd <= limit else Status.VIOLATED
            return Evaluation(Metric.TS_POSE, status, limit, rmsd, anchor_fit.transform, cov, raw,
                "ligand_scored_only_after_protein_anchor_transform", anchor_fit=anchor_fit)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("unsupported finite-precision pose: " + type(exc).__name__) from None
