"""Bounded exact F2 Vietoris--Rips H0/H1 and piecewise geometric objectives.

Reference reduction is implemented independently; installed GUDHI is a test
oracle, not a runtime dependency. Distances are caller-declared comparable
coordinates. This module never symmetrizes a directed graph or certifies causal
or biological equivalence. H2 is deliberately unavailable: killing its classes
would require tetrahedra. Truncated unpaired bars are explicitly right-censored.

The differentiable loss is a local adaptation of Moor et al., ICML 2020,
Section 3: union of source/latent persistence-critical edge coordinates, with
mean-square reduction and optional H1. It is NOT diagram Wasserstein distance.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np


@dataclass(frozen=True)
class PersistenceBar:
    dimension: int
    birth: float
    death: float | None
    status: str
    birth_simplex: tuple[int, ...]
    death_simplex: tuple[int, ...] | None


@dataclass(frozen=True)
class PersistenceResult:
    bars: tuple[PersistenceBar, ...]
    simplex_count: int
    vertex_count: int
    filtration_radius: float
    complete_filtration: bool
    coefficient_field: int = 2
    complete_through_dimension: int = 1

    def intervals(self, dimension: int):
        if type(dimension) is not int or dimension not in (0, 1):
            raise ValueError('Only fully computed H0 and H1 are available')
        return tuple((b.birth, b.death, b.status) for b in self.bars
                     if b.dimension == dimension)


def checked_distances(values, *, max_vertices=64, require_metric=True):
    if type(max_vertices) is not int or not 1 <= max_vertices <= 128:
        raise ValueError('Explicit distance bound must be an integer in 1..128')
    if type(require_metric) is not bool:
        raise ValueError('Metric requirement must be an explicit Boolean')
    raw = np.asarray(values)
    if raw.dtype.kind not in 'fi' or raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError('A square real numeric distance matrix is required')
    n = len(raw)
    if not 1 <= n <= max_vertices:
        raise ValueError('Complete distance matrix exceeds vertex budget; no truncation')
    d = np.array(raw, dtype=np.float64, copy=True)
    if (not np.isfinite(d).all() or (d < 0).any() or np.any(np.diag(d) != 0)
            or not np.array_equal(d, d.T)):
        raise ValueError('Distances must be finite, nonnegative, exactly symmetric with zero diagonal')
    if require_metric:
        tolerance = 1e-12 * max(1., float(d.max()))
        for k in range(n):
            if np.any(d > d[:, k, None] + d[None, k, :] + tolerance):
                raise ValueError('Triangle inequality fails; not a metric/pseudometric')
    return d


def rips_persistence(distances, *, max_radius=None, max_vertices=64,
                     max_simplices=50_000, require_metric=True):
    """Exact sparse column reduction of all admitted vertices, edges and triangles.

    Filtration order is (diameter, dimension, lexicographic vertices), so faces
    always precede cofaces and ties are deterministic. Positive-length bars only
    are returned. Coincident vertices are allowed as an explicit pseudometric.
    Bounds reject the entire request, never silently sample nodes or simplices.
    """
    d = checked_distances(distances, max_vertices=max_vertices, require_metric=require_metric)
    if type(max_simplices) is not int or not 1 <= max_simplices <= 100_000:
        raise ValueError('Positive finite simplex budget up to100000 required')
    radius = float(d.max()) if max_radius is None else max_radius
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise ValueError('Finite nonnegative filtration radius required')
    cells = []
    for size in (1, 2, 3):
        for simplex in combinations(range(len(d)), size):
            value = max((float(d[i, j]) for i, j in combinations(simplex, 2)), default=0.)
            if value <= radius:
                if len(cells) >= max_simplices:
                    raise ValueError('Complete filtration exceeds simplex budget')
                cells.append((value, size - 1, simplex))
    cells.sort()
    index = {simplex: j for j, (_, _, simplex) in enumerate(cells)}
    reduced = {}
    owners = {}
    births = set()
    pairs = {}
    for j, (_, dim, simplex) in enumerate(cells):
        column = set() if dim == 0 else {
            index[simplex[:k] + simplex[k + 1:]] for k in range(len(simplex))}
        while column and max(column) in owners:
            column.symmetric_difference_update(reduced[owners[max(column)]])
        if column:
            pivot = max(column)
            owners[pivot] = j
            reduced[j] = column
            pairs[pivot] = j
        else:
            births.add(j)
    complete = radius >= float(d.max())
    bars = []
    for i in sorted(births):
        birth, dim, simplex = cells[i]
        if dim > 1:
            continue  # no fictitious H2 completion from a2-skeleton
        j = pairs.get(i)
        if j is not None:
            death, _, killed_by = cells[j]
            if death > birth:
                bars.append(PersistenceBar(dim, birth, death, 'finite', simplex, killed_by))
        else:
            bars.append(PersistenceBar(dim, birth, None,
                'essential' if complete else 'right_censored', simplex, None))
    return PersistenceResult(tuple(bars), len(cells), len(d), float(radius), complete)


def _diagram(result, dimension):
    if type(result) is not PersistenceResult:
        raise ValueError('Exact reference persistence result required')
    rows = result.intervals(dimension)
    if any(status == 'right_censored' for _, _, status in rows):
        raise ValueError('Right-censored intervals cannot be treated as essential bars')
    return rows


def bottleneck_distance(left, right, *, dimension=1, max_bars=256):
    """Full finite/essential diagram matching, including multiplicity and diagonal.

    Units/filtration definitions must already agree. Binary-search the finite
    cost levels and test perfect bipartite matchings; no descriptor compression.
    """
    a, b = _diagram(left, dimension), _diagram(right, dimension)
    if type(max_bars) is not int or not 1 <= max_bars <= 512 or len(a) + len(b) > max_bars:
        raise ValueError('Complete diagram exceeds explicit matching budget')
    n, m = len(a), len(b)
    if n + m == 0:
        return 0.
    cost = np.full((n + m, n + m), np.inf)
    for i, (birth, death, _) in enumerate(a):
        for j, (other_birth, other_death, _) in enumerate(b):
            if death is None and other_death is None:
                cost[i, j] = abs(birth - other_birth)
            elif death is not None and other_death is not None:
                cost[i, j] = max(abs(birth - other_birth), abs(death - other_death))
        if death is not None:
            cost[i, m:] = (death - birth) / 2
    for j, (birth, death, _) in enumerate(b):
        if death is not None:
            cost[n:, j] = (death - birth) / 2
    cost[n:, m:] = 0.
    levels = sorted(set(float(x) for x in cost.ravel() if math.isfinite(x)))

    def feasible(threshold):
        neighbors = [np.flatnonzero(row <= threshold).tolist() for row in cost]
        match = [-1] * (n + m)
        def augment(i, seen):
            for j in neighbors[i]:
                if j in seen:
                    continue
                seen.add(j)
                if match[j] == -1 or augment(match[j], seen):
                    match[j] = i
                    return True
            return False
        return all(augment(i, set()) for i in range(n + m))

    if not levels or not feasible(levels[-1]):
        return math.inf
    low, high = 0, len(levels) - 1
    while low < high:
        middle = (low + high) // 2
        if feasible(levels[middle]):
            high = middle
        else:
            low = middle + 1
    return levels[low]


def _critical_edges(result, d):
    """Tie-complete support at invariant persistence birth/death levels.

    Selecting one simplex edge by its vertex IDs changes a geometric objective
    under relabeling when distances tie. Instead retain every edge at a positive
    bar's critical filtration level. This is a deliberate extension of the
    critical-edge loss, not differentiation through persistence pair selection.
    """
    levels = set()
    for bar in result.bars:
        if len(bar.birth_simplex) >= 2:
            levels.add(bar.birth)
        if bar.death is not None:
            levels.add(bar.death)
    return {edge for edge in combinations(range(len(d)), 2) if float(d[edge]) in levels}


def topological_metric_loss(reference, latent, *, max_vertices=64, max_simplices=50_000):
    """Piecewise gradient through critical distances, not through discrete reduction.

    Matrices retain known row correspondence. The reference is detached by
    design; no labels or hidden geometry are selected here. Selecting topology
    is nondifferentiable at filtration ties/pairing changes. Zero topology loss
    alone neither proves isometry nor recovers all graph relations.
    """
    import torch
    if (not isinstance(reference, torch.Tensor) or not isinstance(latent, torch.Tensor)
            or reference.shape != latent.shape or not latent.is_floating_point()
            or not reference.is_floating_point() or latent.device != reference.device):
        raise ValueError('Matching real native distance tensors required')
    x = checked_distances(reference.detach().cpu().numpy(), max_vertices=max_vertices)
    z = checked_distances(latent.detach().cpu().numpy(), max_vertices=max_vertices)
    px = rips_persistence(x, max_vertices=max_vertices, max_simplices=max_simplices)
    pz = rips_persistence(z, max_vertices=max_vertices, max_simplices=max_simplices)
    edges = tuple(sorted(_critical_edges(px, x) | _critical_edges(pz, z)))
    if edges:
        i, j = zip(*edges)
        loss = (latent[list(i), list(j)].double() - reference.detach()[list(i), list(j)].double()).square().mean()
    else:
        loss = latent.double().sum() * 0.
    return dict(loss=loss, critical_edges=edges, reference=px, latent=pz,
                tie_policy='all-edges-at-persistence-critical-levels',
                differentiation='fixed-pairing-piecewise;ties-not-globally-differentiable',
                diagram_distance=False, training_admission=False)
