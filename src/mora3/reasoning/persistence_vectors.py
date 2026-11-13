"""Bounded batched persistence landscapes/images and explicitly named L2 metrics.

Bubenik (2015; 1810.04963): complete landscapes determine finite barcodes.
Adams et al. (JMLR18/16-337): persistence images integrate weighted densities.
Sampled/truncated landscapes and finite images are NOT lossless encodings.
Original PersistenceResult objects, simplex pairing, essential/censored status,
and the separate directed-complex evidence must therefore remain authoritative.

All arithmetic is float64. Exact landscape integration means no quadrature or
rank truncation error in real arithmetic; floating-point rounding still applies.
This code adds no parameters, optimizer updates, dataset or training authority.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
import math
import torch
from .persistent_topology import PersistenceResult

_MAX_ELEMENTS = 16_000_000


def _budget(count: int, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= _MAX_ELEMENTS or count > limit:
        raise ValueError('Complete vectorization exceeds explicit element budget; no truncation')


def _batch(bars: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (not isinstance(bars, torch.Tensor) or not bars.is_floating_point()
            or bars.ndim != 3 or bars.shape[-1] != 2
            or not 1 <= bars.shape[0] <= 1024 or bars.shape[1] > 512
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != bars.shape[:2] or mask.device != bars.device):
        raise ValueError('Bounded BxNx2 real birth/death tensor and explicit Boolean mask required')
    values = bars.double()
    if not torch.isfinite(values).all() or torch.any(values[~mask] != 0):
        raise ValueError('Finite intervals and exactly zero padding required')
    life = values[..., 1] - values[..., 0]
    if not torch.isfinite(life).all() or torch.any(life[mask] <= 0):
        raise ValueError('Finite positive-length birth/death intervals required')
    return values


def _grid(values: torch.Tensor, device, *, minimum: int = 2) -> torch.Tensor:
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.ndim != 1 or not minimum <= values.numel() <= 65536
            or values.device != device or not torch.isfinite(values).all()):
        raise ValueError('Explicit bounded finite grid on the same device required')
    grid = values.double()
    if torch.any(torch.diff(grid) <= 0):
        raise ValueError('Strictly increasing grid required')
    return grid


def landscape_values(bars: torch.Tensor, mask: torch.Tensor, grid: torch.Tensor,
                     *, layers: int | None = None, max_elements: int = _MAX_ELEMENTS) -> torch.Tensor:
    """All tents and rank order statistics, vectorized over diagrams/bars/samples.

    Optional layers is an explicit approximation. Empty diagrams use zero layers
    padded to the declared common feature dimension, not fabricated intervals.
    Fixed-grid derivatives are piecewise; sorting/tent ties are nonsmooth.
    """
    bars = _batch(bars, mask)
    grid = _grid(grid, bars.device, minimum=1)
    batch, count, _ = bars.shape
    layers = max(1, count) if layers is None else layers
    if type(layers) is not int or not 1 <= layers <= 512:
        raise ValueError('Explicit positive landscape layer bound required')
    _budget(batch * max(count, layers) * len(grid), max_elements)
    if count == 0:
        return bars.new_zeros((batch, layers, len(grid)))
    tents = torch.minimum(grid[None, None, :] - bars[..., 0, None],
                          bars[..., 1, None] - grid[None, None, :]).clamp_min(0)
    tents = torch.where(mask[..., None], tents, torch.zeros_like(tents))
    values = torch.sort(tents, dim=1, descending=True).values[:, :layers]
    if layers > count:
        values = torch.cat((values, values.new_zeros((batch, layers-count, len(grid)))), dim=1)
    return values


def sampled_landscape_vectors(bars: torch.Tensor, mask: torch.Tensor, grid: torch.Tensor,
                              *, layers: int | None = None, max_elements: int = _MAX_ELEMENTS) -> dict:
    """Weighted sampled values whose Euclidean norm is trapezoidal L2, not exact L2.

    Domain clipping is rejected. Discretization and optional layer truncation are
    explicitly reported even when a particular example happens to be exact.
    Grid/normalization must be frozen from training data before held-out use.
    """
    bars = _batch(bars, mask)
    grid = _grid(grid, bars.device)
    if mask.any() and (torch.any(bars[..., 0][mask] < grid[0])
                       or torch.any(bars[..., 1][mask] > grid[-1])):
        raise ValueError('Grid must cover the complete finite barcode support')
    values = landscape_values(bars, mask, grid, layers=layers, max_elements=max_elements)
    widths = torch.diff(grid)
    weights = torch.cat((widths[:1]/2, (widths[:-1]+widths[1:])/2, widths[-1:]/2))
    return dict(vectors=(values*weights.sqrt()[None, None, :]).flatten(1), values=values,
                quadrature_weights=weights, metric='sampled-landscape-trapezoidal-L2',
                lossless=False, all_layers_retained=values.shape[1] >= bars.shape[1],
                max_grid_spacing=float(widths.max()), training_admission=False)


def _critical_knots(diagram: torch.Tensor) -> torch.Tensor:
    """Endpoints, peaks and every active opposite-slope tent intersection."""
    if not len(diagram):
        return diagram.new_empty(0)
    birth, death = diagram.unbind(-1)
    midpoint = birth/2 + death/2
    if torch.any((midpoint <= birth) | (midpoint >= death)):
        raise ValueError('Insufficient float64 resolution for an exact landscape peak')
    crossing = birth[:, None]/2 + death[None, :]/2
    valid = ((crossing >= birth[:, None]) & (crossing <= midpoint[:, None])
             & (crossing >= midpoint[None, :]) & (crossing <= death[None, :]))
    return torch.cat((birth, midpoint, death, crossing[valid]))


def exact_landscape_vectors(diagrams: tuple[torch.Tensor, ...],
                            *, max_elements: int = _MAX_ELEMENTS) -> dict:
    """Shared, data-adaptive Euclidean features with exact finite-landscape L2.

    Build a common knot basis containing ALL changes in every landscape. Two
    Gauss points on each interval integrate squared linear differences exactly.
    Feature dimensions/basis are collection-specific: never compare features
    from different calls without recomputing a common basis. All layers remain.
    This is bounded reference/reranking, not a claim of constant-time PH.
    """
    if type(diagrams) is not tuple or not 1 <= len(diagrams) <= 64:
        raise ValueError('Explicit bounded tuple of complete finite diagrams required')
    checked = []
    device = None
    for d in diagrams:
        if not isinstance(d, torch.Tensor) or d.ndim != 2 or d.shape[-1] != 2 or len(d) > 256:
            raise ValueError('Complete Nx2 diagram within exact landscape budget required')
        d = _batch(d[None], torch.ones((1, len(d)), dtype=torch.bool, device=d.device))[0]
        if device is not None and d.device != device:
            raise ValueError('All diagrams must use one device')
        device = d.device
        checked.append(d)
    layers = max(1, max(map(len, checked)))
    knots = torch.unique(torch.cat([_critical_knots(d) for d in checked]), sorted=True)
    if len(knots) == 0:
        knots = checked[0].new_tensor([0., 1.])
    _budget(len(diagrams)*layers*max(1, len(knots)-1)*2, max_elements)
    if len(knots) > 32769:
        raise ValueError('Complete exact landscape knot budget exceeded')
    widths = torch.diff(knots)
    nodes = knots[:-1, None] + widths[:, None] * knots.new_tensor([.5-math.sqrt(3)/6, .5+math.sqrt(3)/6])
    if torch.any(nodes[:, 0] <= knots[:-1]) or torch.any(nodes[:, 1] >= knots[1:]):
        raise ValueError('Insufficient float64 resolution for exact integration nodes')
    padded = checked[0].new_zeros((len(checked), layers, 2))
    mask = torch.zeros((len(checked), layers), dtype=torch.bool, device=device)
    for i, d in enumerate(checked):
        padded[i, :len(d)] = d
        mask[i, :len(d)] = True
    values = landscape_values(padded, mask, nodes.flatten(), max_elements=max_elements)
    weights = (widths/2).repeat_interleave(2)
    vectors = (values*weights.sqrt()[None, None, :]).flatten(1)
    if not torch.isfinite(vectors).all():
        raise ValueError('Exact landscape vectors overflowed float64')
    return dict(vectors=vectors, knots=knots, nodes=nodes, weights=weights,
                metric='exact-finite-landscape-L2', all_layers_retained=True,
                integration='two-point-Gauss-on-all-piecewise-linear-knots',
                basis_scope='this-complete-collection-only', floating_point_rounding=True,
                source_diagrams=tuple(d.detach().clone() for d in checked), training_admission=False)


def _normal_interval_mass(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Survival functions avoid cancellation in the positive Gaussian tail.
    scale = math.sqrt(2)
    sf = .5*(torch.erfc(lo/scale)-torch.erfc(hi/scale))
    cdf = .5*(torch.erfc(-hi/scale)-torch.erfc(-lo/scale))
    return torch.where(lo >= 0, sf, cdf).clamp_min(0)


def gaussian_persistence_images(bars: torch.Tensor, mask: torch.Tensor,
                                birth_edges: torch.Tensor, persistence_edges: torch.Tensor,
                                *, bandwidth: tuple[float, float],
                                max_elements: int = _MAX_ELEMENTS) -> dict:
    """Exact Gaussian rectangle integrals; separable tensor contraction over bars.

    Coordinates are (birth, death-birth); weights equal positive persistence.
    sigma values are standard deviations, not variances. Pixel masses, density-
    normalized L2 vectors and discarded Gaussian tail mass are all explicit.
    Images are lossy feature summaries; the input diagram is never overwritten.
    """
    bars = _batch(bars, mask)
    x = _grid(birth_edges, bars.device)
    y = _grid(persistence_edges, bars.device)
    if (type(bandwidth) is not tuple or len(bandwidth) != 2
            or any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in bandwidth)):
        raise ValueError('Two finite positive Gaussian standard deviations required')
    batch, count, _ = bars.shape
    _budget(batch*count*(len(x)+len(y)) + batch*(len(x)-1)*(len(y)-1), max_elements)
    life = bars[..., 1]-bars[..., 0]
    wx = _normal_interval_mass((x[:-1]-bars[..., 0, None])/bandwidth[0],
                               (x[1:]-bars[..., 0, None])/bandwidth[0])
    wy = _normal_interval_mass((y[:-1]-life[..., None])/bandwidth[1],
                               (y[1:]-life[..., None])/bandwidth[1])
    weights = torch.where(mask, life, torch.zeros_like(life))
    pixels = torch.einsum('bni,bnj,bn->bij', wy, wx, weights)
    areas = torch.diff(y)[:, None]*torch.diff(x)[None, :]
    if not torch.isfinite(pixels).all() or not torch.isfinite(areas).all() or torch.any(areas <= 0):
        raise ValueError('Persistence image or pixel areas overflowed float64')
    vectors = (pixels/areas.sqrt()).flatten(1)
    if not torch.isfinite(vectors).all():
        raise ValueError('Density-normalized persistence image overflowed float64')
    return dict(pixel_mass=pixels, pixel_mass_vectors=pixels.flatten(1),
                density_l2_vectors=vectors, outside_weighted_mass=(weights.sum(1)-pixels.sum((1, 2))).clamp_min(0),
                weight='positive-persistence', coordinates='birth,persistence',
                density_metric='piecewise-constant-persistence-image-density-L2',
                pixel_metric='persistence-image-pixel-mass-Euclidean', lossless=False,
                training_admission=False)


def pairwise_l2(left: torch.Tensor, right: torch.Tensor | None = None,
                *, max_elements: int = _MAX_ELEMENTS) -> torch.Tensor:
    """Vectorized Euclidean distances without cancellation-prone norm expansion.

    Inputs must already share the SAME grid, rank basis, normalization and units.
    This is not a Wasserstein/bottleneck distance and does not certify analogy.
    """
    right = left if right is None else right
    if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]
            or not left.is_floating_point() or not right.is_floating_point()
            or left.device != right.device or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
        raise ValueError('Comparable finite real feature matrices required')
    _budget(left.numel()+right.numel()+left.shape[0]*right.shape[0], max_elements)
    result = torch.cdist(left.double(), right.double(), p=2, compute_mode='donot_use_mm_for_euclid_dist')
    if not torch.isfinite(result).all():
        raise ValueError('L2 distance overflowed float64')
    return result


def native_diagram_batch(results: tuple[PersistenceResult, ...], *, dimension: int) -> dict:
    """Preserve native objects and nonfinite bars separately; never cap infinity.

    Censored filtrations refuse. Essential bars do not belong to this finite L2
    feature space; their full birth multiset must match for finite-part ranking.
    Pairing simplices and all dimensions remain in source_results unchanged.
    """
    if type(results) is not tuple or not 1 <= len(results) <= 1024 or type(dimension) is not int or dimension not in (0, 1):
        raise ValueError('Bounded native persistence tuple and exact H0/H1 dimension required')
    finite, nonfinite, hashes = [], [], []
    for result in results:
        if (type(result) is not PersistenceResult or result.coefficient_field != 2
                or result.complete_through_dimension < dimension):
            raise ValueError('Complete compatible F2 native persistence result required')
        if not result.complete_filtration or any(b.status == 'right_censored' for b in result.bars):
            raise ValueError('Right-censored or incomplete topology cannot be silently vectorized')
        selected = [b for b in result.bars if b.dimension == dimension]
        for b in selected:
            if (not math.isfinite(b.birth) or b.status not in ('finite', 'essential')
                    or (b.status == 'finite' and (b.death is None or not math.isfinite(b.death) or b.death <= b.birth))
                    or (b.status == 'essential' and b.death is not None)):
                raise ValueError('Malformed native interval or essential status')
        finite.append([(b.birth, b.death) for b in selected if b.status == 'finite'])
        nonfinite.append(tuple(b for b in selected if b.status != 'finite'))
        hashes.append(sha256(json.dumps(asdict(result),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
    count = max(1, max(map(len, finite)))
    if count > 256:
        raise ValueError('Complete native barcode exceeds vectorization budget')
    bars = torch.zeros((len(results), count, 2), dtype=torch.float64)
    mask = torch.zeros((len(results), count), dtype=torch.bool)
    for i, rows in enumerate(finite):
        if rows:
            bars[i, :len(rows)] = torch.tensor(rows, dtype=torch.float64)
            mask[i, :len(rows)] = True
    return dict(bars=bars, mask=mask, nonfinite=tuple(nonfinite), source_results=results,
                source_sha256=tuple(hashes), dimension=dimension,
                all_native_information_retained=True, features_alone_lossless=False)


def _essential_signature(row) -> tuple:
    return tuple(sorted((b.birth, b.status) for b in row))


def _native_vectors(results, dimension):
    packed = native_diagram_batch(results, dimension=dimension)
    if any(_essential_signature(r) != _essential_signature(packed['nonfinite'][0]) for r in packed['nonfinite'][1:]):
        raise ValueError('Different essential birth multisets cannot be silently omitted from finite L2')
    diagrams = tuple(packed['bars'][i, packed['mask'][i]] for i in range(len(results)))
    return exact_landscape_vectors(diagrams)


def exact_native_landscape_distance(left: PersistenceResult, right: PersistenceResult) -> float:
    """Joint H0/H1 finite-landscape L2, conditional on identical essential bars."""
    squared = 0.
    for dimension in (0, 1):
        vectors = _native_vectors((left, right), dimension)['vectors']
        squared += float((vectors[0]-vectors[1]).square().sum())
    return math.sqrt(squared)


def rank_landscape_memories(query, candidates, *, limit: int) -> list[dict]:
    """Complete train-only search using batched exact finite-landscape L2.

    No approximate shortlist can discard a closer candidate. Batch-specific
    bases integrate the same exact L2. Directed functor/cost/causal certificates
    remain separate; this function never admits a reasoning trajectory.
    """
    from .analogy_control import TopologyMemory
    if type(query) is not TopologyMemory or type(candidates) not in (tuple, list):
        raise ValueError('Typed query and complete bounded memory collection required')
    query.__post_init__()
    if type(limit) is not int or not 1 <= limit <= 128 or len(candidates) > 1024:
        raise ValueError('Explicit bounded memory and retrieval budgets required')
    if len({c.identity for c in candidates if type(c) is TopologyMemory}) != len(candidates):
        raise ValueError('Unique typed memory identities required')
    eligible = []
    for c in candidates:
        c.__post_init__()
        if c.group_policy_sha256 != query.group_policy_sha256 or c.filtration_tag != query.filtration_tag:
            raise ValueError('Cannot mix leakage policies or incomparable topology filtrations')
        if c.split != 'train' or c.identity == query.identity or set(c.groups) & set(query.groups):
            continue
        if any(not p.complete_filtration or any(b.status == 'right_censored' for b in p.bars) for p in (query.diagram, c.diagram)):
            continue
        if any(_essential_signature(native_diagram_batch((query.diagram,),dimension=k)['nonfinite'][0]) !=
               _essential_signature(native_diagram_batch((c.diagram,),dimension=k)['nonfinite'][0]) for k in (0,1)):
            continue
        eligible.append(c)
    ranked = []
    for start in range(0, len(eligible), 32):
        chunk = eligible[start:start+32]
        results = (query.diagram,)+tuple(c.diagram for c in chunk)
        dimensions = []
        for k in (0, 1):
            v = _native_vectors(results, k)['vectors']
            dimensions.append(pairwise_l2(v[:1],v[1:])[0])
        distances = torch.stack(dimensions).square().sum(0).sqrt()
        for j, c in enumerate(chunk):
            ranked.append(dict(identity=c.identity, distance=float(distances[j]),
                per_dimension=tuple(float(d[j]) for d in dimensions), source_sha256=c.source_sha256,
                distance_metric='exact-finite-landscape-L2', essential_policy='identical-birth-multisets-required',
                all_candidates_scored=True, biological_equivalence_proved=False))
    return sorted(ranked,key=lambda row:(row['distance'],row['identity']))[:limit]
