"""Finite directed square presentations and exact algebraic certificates.

Grandis (2009), Sections1.4,2.1,3.2 and6.3 motivate distinguished paths,
positive chains, path composition and asymmetric costs. This is a finite
combinatorial model, NOT an algorithm for arbitrary continuous d-spaces.

Only supplied, evidence-bound squares identify two paths. Reachability is a
thin summary and is never called the fundamental category. Homology over Q
forgets torsion; the integer boundaries and explicit positive-representative
witnesses retain additional information. Chain homotopy does not establish a
continuous directed homotopy or biological intervention equivalence.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
import re
import numpy as np


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError('Explicit nonempty semantic identifier required')


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Immutable evidence digest required; a digest is not source admission')


@dataclass(frozen=True)
class Arrow:
    name: str
    source: str
    target: str
    relation: str
    cost: float
    evidence_sha256: str

    def __post_init__(self):
        for x in (self.name, self.source, self.target, self.relation):
            _text(x)
        _sha(self.evidence_sha256)
        if type(self.cost) not in (int, float) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError('Native nonnegative finite path cost required; regulatory sign is a relation, not cost')


@dataclass(frozen=True)
class Square:
    left: tuple[str, str]
    right: tuple[str, str]
    evidence_sha256: str

    def __post_init__(self):
        _sha(self.evidence_sha256)
        for side in (self.left, self.right):
            if type(side) is not tuple or len(side) != 2:
                raise ValueError('Each explicit square boundary must contain two directed arrows')
            for name in side:
                _text(name)


@dataclass(frozen=True)
class PathProof:
    status: str
    visited_states: int
    rewrite_count: int | None
    witness: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DirectedComplex:
    vertices: tuple[str, ...]
    arrows: tuple[Arrow, ...]
    squares: tuple[Square, ...]
    cost_unit: str

    def __post_init__(self):
        if (type(self.vertices) is not tuple or not 1 <= len(self.vertices) <= 128
                or len(set(self.vertices)) != len(self.vertices)
                or type(self.arrows) is not tuple or len(self.arrows) > 512
                or type(self.squares) is not tuple or len(self.squares) > 256):
            raise ValueError('Complete immutable finite presentation exceeds its bounds')
        for vertex in self.vertices:
            _text(vertex)
        _text(self.cost_unit)
        if any(type(a) is not Arrow for a in self.arrows) or len({a.name for a in self.arrows}) != len(self.arrows):
            raise ValueError('Unique typed directed arrows required')
        for a in self.arrows:
            a.__post_init__()
            if a.source not in self.vertices or a.target not in self.vertices:
                raise ValueError('Arrow has an unknown endpoint')
        lookup = {a.name: a for a in self.arrows}
        for s in self.squares:
            if type(s) is not Square:
                raise ValueError('Typed evidence-bound square required')
            s.__post_init__()
            if any(name not in lookup for name in (*s.left, *s.right)):
                raise ValueError('Square names an unknown arrow')
            first = lookup[s.left[0]].source
            if lookup[s.right[0]].source != first or self.endpoint(first, s.left) != self.endpoint(first, s.right):
                raise ValueError('Square paths must be composable with the same endpoints')

    def endpoint(self, start, path):
        if start not in self.vertices or type(path) is not tuple or len(path) > 64:
            raise ValueError('Known start and finite immutable directed path required')
        lookup = {a.name: a for a in self.arrows}
        here = start
        for name in path:
            if name not in lookup or lookup[name].source != here:
                raise ValueError('Path uses a missing, reversed or noncomposable arrow')
            here = lookup[name].target
        return here

    def compose(self, start, first, second):
        middle = self.endpoint(start, first)
        self.endpoint(middle, second)
        self.endpoint(start, first + second)
        return first + second

    def path_cost(self, path):
        lookup = {a.name: a for a in self.arrows}
        if type(path) is not tuple or len(path) > 64 or any(name not in lookup for name in path):
            raise ValueError('Complete finite native path required')
        if path:
            self.endpoint(lookup[path[0]].source, path)
        value = sum(lookup[name].cost for name in path)
        if not math.isfinite(value):
            raise ValueError('Path cost overflow')
        return float(value)

    def shortest_cost(self, source, target):
        """Lawvere-style asymmetric reachability summary, not path-class identity."""
        if source not in self.vertices or target not in self.vertices:
            raise ValueError('Known source/target vertices required')
        distance = {source: 0.}
        queue = [(0., source)]
        while queue:
            cost, here = heapq.heappop(queue)
            if cost != distance[here]:
                continue
            if here == target:
                return cost
            for a in self.arrows:
                if a.source == here:
                    new = cost + a.cost
                    if not math.isfinite(new):
                        raise ValueError('Path cost overflow')
                    if new < distance.get(a.target, math.inf):
                        distance[a.target] = new
                        heapq.heappush(queue, (new, a.target))
        return math.inf

    def equivalent(self, start, first, second, *, max_states=20_000):
        """Exact equal-length square-rewrite congruence, or explicit unknown on cap.

        Both rewrite directions generate equivalence, not reversal of1-paths.
        Square rewrites preserve length, so complete closure is a finite decision
        procedure for this specific presentation. No arbitrary commutation added.
        """
        if type(max_states) is not int or not 1 <= max_states <= 100_000:
            raise ValueError('Explicit finite proof-search bound required')
        end = self.endpoint(start, first)
        other = self.endpoint(start, second)
        if end != other or len(first) != len(second):
            return PathProof('disproved-in-presentation', 0, None, ())
        if first == second:
            return PathProof('proved', 1, 0, (first,))
        rules = tuple((a, b) for s in self.squares for a, b in ((s.left, s.right), (s.right, s.left)))
        previous = {first: None}
        queue = deque([first])
        while queue:
            path = queue.popleft()
            for i in range(len(path) - 1):
                for left, right in rules:
                    if path[i:i+2] != left:
                        continue
                    candidate = path[:i] + right + path[i+2:]
                    if candidate in previous:
                        continue
                    if len(previous) >= max_states:
                        return PathProof('unknown-budget', len(previous), None, ())
                    previous[candidate] = path
                    if candidate == second:
                        witness = [candidate]
                        while previous[witness[-1]] is not None:
                            witness.append(previous[witness[-1]])
                        witness.reverse()
                        return PathProof('proved', len(previous), len(witness)-1, tuple(witness))
                    queue.append(candidate)
        return PathProof('disproved-in-presentation', len(previous), None, ())

    def boundaries(self):
        """Cellular integer boundaries: each square is left path minus right path."""
        vi = {v: i for i, v in enumerate(self.vertices)}
        ai = {a.name: i for i, a in enumerate(self.arrows)}
        b1 = np.zeros((len(vi), len(ai)), dtype=np.int64)
        b2 = np.zeros((len(ai), len(self.squares)), dtype=np.int64)
        for i, a in enumerate(self.arrows):
            b1[vi[a.target], i] += 1
            b1[vi[a.source], i] -= 1
        for j, s in enumerate(self.squares):
            for name in s.left:
                b2[ai[name], j] += 1
            for name in s.right:
                b2[ai[name], j] -= 1
        if np.any(b1 @ b2):
            raise ValueError('Boundary-squared invariant failed')
        return b1, b2

    def betti_rational(self):
        b1, b2 = self.boundaries()
        r1, r2 = _rank(b1), _rank(b2)
        return len(self.vertices)-r1, len(self.arrows)-r1-r2, len(self.squares)-r2

    def positive_cycle(self, coefficients):
        z = _integer_array(coefficients, (len(self.arrows),))
        return bool(np.all(z >= 0) and not np.any(self.boundaries()[0] @ z))

    def verify_positive_class(self, cycle, positive_representative, boundary_coefficients):
        """Check a supplied cone-membership witness, not a complete cone solver."""
        z = _integer_array(cycle, (len(self.arrows),))
        p = _integer_array(positive_representative, (len(self.arrows),))
        b = _integer_array(boundary_coefficients, (len(self.squares),))
        b1, b2 = self.boundaries()
        return bool(not np.any(b1 @ z) and np.all(p >= 0) and not np.any(b1 @ p)
                    and np.array_equal(z-p, b2 @ b))

    def opposite(self):
        return DirectedComplex(self.vertices,
            tuple(Arrow(a.name, a.target, a.source, a.relation, a.cost, a.evidence_sha256) for a in self.arrows),
            tuple(Square(s.left[::-1], s.right[::-1], s.evidence_sha256) for s in self.squares), self.cost_unit)


class DirectedFunctor:
    """Maps a whole explicit presentation (or caller-selected subpresentation).

    Objects map to objects, generators to composable paths, and relation labels
    to fixed words. Extending by concatenation preserves identities/composition.
    Source square relations must have checked target rewrite witnesses. Neither
    faithfulness/fullness nor causal validity follows from these conditions.
    """
    def __init__(self, source, target, objects, arrows, relations, evidence_sha256):
        if type(source) is not DirectedComplex or type(target) is not DirectedComplex:
            raise ValueError('Typed finite source and target presentations required')
        _sha(evidence_sha256)
        if (type(objects) is not dict or set(objects) != set(source.vertices)
                or type(arrows) is not dict or set(arrows) != {a.name for a in source.arrows}
                or type(relations) is not dict or set(relations) != {a.relation for a in source.arrows}):
            raise ValueError('Complete explicit object, arrow and relation maps required')
        self.source, self.target = source, target
        self.objects, self.arrows, self.relations = dict(objects), dict(arrows), dict(relations)
        self.evidence_sha256 = evidence_sha256
        self._frozen_maps = (source, target, tuple(sorted(objects.items())),
                             tuple(sorted(arrows.items())), tuple(sorted(relations.items())), evidence_sha256)
        self._check()

    def _check(self):
        if (type(self.objects) is not dict or type(self.arrows) is not dict or type(self.relations) is not dict
                or set(self.objects) != set(self.source.vertices)
                or set(self.arrows) != {a.name for a in self.source.arrows}
                or set(self.relations) != {a.relation for a in self.source.arrows}
                or (self.source, self.target, tuple(sorted(self.objects.items())),
                    tuple(sorted(self.arrows.items())), tuple(sorted(self.relations.items())),
                    self.evidence_sha256) != self._frozen_maps):
            raise ValueError('Frozen complete directed functor mapping changed')
        lookup = {a.name: a for a in self.target.arrows}
        for v in self.objects.values():
            if v not in self.target.vertices:
                raise ValueError('Unknown mapped vertex')
        for a in self.source.arrows:
            path = self.arrows[a.name]
            if self.target.endpoint(self.objects[a.source], path) != self.objects[a.target]:
                raise ValueError('Mapped arrow path has wrong endpoints')
            word = self.relations[a.relation]
            if type(word) is not tuple or tuple(lookup[e].relation for e in path) != word:
                raise ValueError('Mapped path changes the declared relation word')

    def map_path(self, start, path):
        self._check()
        self.source.endpoint(start, path)
        result = tuple(e for name in path for e in self.arrows[name])
        self.target.endpoint(self.objects[start], result)
        return result

    def verify(self, *, max_states=20_000):
        self._check()
        lookup = {a.name: a for a in self.source.arrows}
        proofs = []
        for s in self.source.squares:
            start = lookup[s.left[0]].source
            proofs.append(self.target.equivalent(self.objects[start],
                self.map_path(start, s.left), self.map_path(start, s.right), max_states=max_states))
        proved = all(p.status == 'proved' for p in proofs)
        ratios = []
        if self.source.cost_unit != self.target.cost_unit:
            weighted = False
        else:
            weighted = True
            for a in self.source.arrows:
                cost = self.target.path_cost(self.arrows[a.name])
                ratios.append(0. if cost == 0 else math.inf if a.cost == 0 else cost/a.cost)
        return dict(status='proved' if proved else 'unknown' if any(p.status == 'unknown-budget' for p in proofs) else 'refused',
            relations_preserved=proved, composition_preserved_by_construction=True,
            identities_preserved_by_construction=True, square_proofs=tuple(proofs),
            generator_cost_lipschitz_bound=max(ratios, default=0.) if weighted else None,
            cost_units_compatible=weighted, faithful_or_full_proved=False,
            continuous_homotopy_proved=False, biological_validity_proved=False)


def _integer_array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iu' or np.any(np.abs(a.astype(object)) > 1_000_000):
        raise ValueError('Bounded exact integer chain/map coefficients required')
    return a.astype(object)  # Python integers avoid overflow in exact certificate checks


def _rank(matrix):
    """Exact rational elimination; Betti numbers only, no integral torsion claim."""
    rows, cols = matrix.shape
    a = [[Fraction(int(x)) for x in row] for row in matrix]
    pivot = 0
    for col in range(cols):
        candidate = next((i for i in range(pivot, rows) if a[i][col]), None)
        if candidate is None:
            continue
        a[pivot], a[candidate] = a[candidate], a[pivot]
        value = a[pivot][col]
        a[pivot] = [x/value for x in a[pivot]]
        for i in range(pivot+1, rows):
            scale = a[i][col]
            if scale:
                a[i] = [x-scale*y for x, y in zip(a[i], a[pivot])]
        pivot += 1
        if pivot == rows:
            break
    return pivot


def _chain(chain):
    if type(chain) is not tuple or len(chain) != 2:
        raise ValueError('Finite C2 -> C1 -> C0 chain required')
    a, b = map(np.asarray, chain)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
            or max(*a.shape, *b.shape) > 512):
        raise ValueError('Complete bounded chain dimensions required')
    a, b = _integer_array(a, a.shape), _integer_array(b, b.shape)
    if np.any(a @ b):
        raise ValueError('Boundary squared must vanish exactly')
    return (a, b), (a.shape[0], a.shape[1], b.shape[1])


def _maps(source, target, maps, positive):
    s, ns = _chain(source); t, nt = _chain(target)
    if type(maps) is not tuple or len(maps) != 3:
        raise ValueError('All three degree maps required')
    f = tuple(_integer_array(m, (nt[k], ns[k])) for k, m in enumerate(maps))
    if positive and any(np.any(m < 0) for m in f):
        raise ValueError('Directed chain maps must preserve the designated positive cones')
    if any(not np.array_equal(t[k] @ f[k+1], f[k] @ s[k]) for k in (0, 1)):
        raise ValueError('Chain map must commute with the boundary')
    return s, t, ns, nt, f


def verify_positive_chain_map(source, target, maps):
    _maps(source, target, maps, True)
    return True


def verify_chain_homotopy(source, target, first, second, homotopy, *, require_positive=True):
    """Check G-F = dH+Hd exactly; not a proof of spatial homotopy realization."""
    if type(require_positive) is not bool:
        raise ValueError('Explicit positivity policy required')
    s, t, ns, nt, f = _maps(source, target, first, require_positive)
    *_, g = _maps(source, target, second, require_positive)
    if type(homotopy) is not tuple or len(homotopy) != 2:
        raise ValueError('Complete H0/H1 homotopy maps required')
    h = tuple(_integer_array(m, (nt[k+1], ns[k])) for k, m in enumerate(homotopy))
    if require_positive and any(np.any(m < 0) for m in h):
        raise ValueError('Requested positive chain homotopy has negative coefficients')
    rhs = (t[0] @ h[0], t[1] @ h[1] + h[0] @ s[0], h[1] @ s[1])
    if any(not np.array_equal(g[k]-f[k], rhs[k]) for k in range(3)):
        raise ValueError('Exact chain-homotopy equation failed')
    return True
