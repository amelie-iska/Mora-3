"""Bounded CPU reference for ConvexTok 2605.22821v1 §3.2–3.3.

Independent equation implementation, not upstream code. Candidate restrictions
are explicit: the LP bounds only this candidate universe, not all substrings.
All byte tokens and control IDs are reserved outside the learned-token budget.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .shortest_path import ByteTokenizer


@dataclass(frozen=True)
class VocabularyFit:
    learned: tuple[bytes, ...]
    scores: tuple[tuple[bytes, float], ...]
    lp_objective: float
    dual_objective: float
    rounded_objective: int
    max_constraint_violation: float
    dual_residual: float
    edge_count: int
    vertex_count: int


def substring_candidates(corpus, *, max_length, max_candidates=4096):
    """Enumerate bounded substrings within fields; never silently prune."""
    if type(max_length) is not int or max_length < 2:
        raise ValueError('max_length must be an integer >= 2')
    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError('Invalid candidate limit')
    candidates = set()
    for payload in corpus:
        if type(payload) is not bytes:
            raise ValueError('Corpus fields must be bytes')
        for start in range(len(payload)):
            for length in range(2, min(max_length, len(payload) - start) + 1):
                candidates.add(payload[start:start + length])
                if len(candidates) > max_candidates:
                    raise ValueError('candidate limit exceeded; choose an explicit smaller universe')
    return tuple(sorted(candidates))


def round_vocabulary(scores, budget, *, method='det'):
    """Paper Det/Bias/Int rules, byte-lexicographic ties and final token IDs."""
    if type(budget) is not int or budget < 0 or method not in ('det', 'bias', 'int'):
        raise ValueError('Invalid rounding budget or method')
    ByteTokenizer(scores)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Colour scores must lie in [0, 1]')
    ranked = sorted(scores, key=lambda t: (-(scores[t] / len(t) if method == 'bias'
                                            else scores[t]), t))
    if method == 'int':
        ranked = [t for t in ranked if scores[t] >= .999]
        # Numerical thresholding must not permit an oversized vocabulary.
    return tuple(sorted(ranked[:budget]))


def fit_vocabulary(corpus, candidates, *, vocab_size, controls=(), rounding='det',
                   max_bytes=100_000, max_edges=500_000, time_limit=120.0, lp_method='highs'):
    """Minimise sum of occurrence flows with unit flow per nonempty field.

    x_e <= c_token(e), sum(c) <= K, and all variables in [0,1].
    Repeated fields remain repeated objectives. Canonical ordering stabilises
    solver input; cross-solver/version equality for degenerate optima is not
    promised. The explicit LP method is part of caller fit provenance; changing
    it never changes the feasible problem. Failure/time limits raise; there is
    no heuristic fallback.
    """
    corpus, candidates, controls = tuple(corpus), tuple(candidates), tuple(controls)
    ByteTokenizer(candidates, controls=controls)
    if type(vocab_size) is not int or vocab_size < 256 + len(controls):
        raise ValueError('Vocabulary budget must reserve all bytes and controls')
    if any(type(p) is not bytes for p in corpus):
        raise ValueError('Corpus fields must be bytes')
    if any(type(n) is not int or n <= 0 for n in (max_bytes, max_edges)):
        raise ValueError('Resource limits must be positive integers')
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be finite and positive')
    if type(lp_method) is not str or lp_method not in ('highs', 'highs-ds', 'highs-ipm'):
        raise ValueError('Explicit supported HiGHS LP method required')
    if sum(map(len, corpus)) > max_bytes:
        raise ValueError('byte limit exceeded')
    corpus = tuple(sorted(p for p in corpus if p))
    if not corpus:
        raise ValueError('At least one nonempty training field required')
    candidates = tuple(sorted(candidates))
    # Validate the rounding rule before doing expensive work.
    budget = vocab_size - 256 - len(controls)
    round_vocabulary({}, budget, method=rounding)
    lookup = {t: i for i, t in enumerate(candidates)}
    lengths = sorted({len(t) for t in candidates})
    edges, demand = [], []
    for payload in corpus:
        offset = len(demand)
        demand.extend([-1.] + [0.] * (len(payload) - 1) + [1.])
        for start in range(len(payload)):
            edges.append((offset + start, offset + start + 1, None))
            for length in lengths:
                if start + length > len(payload):
                    break
                colour = lookup.get(payload[start:start + length])
                if colour is not None:
                    edges.append((offset + start, offset + start + length, colour))
            if len(edges) > max_edges:
                raise ValueError('edge limit exceeded')
    ne, nc = len(edges), len(candidates)
    nv = len(demand)
    er, ec, ev, ur, uc, uv = [], [], [], [], [], []
    row = 0
    for index, (start, end, colour) in enumerate(edges):
        er.extend((start, end)); ec.extend((index, index)); ev.extend((-1., 1.))
        if colour is not None:
            ur.extend((row, row)); uc.extend((index, ne + colour)); uv.extend((1., -1.))
            row += 1
    for colour in range(nc):
        ur.append(row); uc.append(ne + colour); uv.append(1.)
    eq = coo_matrix((ev, (er, ec)), shape=(nv, ne + nc)).tocsr()
    ub = coo_matrix((uv, (ur, uc)), shape=(row + 1, ne + nc)).tocsr()
    rhs = np.array([0.] * row + [float(budget)])
    demand = np.array(demand)
    cost = np.r_[np.ones(ne), np.zeros(nc)]
    result = linprog(cost, A_ub=ub, b_ub=rhs, A_eq=eq, b_eq=demand,
                     bounds=(0., 1.), method=lp_method, options={'time_limit': time_limit})
    if not result.success:
        raise RuntimeError(f'Vocabulary LP failed ({result.status}): {result.message}')
    x = result.x
    violation = float(max(np.max(np.abs(eq @ x - demand)), np.max(ub @ x - rhs),
                          np.max(-x), np.max(x - 1), 0.))
    # HiGHS marginals use derivatives w.r.t. RHS; upper-bound terms matter.
    dual = float(demand @ result.eqlin.marginals + rhs @ result.ineqlin.marginals
                 + result.upper.marginals.sum())
    stationarity = cost - (eq.T @ result.eqlin.marginals + ub.T @ result.ineqlin.marginals
                          + result.lower.marginals + result.upper.marginals)
    dual_residual = float(max(np.max(np.abs(stationarity)),
                              np.max(result.ineqlin.marginals),
                              np.max(result.upper.marginals),
                              np.max(-result.lower.marginals), 0.))
    if (violation > 1e-6 or dual_residual > 1e-6
            or abs(result.fun - dual) > 1e-6 * max(1., abs(result.fun))):
        raise RuntimeError('LP numerical certificate failed')
    scores = {t: float(np.clip(x[ne + i], 0., 1.)) for i, t in enumerate(candidates)}
    learned = round_vocabulary(scores, budget, method=rounding)
    tokenizer = ByteTokenizer(learned, controls=controls)
    rounded = sum(len(tokenizer.encode(p)) for p in corpus)
    return VocabularyFit(learned, tuple(scores.items()), float(result.fun), dual,
                         rounded, violation, dual_residual, ne, nv)
