"""Verified relational proposals, topological retrieval and native FoT control.

This module composes the existing finite inference forest. It never alters GFN
transition densities, starts training, or turns an analogy into biological truth.
Source evidence and held-out group policies are caller-owned qualifications;
a hash only binds the supplied artifact. Structural certificates are checked
against the exact proposed latent state and context before candidate selection.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
import torch

from .directed_topology import DirectedComplex, DirectedFunctor
from .persistent_topology import PersistenceResult, bottleneck_distance
from .fot_forest import grow_forest


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Exact immutable artifact digest required')


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError('Explicit nonempty semantic identifier required')


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def state_fingerprint(state):
    if (not isinstance(state, torch.Tensor) or state.ndim != 1 or not state.is_floating_point()
            or not 1 <= state.numel() <= 4096 or not torch.isfinite(state).all()):
        raise ValueError('Complete bounded finite latent vector required')
    raw = state.detach().cpu().contiguous()
    header = _wire(dict(dtype=str(raw.dtype), shape=list(raw.shape)))
    return sha256(header + raw.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class AnalogyDecision:
    status: str
    penalty: float
    state_sha256: str
    context_sha256: str
    proof_sha256: str
    reason: str

    def __post_init__(self):
        if self.status not in ('structurally_admissible', 'refused', 'unknown'):
            raise ValueError('Explicit structural acceptance/refusal/abstention required')
        if type(self.penalty) not in (int, float) or not math.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError('Finite nonnegative declared ranking penalty required')
        for x in (self.state_sha256, self.context_sha256, self.proof_sha256):
            _sha(x)
        _text(self.reason)

    @property
    def biological_validity(self):
        return False


def certify_analogy(functor, *, state_sha256, context_sha256, source_topology,
                    target_topology, topology_tolerance, filtration_contract, max_states=20_000):
    """Check both typed directed relations and full H0/H1 diagram comparisons.

    The explicit filtration contract names method, normalized unit and protocol
    digest. Equality is a caller assertion of comparability, not a unit converter.
    A common barcode is not a proof of isometry, functoriality or causal transfer.
    Budget exhaustion/censoring abstains rather than quietly admitting a proposal.
    """
    if type(functor) is not DirectedFunctor:
        raise ValueError('Explicit finite directed functor proposal required')
    _sha(state_sha256); _sha(context_sha256)
    if (type(filtration_contract) is not tuple or len(filtration_contract) != 3
            or type(topology_tolerance) not in (int, float)
            or not math.isfinite(topology_tolerance) or topology_tolerance < 0):
        raise ValueError('Predeclared comparable filtration and finite tolerance required')
    _text(filtration_contract[0]); _text(filtration_contract[1]); _sha(filtration_contract[2])
    if type(source_topology) is not PersistenceResult or type(target_topology) is not PersistenceResult:
        raise ValueError('Complete reference persistence results required')
    result = functor.verify(max_states=max_states)
    status = 'structurally_admissible' if result['status'] == 'proved' else result['status']
    reason = 'directed generator/composition/square checks passed; biological validity unestablished'
    distances = []
    if status == 'refused':
        reason = 'a declared source path relation is not preserved by the proposed target map'
    elif status == 'unknown':
        reason = 'directed relation proof exceeded its finite search budget'
    else:
        if any(b.status == 'right_censored' for p in (source_topology, target_topology) for b in p.bars):
            status = 'unknown'
            reason = 'right-censored topology cannot be treated as completed homology'
        else:
            distances = [bottleneck_distance(source_topology, target_topology, dimension=k) for k in (0, 1)]
            if any(not math.isfinite(x) or x > topology_tolerance for x in distances):
                status = 'refused'
                reason = 'predeclared complete-diagram tolerance exceeded'
    # The receipt remains finite even when essential-bar multiplicities disagree.
    bound = result['generator_cost_lipschitz_bound']
    proof = dict(schema='Mora3.StructuralAnalogyCertificate/v1',
        source=asdict(functor.source), target=asdict(functor.target),
        objects=functor.objects, arrows=functor.arrows, relations=functor.relations,
        mapping_evidence_sha256=functor.evidence_sha256,
        source_topology=asdict(source_topology), target_topology=asdict(target_topology),
        path_status=result['status'], square_proofs=[asdict(p) for p in result['square_proofs']],
        generator_cost_lipschitz_bound=bound if bound is None or math.isfinite(bound) else 'infinity',
        filtration_contract=filtration_contract, topology_tolerance=topology_tolerance,
        diagram_distances=[x if math.isfinite(x) else 'infinity' for x in distances],
        max_states=max_states, state_sha256=state_sha256, context_sha256=context_sha256,
        semantic_mapping_verified_externally=False, biological_validity=False)
    penalty = sum(distances) if distances and all(math.isfinite(x) for x in distances) else 0.
    return AnalogyDecision(status, penalty, state_sha256, context_sha256,
                           sha256(_wire(proof)).hexdigest(), reason)


def grow_analogy_forest(roots, expand, certify, *, context_sha256,
                        branches, beam, max_depth, max_candidates, min_distance=0.):
    """Use the real FoT implementation with state-bound structural refusal masks.

    The verifier is an explicit caller dependency, not a learned plausibility
    score. All candidate decisions survive, including unknown/refused proposals.
    Scores subtract the declared penalty; its calibration is not learned here.
    This is inference-only pruning, never an on-policy GFN training kernel.
    """
    _sha(context_sha256)
    if not callable(expand) or not callable(certify):
        raise ValueError('Explicit proposal and independent structural verifier required')
    receipts = []
    def checked_expand(parent, depth, count):
        states, scores, valid, terminal = expand(parent, depth, count)
        if (not isinstance(states, torch.Tensor) or states.shape != (count, roots.shape[1])
                or not isinstance(scores, torch.Tensor) or scores.shape != (count,)
                or not isinstance(valid, torch.Tensor) or valid.shape != (count,) or valid.dtype != torch.bool
                or not isinstance(terminal, torch.Tensor) or terminal.shape != (count,) or terminal.dtype != torch.bool):
            raise ValueError('Exact complete native FoT expansion tensors required')
        mask = valid.clone()
        penalties = []
        for i in range(count):
            fingerprint = state_fingerprint(states[i])
            decision = certify(states[i].detach().clone(), depth)
            if type(decision) is not AnalogyDecision:
                raise ValueError('Exact explicit structural verifier decision required')
            decision.__post_init__()
            if decision.state_sha256 != fingerprint or decision.context_sha256 != context_sha256:
                raise ValueError('Stale or cross-context analogy certificate')
            mask[i] &= decision.status == 'structurally_admissible'
            penalties.append(decision.penalty)
            receipts.append(dict(proposal=len(receipts), depth=depth,
                                 parent_state_sha256=state_fingerprint(parent), **asdict(decision)))
        adjusted = scores - scores.new_tensor(penalties)
        return states, adjusted, mask, terminal
    result = grow_forest(roots, checked_expand, branches=branches, beam=beam,
        max_depth=max_depth, max_candidates=max_candidates, min_distance=min_distance)
    return dict(**result, analogy_certificates=tuple(receipts), context_sha256=context_sha256,
                training_admission=False, gfn_sampling_kernel=False,
                biological_intervention_validated=False)


@dataclass(frozen=True)
class TopologyMemory:
    identity: str
    role: str
    split: str
    groups: tuple[str, ...]
    source_sha256: str
    group_policy_sha256: str
    filtration_tag: str
    diagram: PersistenceResult

    def __post_init__(self):
        _text(self.identity); _text(self.filtration_tag)
        _sha(self.source_sha256); _sha(self.group_policy_sha256)
        if self.role != 'condition' or self.split not in ('train', 'validation', 'test'):
            raise ValueError('Memory descriptors must come from condition-only native records')
        if (type(self.groups) is not tuple or not self.groups or len(set(self.groups)) != len(self.groups)
                or type(self.diagram) is not PersistenceResult):
            raise ValueError('Explicit leakage groups and exact topology result required')
        for group in self.groups:
            _text(group)


def rank_topological_memories(query, candidates, *, limit, metric="landscape_l2"):
    """Retrieve only train memories disjoint under one explicit grouping policy.

    Group completeness, rights, source revisions and protocol calibration remain
    external admission requirements. Query labels never enter this function.
    Ties use immutable IDs, not incoming collection order or hidden target values.
    """
    if metric == "landscape_l2":
        from .persistence_vectors import rank_landscape_memories
        return rank_landscape_memories(query, candidates, limit=limit)
    if metric != "bottleneck":
        raise ValueError('Explicit landscape_l2 or legacy bottleneck retrieval metric required')
    if type(query) is not TopologyMemory or type(candidates) not in (tuple, list):
        raise ValueError('Typed query and complete bounded memory collection required')
    query.__post_init__()
    if type(limit) is not int or not 1 <= limit <= 128 or len(candidates) > 1024:
        raise ValueError('Explicit bounded memory and retrieval budgets required')
    if len({c.identity for c in candidates if type(c) is TopologyMemory}) != len(candidates):
        raise ValueError('Unique typed memory identities required')
    ranked = []
    for c in candidates:
        c.__post_init__()
        if c.group_policy_sha256 != query.group_policy_sha256 or c.filtration_tag != query.filtration_tag:
            raise ValueError('Cannot mix leakage policies or incomparable topology filtrations')
        if c.split != 'train' or c.identity == query.identity or set(c.groups) & set(query.groups):
            continue
        if any(b.status == 'right_censored' for p in (query.diagram, c.diagram) for b in p.bars):
            continue  # unavailable distance, never a fabricated low score
        distances = tuple(bottleneck_distance(query.diagram, c.diagram, dimension=k) for k in (0, 1))
        if not all(math.isfinite(x) for x in distances):
            continue
        ranked.append(dict(identity=c.identity, distance=sum(distances), per_dimension=distances,
                           source_sha256=c.source_sha256, biological_equivalence_proved=False))
    return sorted(ranked, key=lambda row: (row['distance'], row['identity']))[:limit]


def functor_transport_loss(complex_, operators):
    """Latent square-commutation residual only for supplied source2-cells.

    For a path(a,b), column-vector transport is T_b @ T_a. Arbitrary actions are
    never forced to commute. This auxiliary objective can train caller-owned
    operators, but supplies neither new parameters nor a μP role declaration.
    """
    if type(complex_) is not DirectedComplex or type(operators) is not dict:
        raise ValueError('Explicit directed square presentation and operator map required')
    if set(operators) != {a.name for a in complex_.arrows} or not operators:
        raise ValueError('All and only source generator operators required')
    first = next(iter(operators.values()))
    if (not isinstance(first, torch.Tensor) or first.ndim != 2
            or first.shape[0] != first.shape[1] or not 1 <= first.shape[0] <= 256
            or not first.is_floating_point()):
        raise ValueError('Bounded square latent transport matrices required')
    for matrix in operators.values():
        if (not isinstance(matrix, torch.Tensor) or matrix.shape != first.shape
                or matrix.dtype != first.dtype or matrix.device != first.device
                or not torch.isfinite(matrix).all()):
            raise ValueError('Matching complete finite latent operators required')
    pieces = []
    for square in complex_.squares:
        a, b = square.left
        c, d = square.right
        left = operators[b].double() @ operators[a].double()
        right = operators[d].double() @ operators[c].double()
        pieces.append((left-right).square().mean())
    loss = torch.stack(pieces).mean() if pieces else sum(m.double().sum()*0 for m in operators.values())
    if not torch.isfinite(loss):
        raise ValueError('Latent transport composition overflow')
    return dict(loss=loss, observed_squares=len(pieces), adds_learned_parameters=False,
                biological_relations_inferred=False, training_admission=False)
