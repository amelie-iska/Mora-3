# Complete directed trajectory analogies and native GFlowNet integration

## Status and scope

This extends the earlier geometry/topology package from commit
`c35979190718f5a3294ca7b778166a5edaeeeb60`. It is implemented research software
with scoped mathematical, gradient and integration tests. It is NOT evidence of
learned reasoning improvements, complete expanded-corpus readiness or a training
launch. Read the final `outputs/trajectory-analogy-20261110/` receipts for exact
current code hashes and verification counts.

The user's cap is unchanged: **24 proxy muP trials plus TWO approximately80M
full-model training runs with transferred hyperparameters**. The two target runs
retain muP parameterization. No extra SP baseline, intermediate-width, training
ablation or fresh retry was started. All qualification in this extension used
zero optimizer updates. The old44-run comparison design is superseded.

## What the research supports, and what is our adaptation

The four requested versions were reconfirmed against the unversioned arXiv
records on10 November2026. The previous review remains in README.md/sources.json.

* Korchinski et al.,2505.18651v2: a factorized semantic-attribute/co-occurrence
  model explains when linear word analogies arise. This does not justify reducing
  biochemical transformations or complete solution strategies to one offset.
* Shen,Druckmann,Zou,2605.11258v1: shared relational structure supports cross-domain
  scientific proposal generation. Novel proposals still need independent tests;
  a model-generated strategy is not an observed biological intervention.
* Minegishi et al.,2602.01992v5: controlled transformer tasks distinguish relational
  alignment and mapping application, with dependence on data,optimization and
  scale. A low geometric diagnostic is not a universal proof of good reasoning.
* Yasunaga et al.,2310.01714v3: generated exemplars can help reasoning in studied
  large models, but may be incorrect and performance depends on model scale.
  Generated traces therefore retain their evidence class and never become gold.

Grandis's diagram on printed pp.2–3 illustrates spaces with the same endpoints
and ordinary topology but different directed path classes. Sections1.4 and3.2
supply distinguished paths,composition and the fundamental category;2.1 supplies
positive chains and preordered homology;6.3 supplies weighted directed morphisms.
The book is used selectively, not claimed to have been exhaustively proof-checked.
The finite computational presentations here do not reconstruct a continuous
biological fundamental category or certify a causal intervention.

Continuous GFN theory [5] supplies the measure/kernel discipline: densities must
be defined on a compatible structured state space, with proper forward/backward
kernels and termination. Subtrajectory balance [6] is relevant to long reasoning,
but requires intermediate state-flow semantics. This change implements a complete
path-space TB objective instead of introducing an unqualified SubTB flow head.
The path-space state/reward construction,typed source rules and shared-prefix
conditioning below are local adaptations,not reproductions of those experiments.

## Full reasoning trajectories, not endpoint pairs

`trajectory_contracts.py` defines immutable typed facts,nodes,steps and a whole
reasoning graph. Each step states its action,required facts,immutable evidence and
one of four direction classes: computational,temporal,observational,interventional.
An interventional label requires a separate intervention-evidence reference, but
that reference alone still does not establish causal identification.

The compiler enforces a strictly forward layer order. It rejects cycles formed
by treating a repeated latent vector as the same time-indexed reasoning state.
Every state must be reachable from one explicit root. Genuine terminal,censored
and invalid states remain distinct. A resource-truncated leaf is not relabeled
as a completed solution or a complete GFN episode.

Fact availability is prefix-local. A derived fact can depend only on facts
available before its production. Future answers,sibling-branch results and
held-out targets cannot enter a prefix. Merging alternative construction histories
requires an explicit state-equivalence artifact, and retained facts must be
available on every incoming history. This is an equivalence merge,not an implicit
AND-join of unexecuted branches: combining distinct proof results must be an
explicit information-acquisition/composition step in a valid history.

Data preparation produces real native `GraphRecord`/`ConditionalTask` objects,
not just a readiness label. Ordered state nodes and directed action edges retain
fact-dependency fields,action prerequisites,direction-evidence classes and missing
next-state targets. Predicted/generated facts retain predicted evidence status.
A dominance attention mask is also available. This mask is a prepared artifact;
it is not falsely claimed to retrofit arbitrary dense attention. The history
encoder instead receives only the available prefix,so no future node is present.

A whole-trajectory analogy is a checked map between complete finite presentations.
Source steps may map to explicit target subpaths. Root,intermediate endpoints,
action words,defining squares and complete terminal trajectories are checked.
All branches are inspected up to explicit budgets; overflow refuses rather than
silently selecting the easy branches. One regression checks a three-stage graph
with eight six-step complete paths and twelve generators; deleting one required
square causes refusal despite unchanged final endpoints.

The verifier establishes correctness relative to the DECLARED presentations.
It neither discovers the semantic correspondence automatically nor turns a
computational analogy into biological causality. Complete train exemplars must
be group-disjoint from query/evaluation problems under the same declared policy;
completeness of the biological grouping policy remains a source-owner audit.

## Whole-history FoT inference

The existing `grow_forest` gains an optional backward-compatible history callback.
It receives cloned root-to-parent ancestry with stable node IDs. Default behavior
is unchanged. `trajectory_forest.py` verifies the entire root-to-candidate path,
not only the last vector,and binds its receipt to full history plus context.

Equal endpoint vectors reached by different histories remain distinguishable.
The regression fixture refuses an invalid high-score100 history while retaining
valid score1 histories. All proposal receipts,including refusals and abstentions,
are retained. The adapter exposes complete and censored paths separately.

This remains inference beam search. A beam-selected or rejected-and-renormalized
FoT trace is NOT mislabeled as an on-policy sample of the unmodified GFN kernel.
No modification to original endpoint GFN sampling is hidden in the FoT adapter.

## Complete-path GFlowNet training objective

The opt-in path-space state is

    h_t = (context, z_1, ..., z_t).

The terminal object is the ENTIRE fixed-horizon trajectory,together with its
conditioning context. Appending z_{t+1} uses a normalized native Gaussian kernel.
The backward construction deletes the last prefix entry deterministically,so its
Dirac mass is1 and log backward mass is0. This is NOT a Gaussian transition that
physically reverses a biological process.

Consequently the complete-path trajectory-balance residual is

    log Z_theta(c, path-task)
      + sum_t log p_theta(z_{t+1} | h_t, c, path-task)
      - log R(z_1,...,z_H | c).

The loss is the mean squared residual. Fixed-horizon final stopping has mass1.
The original endpoint-space learned backward model remains unchanged and is not
used for this path-space objective. Marginalizing over these paths does NOT
necessarily recover the old endpoint reward distribution: histories are distinct
terminal objects,not automatically quotient classes of equivalent answers.

The reward is deliberately integrable:

    R(tau | c) = q_0(tau | c) exp(-E(tau, guide)),  E >= 0.

q_0 is a normalized Gaussian random walk on Euclidean increments over a fixed
finite horizon. The coordinate transformation from increments to complete states
is triangular with unit Jacobian. Therefore0 < integral R <=1 whenever the finite
energy is defined. The guide's full pairwise geometry and optional persistence
penalty affect every stage,not just the endpoint. This mathematical bound is not
an empirical convergence or biological-quality result.

Guide trajectories are frozen,train-only and explicitly computed/model-generated
or synthetic fixtures. Their source,encoder,alignment,group-policy,scale and loss
weights are bound. The compiler does not invent an alignment between different
latent gauges or claim that declared groups prove global leakage safety. Reward
values are detached. Hidden target values cannot modify policy inputs or sampled
paths. Full-path states are also detached for score-function differentiation.

## Actual shared full-prefix conditioning

The new objective can use the original last-state policy as a diagnostic reference.
For production use,the native entry requires a complete-prefix adapter.
`NativePrefixContext` uses an EXISTING registered `GeometryConditioner` whose
TokenGT transforms are literally shared with the native `ProfileGFNPolicy`.
Every past latent state is a reasoning-state node; forward computational edges
preserve order. These latent coordinates are labeled generated,not molecular
positions. An explicit path-objective tag separates this task from endpoint TB.

Let c be the native biological context and g(h_t) the shared prefix-graph token.
The declared combination is

    c_t = c + alpha g(h_t) * (1 + c),

with coordinatewise multiplication and a frozen positive alpha. This retains
gradients to the biological context and shared prefix encoder while permitting
an objective-conditioned normalizer. It adds no parameters. It is an adaptation,
not a theorem of muP scaling; the coefficient and computation must be included
in numerical coordinate and complete-source objective qualification.

Sampling and rescoring use the SAME full-prefix conditioner and normalized
native transition parameters. Receipts refuse changed source inputs,policy or
history weights,configuration,trajectory contents and guide contents. A pruned
beam sample cannot claim native on-policy status. The original conditional
profile encoder,forward policy and normalizer receive gradients; the endpoint
backward head appropriately does not receive this objective's gradients.

Native owner tests at widths8/12/16 verify the actual explicit-muP factory,shared
operators,finite gradients and unchanged parameter/alias inventories. A separate
12-transition test exercises full-prefix sampling,rescoring and backward gradients.
These tests execute ZERO optimizer updates. They do not establish full muTransfer,
source-admitted biological training,large-GPU feasibility or learned improvement.

## Persistent-homology correction

A previous critical-edge tie-break produced a real invariance failure. On the
same square and latent configuration,simultaneous vertex permutations changed
loss from0.37031457505076204 to0.3657662350913716. The new implementation retains
ALL edges at persistence-critical filtration levels,which are permutation
invariant,not one lexicographically chosen edge. All24 permutations of the
fixture now give equivariant gradients and equal loss.

This tie-complete support is a documented adaptation of the critical-distance
objective inspired by Topological Autoencoders [7]. `GeometryAnalogyPolicy/v2`
binds it explicitly; old policy hashes must not silently stand for new math.
Pairing/level selection remains piecewise differentiable,not smooth across all
filtration changes. Equal barcodes still do not prove isometry or causal equivalence.
Distance-only geometry cannot distinguish enantiomers; ordered signed volumes,
periodic coordinates and typed directed relations remain separate constraints.

## Limits and work still required before production activation

The exact reference implementation is bounded: native trace presentations have
at most128 states,512 generators and64 path edges. The path-space reward accepts
at most63 transitions and64 latent dimensions,subject to the supplied existing
encoder's stricter exact identifier rank. The currently tested registered rank16
does not qualify longer than15-transition prefixes. Real workload token/graph
shapes and GPU resources remain to be measured. Fixed-horizon Euclidean path-space
sampling does not claim normalized kernels on raw angle tori or SE(3) manifolds.

No new original biological trajectory corpus was acquired or admitted. The data
compiler must be connected to qualified source traces with genuine provenance,
rights,group closure,observed masks and outcome semantics. Real guide and
cross-domain alignment quality remain unmeasured.

The complete native route/task schedule must freeze this objective's source,
encoder/guide/policy digests and coefficients before proxy selection. Checkpoint
integration must persist the explicit sampler generator state and source cursor
alongside the already existing optimizer/model/RNG recovery; this package's
transient same-process rollout receipts are not a fresh-process recovery claim.
No extra training ablation is authorized by the26-run cap. Evaluations that do
not update weights can compare inference controllers on the same held-out checkpoint.

At the current live review,full_dataset_ready remains false,construction/model
attestations are absent and the native owner still has an unconditional training
refusal. Full expanded80M counting,numerical operator qualification,source
admission and measured resource/recovery evidence are incomplete. These are
substantive scientific/implementation gaps,not permission to replace the gate
with a Boolean. No training or production default was enabled by this extension.

## Primary references

[1] https://arxiv.org/abs/2505.18651v2
[2] https://arxiv.org/abs/2605.11258v1
[3] https://arxiv.org/abs/2602.01992v5
[4] https://arxiv.org/abs/2310.01714v3
[5] https://arxiv.org/abs/2301.12594v2 — A Theory of Continuous Generative Flow Networks.
[6] https://proceedings.mlr.press/v202/madan23a.html — Learning GFlowNets From Partial Episodes.
[7] https://proceedings.mlr.press/v119/moor20a.html — Topological Autoencoders.
[8] Marco Grandis,Directed Algebraic Topology: Models of Non-Reversible Worlds,
    Cambridge University Press,2009; attached user book,selected sections listed above.
