# Hybrid complete-forest teacher and flow student

Implementation: `model/hybrid_forest_teacher.py` and
`research/hybrid_forest_routes.py`. This additive route reuses the native
`ProfileGFNPolicy`, `NativeForestTopologyHead`, shared geometry/backbone, and
`NativeForestFlowHead`. It adds **no learned parameters** and does not create an
optimizer, a worker, a run-budget exception, or an admission authority.

## Terminal object and measure

Given the original conditional problem c and R supplied root embeddings, let T
be a complete ordered rooted forest in the frozen finite DFS grammar. Roots
precede non-root vertices; a canonical trace uniquely constructs T. The output
is `(T, z)`, including *all* N(T)-R generated d-dimensional node embeddings, not
just a terminal answer. Use counting measure on the finite topology set and,
conditionally on T, Lebesgue measure on R^[d(N(T)-R)]. Root-only forests have a
zero-dimensional continuous factor of mass one. This is not a measure over
unlabelled isomorphism classes or independently verified biological mechanisms.

The joint policy is

    p_theta(T,z | c,roots) = p_top_theta(T | c,R)
       product_{i non-root} Normal(z_i-z_parent(i); mu_theta(a_i,c), diag(sigma_theta(a_i,c)^2)).

Here a_i contains only given roots and the root-to-parent ancestral prefix of i.
No hidden problem targets, future latents, sibling latents, unknown future
subtree sizes, or target-derived alignments enter a transition. Topology uses
only its already-built DFS prefix. All edges created by this sampler are marked
computational; no temporal/interventional causality is inferred.

**Proposition 1 (normalization).** Every conditional action kernel in the finite
DFS grammar is normalized; it completes in at most 2M-R actions. Every Gaussian
increment is normalized and depends only on previously available ancestors.
Consequently the joint forest law is normalized. **Proof.** Sum the finite
grammar by backwards induction. For each topology, integrate non-root latents
in reverse topological order; each Gaussian contributes one. The change from
parent-relative increments to node values is triangular with unit determinant.
Finally sum the normalized topology masses. The root-only case is the empty
product and is included, not rejected.

## Fixed integrable reward, not an unconstrained score

Let q_top(T) be the product of uniform probabilities over each prefix's allowed
DFS actions. It is normalized but is generally *not* uniform over final trees.
Let q_0(z|T) be a centered independent Gaussian density on parent-relative
increments, with a fixed positive reference scale. Define

    R(T,z|c) = q_top(T) q_0(z|T) exp(-E(T,z,c)).

E is nonnegative and uses a separate frozen native criterion scorer. It is the
mean softplus of negative whole-forest logits over explicitly supported
criteria, plus a fixed nonnegative weight times the corresponding mean over
non-root node logits. Unknown criteria are not silently labelled bad; the
supported criterion subset and source label policy must be explicitly bound.
This is a **surrogate preference reward**, not hard constraint certification.
A frozen untrained scorer is useful only as a correctness fixture, never as an
admitted scientific reward. Production requires a qualified source-trained
checkpoint, source/group/encoder commitments and declared applicable criteria.
The scorer must not share parameter storage with the changing policy/student.

**Proposition 2 (finite positive partition).** If E is finite and measurable
almost everywhere, then 0 < Z(c) = sum_T integral R(T,z|c) dz <= 1.
**Proof.** q_top q_0 is a normalized positive reference law by Proposition 1.
The factor exp(-E) lies in (0,1] almost everywhere, so its expectation is strictly
positive and at most one. This does not assert a useful numeric lower bound.
Overflow/nonfinite numerical evaluations are refused without clipping, rejection
resampling or treating the failed sample as a successful output.

## Whole-forest trajectory balance

Canonical topology construction followed by canonical ancestral increments has
one construction trajectory per terminal forest. Reverse construction deletes
the last recorded operation deterministically; its mass is one. Forced finish
is also mass one. The loss is

    L_TB = mean_samples [log Z_theta(c,roots) + log p_theta(T,z|c,roots) - log R(T,z|c)]^2.

The normalizer sees original conditions and given roots **only**. It does not
receive sampled topology, node count, terminal values or sample-specific quality.
Samples and frozen rewards are detached; gradients enter the exact registered
forward Gaussian kernels, topology policy, normalizer and shared transformer.
Existing profile endpoint losses, backward/stop heads and their semantics are
not rewritten by this additional objective.

**Proposition 3 (conditional zero-loss implication).** If the joint family can
represent R/Z and a zero residual is achieved almost everywhere under a
full-support sampling law, then its terminal law equals R/Z. **Proof.** Exponentiate
the zero residual, obtaining p_theta = R / exp(log Z_theta); integration and
normalization identify exp(log Z_theta)=Z. These are essential assumptions,
not a convergence claim. In particular, the ancestor-only factorization may
be unable to represent arbitrary sibling-coupled global rewards exactly.
Minimizing a finite sampled loss does not establish the premise, correctness,
calibration, diversity, or downstream performance.

## Conditional flow student and inference separation

For every unedited teacher sample, reuse its generated topology and all free
latents as a detached conditional flow-matching target. The existing Gaussian
path, sigma_min, coordinate masks, root clamps and native velocity head are
unchanged. Each forest contributes its own coordinate-mean loss; root-only
samples contribute zero rather than being dropped or reweighting the topology
law. Loss coefficients are fixed in the route declaration. A joint optimizer
step differentiates the teacher and student objectives through their shared
registered parameters; neither loss differentiates through random samples.

The student learns the **current teacher's** conditional samples. Equality with
the reward law additionally needs teacher qualification/convergence,
representation capacity and flow approximation assumptions. At positive sigma_min,
the terminal conditional Gaussian path is the teacher target convolved with that
residual Gaussian noise, not the unsmoothed teacher law. The implementation
does not assign a tractable density to the distilled flow or decoded tokens.
Nearest-token projection has a many-to-one pushforward distribution and must
not inherit a continuous density claim. Few-step distillation is still a
separately preregistered stage, not an extra uncounted training run.

FoT pruning, subtree edits, reranking and post-decode corrections remain
inference operations. Their selected/edited outputs cannot be substituted into
this on-policy batch. Complete batch fingerprints bind actual weights and
buffers, all generated forests, roots, method, context and generator states;
modified, reordered, subset, beam-pruned and stale batches refuse.

## Native update and recovery contract

`HybridForestTrainingRoute` freezes original train-only conditions, ordinals,
source/encoder/group/root-policy identities, reward checkpoint digest, candidate
count, objective weights, repeat budget and resource limits. One route window
consumes one original condition, not one generated candidate. It uses the
existing `Mora3.ForestTrainingCursor/v1` and native checkpoint schema v6 with an
explicit opt-in `Mora3.NativeForestObjectiveBinding/v3`. v1/v2 bindings retain
their previous exact method contracts; the three schema/method/skeleton tuples
cannot be interchanged.

`native_engine_hybrid_forest_loss` checks full owner admission *before* reading
route inputs, uses the existing engine-owned generator, and restores owned and
global RNG on a pre-update failure. Preparing a loss does not advance a cursor
or optimizer. `run_native_hybrid_forest_update` passes that loss and exact
before/after cursor to the existing native engine. After entry into an optimizer
update, only qualified counted same-run checkpoint recovery may resume; the
adapter does not silently retry or pretend to roll back a partial update.

No campaign release is added by this implementation. The 24 proxy slots and
at most two full-model slots remain the only substantive runs. The frozen
reward's immutable checkpoint is an external pinned input to recovery, not a
second optimizer-owned live model. Independent full-owner/source/rights/split,
real-data replay, complete-width/operator, actual resource and recovery
qualification remain mandatory before any campaign launch.

## References and scope of derivation

- *A Theory of Continuous Generative Flow Networks*, arXiv:2301.12594:
  measure-theoretic continuous/hybrid GFN foundation.
- *Trajectory Balance: Improved Credit Assignment in GFlowNets*,
  arXiv:2201.13259: trajectory-balance objective and conditional zero-residual
  characterization.
- *Flow Matching for Generative Modeling*, arXiv:2210.02747: conditional
  probability paths and velocity regression.

The canonical forest measure, ancestral conditioning composition, frozen
criterion reference reward and native adapter above are project-specific
constructions. The cited papers do not establish Mora-3 empirical improvements.
