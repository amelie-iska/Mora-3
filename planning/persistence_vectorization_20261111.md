# Persistent-homology vectorization and L2 integration

Date: 2026-11-11. User-directed extension of the geometry/trajectory-analogy work.
Scope: one bounded implementation/integration with reference tests, interactive
visualization and matching manuscript graphics. No optimizer update, campaign
reservation, training admission, dependency installation or benchmark tuning.

## Scientific contract

The requested speed improvement must not silently replace exact evidence with a
compressed feature vector. Bubenik's complete landscape is invertible for finite
persistence diagrams (modulo the usual zero-length interval convention). Fixed-grid
sampling, finite rank truncation and finite pixel images are not generally
lossless. Persistence itself is not a complete invariant of the original metric,
biological structure or directed reasoning graph. We preserve the original native
`PersistenceResult`, including all dimensions, interval multiplicities, pairing
simplices, coefficient field, filtration radius and essential/censored status.
Directed path/2-cell presentations, biological provenance, units, masks, source
versions and chirality constraints remain separate; no symmetrization is inferred.

## Implemented numerical paths

`src/mora3/reasoning/persistence_vectors.py` supplies:

* Batched tensor persistence landscapes, computed as the order statistics of
  `max(0, min(t-b_i, d_i-t))`, with complete finite rank depth by default.
* Sampled landscape vectors with square-root trapezoidal quadrature weights.
  Euclidean comparison therefore approximates functional L2 rather than changing
  simply because a grid has more samples. Cropped support is rejected; optional
  rank truncation and all sampling are explicitly labeled approximate.
* Exact finite-landscape L2 features on a shared adaptive basis. The basis includes
  every birth, midpoint, death and active crossing between an ascending and a
  descending tent segment. On each interval the landscape difference is linear,
  so two-point Gauss quadrature integrates its square exactly in real arithmetic.
  All ranks are retained. Float64 rounding remains; unrepresentable peaks or
  integration nodes refuse. Features from independently constructed bases must
  never be compared without recomputing their common basis.
* Gaussian persistence images in birth/persistence coordinates, using positive
  persistence weights and separable normal-CDF rectangle integrals. This avoids
  an intermediate batch × bars × height × width tensor. Standard deviations,
  pixel edges, pixel masses, density-normalized vectors and weighted mass outside
  the image domain are explicit. Positive tails use survival-function differences
  to avoid avoidable cancellation. Pixel-mass Euclidean distance differs from
  piecewise-constant image-density L2 (`mass / sqrt(pixel_area)`). Neither feature
  preserves the full barcode or equals bottleneck/Wasserstein distance.
* Stable vectorized pairwise Euclidean distance without the cancellation-prone
  squared-norm expansion for large common offsets.

All paths reject whole requests exceeding declared budgets; no hidden landmark
subsampling, interval deletion, infinity clipping or rank truncation is permitted.
Finite features have no claim of preserving an essential interval. Native ranking
requires identical essential birth multisets; incomplete/right-censored results
cannot be treated as completed evidence. All original records remain available.

## Real integration point

`analogy_control.rank_topological_memories` now defaults to `metric="landscape_l2"`.
It calls the batched exact finite-landscape implementation for all eligible
training memories; no approximate shortlist can silently lose a nearer candidate.
`metric="bottleneck"` remains an explicit compatibility mode. The scores are
reported under a versioned metric name and use immutable-ID tie-breaking.
Condition-only roles, train-only memory, disjoint biological groups and one
source grouping/filtration contract remain required.

The acceptance verifier `certify_analogy` is unchanged: directed generator,
composition and 2-cell checks and its original complete-diagram bottleneck
criterion still apply. A bottleneck tolerance is not silently reinterpreted as
L2. Retrieval is not causal or biological validation. No GFlowNet transition
kernel, trajectory reward, native training schedule or owner gate is changed.
The earlier differentiable persistence-critical geometric objective is retained;
this vectorization does not pretend the combinatorial PH reduction disappeared.
Fixed-grid landscape and image tensor functions support piecewise gradients,
checked numerically away from ties. They are not a proof of whole-model μTransfer.

## Visualization and manuscript

`docs/project/persistence-vectors.html` shows original diagrams, complete and
sampled landscapes, integrated images and their different L2 values. Resolution,
rank depth, a comparison shift and an easily missed short interval are real
controls. Export retains the original intervals. It is a finite mathematical
fixture, never a biological evaluation or trained reasoning example.
`docs/project/persistence-vector-math.js` is the browser/Node numerical renderer;
its values are independently checked against the Python tensor implementation.
The matching SVG appears in the live paper and Markdown manuscript. The earlier
seven-view flow-forest explorer and all existing public visualizations remain.

## Qualification and performance criteria

Tests cover independent scalar order statistics, installed GUDHI landscapes
(with its sqrt(2) convention accounted for), exact analytic triangle integrals,
all-rank crossings, common-basis invariance, independent Gaussian CDF integrals,
empty diagrams, multiplicities, padding, large offsets, tails, gradient checks,
resource refusal, censoring, essential mismatch and source/group leakage rules.
Explicit counterexamples show a positive interval missed by coarse sampling and
two distinct diagrams with equal one-pixel images. Such tests prevent a false
losslessness claim. Browser QA must exercise controls, exports, mobile layout,
console errors and the matching static SVG, not merely render a screenshot.

Measure vectorized versus scalar reference feature construction on named finite
fixtures, report workload, CPU threads, median times and numerical disagreement.
These measurements are NOT full-corpus PH speedups or model-quality evaluations.
The exact native width/operator/source/recovery/resource release and the fixed
24+2 campaign remain separate. Frozen training contracts must explicitly bind
any new retrieval/objective policy before activation. All existing 26-run limits
and the ban on additional unallocated post-training comparisons remain in force.

## Primary sources

1. Peter Bubenik (2015), Statistical Topological Data Analysis using Persistence
   Landscapes. JMLR 16(3), 77–102. https://jmlr.org/papers/v16/bubenik15a.html
2. Peter Bubenik (2018), The persistence landscape and some of its properties.
   https://arxiv.org/abs/1810.04963. Invertibility, exact/discrete distinction,
   tropical rational structure and metric nonequivalence.
3. Henry Adams et al. (2017), Persistence Images: A Stable Vector Representation
   of Persistent Homology. JMLR 18(8), 1–35.
   https://jmlr.org/papers/v18/16-337.html. Pixel integration and stability are
   not claims of arbitrary fixed-grid invertibility.
4. GUDHI representations documentation; the installed 3.12.0 implementation is
   used only as an independent test oracle. No package was installed or upgraded.
   https://gudhi.inria.fr/python/latest/representations.html
5. Grandis (2009), Directed Algebraic Topology, the user-supplied reference:
   non-reversibility, directed path categories and positive-chain structure remain
   independent of symmetric Euclidean persistence features. The book is not
   copied or published with this source release.
