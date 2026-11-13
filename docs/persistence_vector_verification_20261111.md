# Persistence-vector and integration verification — 11 November 2026

This source integration combines the retained trajectory/topology/campaign26 work,
the staged native29 source-factory components, and the new vectorized persistence
representations. It preserves SCRI registration and the closed native training
admission boundary. It is not a completed training campaign or biological result.

## Exact retention and metrics

The original persistence records remain available, including interval multiplicity,
birth/death simplices, coefficient field and essential/censored status. Full finite
landscapes are evaluated on a common set of all linear breakpoints and integrated
with two-node Gaussian quadrature per interval. Their vector Euclidean distance
is the finite-landscape L2 distance in exact arithmetic, subject to float64 rounding.
Different batches require a common comparison basis; this is not a universal
fixed-dimensional injective embedding for arbitrarily many bars.

Fixed-grid/rank-limited landscapes and finite persistence images remain explicit
approximations. Images use Gaussian rectangle integrals, not pixel-center samples.
Image-mass L2 and piecewise-constant image-density L2 are separately named. The
whole source diagram is never replaced by a grid. The native memory path excludes
held-out/shared-group candidates and incomplete or incompatible essential-bar
records. Original bottleneck structural acceptance remains available and distinct.

## Workstation tests

The first targeted run passed55 tests, including independent scalar references,
installed-GUDHI landscape comparison, browser-JavaScript versus Python/SciPy
numeric agreement, native SCRI coexistence and main-site links. A broader focused
run passed382 tests and56 subtests. Those sets overlap and must not be added as
independent coverage. The broader run used a process-local guard that forbids
real optimizer steps; it recorded zero attempted optimizer steps.

The actual installed Chrome browser passed six persistence-interface tests and
nine flow-forest tests, with keyboard/hash navigation, mobile layout, all controls,
SVG/JSON exports and absence of external page requests. Eighteen JavaScript
mathematical tests also passed. Screenshots were reviewed on desktop and mobile.
This is not a claim that the whole repository suite or every possible UI state was
executed. These tests are mathematical, software and source-boundary evidence,
not learned quality or empirical hyperparameter-transfer evidence.

A real cross-runtime defect was reproduced: Node and Chromium differed in the
last binary rounding bit of Math.exp, producing different SVG coordinate strings.
Only geometry attributes are now serialized to nine decimal places (at most
5e-10 viewBox units). Underlying numerical calculations and reported solver errors
are unchanged. The same renderer now produces identical stored and interactive
SVGs on the tested runtimes. Original failing and corrected test receipts are kept
on the workstation.

## Measured CPU feature-construction reference

Synthetic workload:8 diagrams,24 finite intervals each, float64, one CPU thread,
257 landscape grid points or24x24 image pixels. Two warmups and seven measured
repetitions per method. The comparator is a straightforward independent Python
scalar oracle, NOT an optimized persistence package or the neural training loop.

| Representation | Scalar oracle median | Vectorized median | Ratio | Maximum absolute error |
|---|---:|---:|---:|---:|
| Sampled landscapes |23.970 ms|0.235 ms|101.8x|0|
| Integrated images |19.557 ms|0.365 ms|53.6x|1.11e-16|

A separate complete-landscape reference for four diagrams/eight bars each used
120 knots and1904 vector coordinates and took0.366 ms median. Peak process RSS
was537,010,176 bytes, including the Python/Torch runtime. No GPU was used. These
measurements cover feature construction, not persistence reduction, source loading,
large-scale retrieval throughput, whole-model speed or a guaranteed acceleration
on another workload/device.

## Publication and launch boundaries

The publication contains reviewed source, tests, plans and mathematical figures;
not dataset payloads, model weights, private specifications, credential files,
environments or private progress logs. Source/model attestations and full dataset
admission remain incomplete. No source signature or admission allowlist entry was
fabricated. New forest-flow neural heads, learned retrieval, vocabulary migration,
decoding, distillation and adapters remain at their separately stated implementation
stages; a visualization does not make them trained capabilities.

The governing allocation remains24 proxy trials and at most two target runs.
Additional distillation/preference optimization and trained QLoRA comparisons are
not silently added. No optimizer update, sweep or target training was performed
by this verification. Final held-out model evaluations require real checkpoints.
