# Geometry, directed topology and analogical reasoning: implementation review

Reviewed 10 November 2026. This is implemented, tested research infrastructure,
not evidence of a trained reasoning improvement or a campaign launch approval.
The active campaign is capped at **24 proxy runs + two approximately80M ordinary
full-model runs with transferred hyperparameters and retained muP scaling**.
There are no additional SP baseline, intermediate-width or diagnostic training
runs in this task. All verification reported here used zero optimizer updates.

## Source scope and scientific assessment

The four requested papers were reviewed in the versions below. The attached
Grandis monograph was reviewed selectively for the definitions and constructions
used here: Sections1.1/1.4,2.1/2.2,3.2 and6.1/6.3, including its diagrams. This
is not a claim to have checked every proof on all446 PDF pages. The uploaded
book has SHA256 f80980597a19a1b43224e990143c1ad94ad4c583f1eeec11419bcd508bb362d0.
No copyrighted book text was copied into the repository.

**[1] Korchinski et al., 2505.18651v2.** Factorized semantic attributes yield a
tractable co-occurrence spectrum; dominant attribute modes and PMI explain
linear analogies under the model's assumptions. This supports testing relational
subspaces, not assuming every biological intervention is an additive embedding
offset. Polysemy, dependent attributes and domain-specific geometric symmetries
limit that inference. The official reproduction repository was pinned at
`a03c36ea4c7a5fda35cd617ec1eda0554b3ac6a3`. Its co-occurrence script was inspected;
it allocates dense vocabulary-by-vocabulary arrays and uses pickle-backed source
artifacts. It was not executed or vendored, and its production suitability is
not established by this review.

**[2] Shen, Druckmann and Zou, 2605.11258v1.** The proposal method explicitly
separates cross-domain object mappings and shared relations from solution
transfer. It is useful as a search-expansion principle. Its human evaluation
reports78% preference for AR novelty, but reasonableness67% versus86% for the
cross-domain comparator, with lower agreement on AR reasonableness. Thus novelty
is not an acceptance test. The four biomedical demonstrations are task-specific;
model-agreement or distributional metrics are not causal/clinical validation.
No general biological-success guarantee is transferred into Mora-3.

**[3] Minegishi et al., 2602.01992v5.** The controlled relational tasks suggest
structural alignment and learned mapping application, with strong data and
optimization dependence. Mechanistic signals and nonmonotonic scaling motivate
separate relation, geometry and task-performance measurements. They do not prove
that minimizing a scalar Dirichlet energy creates analogy. The official MIT
repository was pinned at `291dc97ead7c5a3d9781518aae8d4a3f9560e6cc`; its inspected
Dirichlet-energy implementation converts tensors to Python scalars and normalizes
by embedding norm, returning zero for zero norm. That is an analysis diagnostic,
not a ready differentiable biological training objective. Collapse is separately
reported here. The February code pin is not proof that every later v5 experiment
is reproduced by that checkout.

**[4] Yasunaga et al., 2310.01714v3.** Self-generated knowledge and exemplars
improve several prompting benchmarks, but can contain irrelevant or incorrect
solutions. The paper's own scale comparison favors retrieved exemplars for
weaker models. It does not demonstrate this capability in an80M multimodal
biological model. Mora-3 therefore receives input-only, split-aware memory
retrieval and explicit structural validation rather than generated exemplars
being relabeled as experimentally observed supervision.

Related primary work: Moor et al. [6] supplies the persistence-critical-distance
loss principle; Bodnar et al. [7] motivates nontrivial relation-specific transport
rather than indiscriminate smoothing; Chowdhury and Memoli [8] explains why
symmetrizing directed networks loses information; Altabaa et al. [9] motivates
separating relational information from object attributes. EGNN [10] is relevant
to spatial equivariance, but was not installed as a replacement for the native
geometry/UMA/eSEN owners. Learned Abstractor and Neural Sheaf Diffusion modules,
full persistent path homology and new equivariant backbones are NOT claimed as
implemented in this change.

## Mathematical design and its boundaries

An analogy is a map of an explicit source relational presentation to a target
presentation, not just `z_b - z_a + z_c`. Objects map to objects; generating
arrows map to directed paths. A complete mapping on an explicitly chosen
subpresentation represents a partial-domain analogy. Parallel paths are equated
only by supplied, evidence-bound square relations.

If generator images have compatible endpoints and preserve every defining
relation, their extension by concatenation descends to a functor on the presented
path categories. This follows because each generating rewrite has the same image
class, hence so does its closure under composition. It does not establish that
an inferred biochemical relation is true, that the functor is full/faithful, or
that a continuous biological state space has been reconstructed.

A natural-transformation witness checks on each generator `a:x->y`:

    eta_y o F(a) = G(a) o eta_x.

This extends to composed paths. In the book's categorical setting natural
transformations express directed homotopies, but this implementation does not
infer a corresponding continuous homotopy of molecule or tissue spaces.

For the finite square2-complex, integer boundary matrices satisfy `B1 B2 = 0`.
The positive cone consists of nonnegative generator coefficients. A supplied
positive homology witness is checked exactly as `z - p = B2 b`, `B1 p = 0`,
`p >= 0`. A positive chain map obeys `B_target F = F B_source`; a supplied chain
homotopy obeys `G-F = dH+Hd`. Rational Betti numbers are reported separately;
integral torsion and the entire homology preorder are not reconstructed.

Metric geometry uses complete source-unit distances and explicit frozen scales.
For model width `w`, latent distance is `||z_i-z_j||_2 / sqrt(w)`. This normalizes
coordinate count, but is NOT a theorem that the new loss is muP-compatible.
The geometric objective combines full pairwise metric stress with a persistence-
critical-edge loss. Critical edges are selected from source and latent H0/H1
reductions; autograd traverses the selected distances, not discrete pairings.
The result is piecewise differentiable away from ties/pairing changes. It is NOT
a Wasserstein or bottleneck loss, and the loss is not guaranteed to recover from
an exactly collapsed stationary embedding without the main native objective.

The physical target geometry is rigid-invariant; the scalar native payload
encoder itself has NOT thereby become SE(3)-equivariant. Pair distances also
identify mirror images. An ordered signed tetrahedron volume is exposed for
chirality-sensitive comparisons, and periodic torsion distances use wrapped
coordinates instead of linear angle subtraction. Anchor identity/order and
all coordinate-frame transformations remain source responsibilities.

GRNs are not assigned invented3D coordinates. Activating/inhibiting labels remain
directed relations; nonnegative path cost is a separate field with its own unit.
The sheaf-inspired residual `||R_u z_u - R_v z_v||^2` accepts explicit caller-owned
restriction maps; it does not make inhibitory neighbors equal or silently add
learned operators. Latent matrix-composition loss is applied only to declared
squares, never to arbitrary pairs of noncommuting interventions.

## Implemented modules

- `persistent_topology.py`: exact bounded F2 Rips H0/H1 column reduction, complete
  bottleneck matching including diagonal/essential bars, and differentiable
  critical-distance objective. Cutoff-unpaired bars are right-censored, not
  essential. H2 is deliberately unavailable without tetrahedral death events.
- `directed_topology.py`: evidence-bound directed arrows/squares, path composition,
  bounded rewrite proof search, opposite presentations, asymmetric costs,
  generator/path functors, positive chains and exact chain-map/homotopy checks.
- `geometry_analogy.py`: actual shared GeometryConditioner/TokenGT node states,
  native masks/units/frames, physical and periodic distances, chirality utility,
  metric/topological losses and explicit relation-transport residuals.
- `native_relations.py`: condition-only GraphRecord projection and checked
  natural-transformation squares. It adds no reversed edge or inferred2-cell.
- `analogy_control.py`: group-disjoint train-memory retrieval, combined persistence
  and directed-map checks, latent operator coherence and a wrapper around the
  actual existing FoT forest that retains all acceptance/refusal/unknown receipts.

The FoT certificate is bound to exact candidate-state bytes and context. A
high-scoring invalid or unresolved candidate is not rescued by barcode similarity.
This is inference-only search pruning, NOT a changed GFN sampling kernel or an
on-policy GFlowNet training distribution. Existing latent GraphCG directions
remain available as proposals; they are not treated as proof of an analogy.

## Native training integration contract

`geometry_regularized_native_loss` composes an existing admitted native objective
with the geometry objective on an explicitly registered owner module. It calls
`SourceBoundOwner.require_training_admission()` first and cannot open that gate.
No second model or optimizer is created, and no parameter is added by the loss.
The exact module path, source/scale/policy digests, objective weights, selected
source roles and evaluation denominators must be frozen before the24-run grid.

The caller must supply fully admitted known-topology records. Coordinates used
as a geometric reference must be observed condition fields, never targets,
predicted measurements or unknown zeros. Unknown topology generation needs its
own contract. Source record IDs and hidden target values cannot alter these
auxiliary inputs. Native training remains disabled until the full source/model
requirements are genuinely satisfied; an API implementation is not a completed
campaign route/schedule qualification.

## Measured verification

The combined focused suite passed **261 tests plus56 subtests**, with zero skips,
zero optimizer updates and no GPU allocation by this work. It includes earlier
SCRI and campaign26 tests, full-owner construction/checkpoint checks, the native
worker/pin-store regression tests and the new topology suite. This is not a claim
that the entire project test suite was run.

New checks include independent installed-GUDHI diagram/bottleneck comparison,
permutation invariance of diagrams, exact integer boundary/chain identities,
positive-cone counterexamples, naturality, target/ID isolation, wrong native units
and frames, finite gradients at widths8/12/16, torus periodicity and chirality,
noncommuting operator order, complete26-slot/failure accounting, and an actual
FoT regression in which an invalid score100 proposal loses to a verified score1
proposal. These are mathematical/software fixtures, not learned biological results.

A single-thread CPU synthetic profile measured:

| Points | Rips vertices/edges/triangles | Seconds | Cumulative peak process RSS |
|---:|---:|---:|---:|
|8|92|0.000261|37,056,512 bytes|
|16|696|0.001846|37,056,512 bytes|
|32|5,488|0.023774|38,891,520 bytes|
|64|43,744|0.228945|52,457,472 bytes|

This reference defaults to64 complete points and50,000 simplices; larger inputs
are rejected, never silently truncated. It is not a measured whole-protein,
large-tissue or whole-GRN training throughput claim. Physical-domain support
requires actual representative length distributions and resource qualification.

## Evaluation inside the26-run cap

Freeze auxiliary policies before the existing24-trial LR/readout/attention grid.
Use the same masks, data/splits, geometry scaling and measured denominators at
all widths. Select only through the existing paired-seed validation selector.
The two target runs retain the selected hyperparameters; no80M retuning or new
training ablations are authorized. Actual complete expanded-head parameter
counts must be measured; the earlier80,304,407 count omits optional source heads.

At existing validation checkpoints record ordinary native task losses, analogue
transfer errors, direction/relation violations, chirality failures, persistence
and metric distortion, collapse indicators, and abstention coverage. Train-only
memory must be disjoint under the actual globally qualified biological groups.
Chemical scaffold/protein family/donor/study/perturbation grouping follows each
source's scientific unit, not filename-based random splitting. Compare inference
controller enabled/disabled on the same fixed checkpoint without updates; this
measures inference control only, not a causal training-ablation benefit.

## Current remaining launch gates

The current full dataset flag and two owner signatures remain incomplete. The
native owner still unconditionally refuses training. Required source rights,
complete native axes/masks/global splits, expanded routes, all-head numerical
coordinate checks, optimizer/recovery/resource evidence and exclusive-worker
release are not replaced by this package. The tasmorpho historical summary is
not an independently regenerated experiment. The final two full-run update/time
budgets and actual expanded80M owner are not frozen. **No campaign was started.**

## Primary references

[1] https://arxiv.org/abs/2505.18651v2
[2] https://arxiv.org/abs/2605.11258v1
[3] https://arxiv.org/abs/2602.01992v5
[4] https://arxiv.org/abs/2310.01714v3
[5] Marco Grandis. Directed Algebraic Topology: Models of Non-Reversible Worlds.
    Cambridge University Press,2009. User-provided PDF; cited sections above.
[6] https://proceedings.mlr.press/v119/moor20a.html
[7] https://arxiv.org/abs/2202.04579v4
[8] https://arxiv.org/abs/1701.00565v2
[9] https://arxiv.org/abs/2304.00195v4
[10] https://arxiv.org/abs/2102.09844v3

No third-party model checkpoint or paper implementation was installed, copied or
run. The code here is a local, bounded adaptation with independent mathematical
oracles and explicit limitations, not a reproduction claim for all cited papers.
