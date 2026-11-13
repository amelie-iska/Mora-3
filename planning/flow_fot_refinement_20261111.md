# Flow-seeded, multi-output forest refinement — 11 November 2026

This addendum implements the user's request for several initially promising complete
latent reasoning forests, subsequent FoT-style correction, and more than one final
quality-qualified output. It is an inference component, not a training completion.

## Source boundary
Bi et al., Forest-of-Thought, ICML2025 (arXiv2412.09078v3), sections3.1–3.3,
Algorithms1–2, specifies multiple trees, sparse layer-wise selection, score-triggered
rule/knowledge-assisted self-correction and consensus/expert decision. It does NOT
specify a latent flow head, arbitrary subtree crossover, or a top-k multi-output API.
Those are Mora-3 adaptations; do not cite them as literal paper algorithms.
References: https://proceedings.mlr.press/v267/bi25a.html and
https://arxiv.org/html/2412.09078v3 .

## Implementation contract
Generate a batch of complete forest embeddings through the existing Euler-flow
solver and a caller-bound conditional velocity field. Explicit generator state,
context/coordinate revisions, full skeleton and known-node clamps are retained.
Free template entries are ignored: neither a clean future target nor a held-out
answer is a sampler argument. A fixed skeleton is declared/generated, never a
secret target skeleton. This component adds no learned parameters and is not a
claim that the unfinished native trajectory velocity head is already trained.

Score every full forest with a separate, content-bound verifier. Preserve score,
valid/refused/unknown disposition, evidence, evaluator revision and all proposals.
Rank valid candidates by score with explicit whole-forest diversity; a high score
cannot override failed constraints. Low-quality repairable candidates may enter
correction but may not be returned as successful outputs.

Make non-destructive branch/subtree revisions. Retain all unedited ancestors and
unrelated branches, clamped observations, context and latent coordinate revision.
Replacing an ancestor replaces its complete descendant subtree; every edited
forest receives fresh whole-forest verification. Multiple proposed revisions fork
new candidates; sparse activation never erases the original audit history.
Original valid candidates remain available, preventing a bad edit from silently
replacing a better result. No graft across incompatible histories or purported
causal equivalence is inferred from proximity.

Return K>=2 distinct decoded outputs only when that many pass the declared score
and validity criteria. Decode only complete candidate forests; check the decoded
objects again and deduplicate both exact token sequences and explicit canonical
output identities. Optional post-decode correction has its own finite round cap
within the same total candidate budget. Return a transparent shortfall rather than
duplicate padding or fabricated successful candidates. Consensus is a diagnostic,
not a rule that collapses every task to one answer or certifies correctness.

Inference top-k, pruning, edits and decoding alter the distribution. Keep raw
proposals separate; mark selected outputs as not on-policy GFlowNet rollouts.
No Gaussian log probability is invented for a deterministic ODE. Frozen scorer,
decoder and context revisions accompany lineage. Best-of-N, edits and decoder
verification count toward inference compute. They consume no training runs when
weights are unchanged. The existing24+2 run cap and all admission gates remain.

## Tests and visual evidence
Tests first: exact clamping, generator replay, all-node rather than endpoint
fingerprints, complete subtree replacement, immutable inputs, malformed/sibling
edit refusal, invalid high-score rejection, elite retention, multiple independent
outputs, decode failure repair, strict budgets and explicit insufficient-quality
results. A deterministic arithmetic fixture will exercise the actual controller;
its scores and graphic must be labeled synthetic software checks, not learned
quality. Add an interactive lineage explorer and matching static SVG to the local
project and manuscript. Record observed results only after execution.

## Implemented scope and measured software checks

`src/mora3/reasoning/flow_forest_refinement.py` implements the sampling, immutable
revision and multi-output controller.39 new CPU tests pass; the focused regression
covering existing FoT, trajectory-GFN, persistence and Gaussian-flow components
passes123 tests with3 CUDA-only skips because CUDA was deliberately hidden.
A rename-only diversity failure was reproduced and fixed: corresponding complete
layouts compare all node coordinates by L2 even when revision node IDs change.
Different structural layouts require a qualified custom comparison for semantic
alignment; treating them as distinct strata does not certify meaningful novelty.

The field, scorer, editor, decoder and decoded verifier are explicit callbacks.
Their registered native model/source bindings and empirical quality are NOT
completed by this controller. No learned weights or full-owner parameter counts
change in this package. Native source admission and the unchanged26-run campaign
remain separate. Pure tests and the synthetic visualization use zero optimizer
updates. Source-owner ledgers and ongoing M-CSA/Tox21 work remain untouched.
