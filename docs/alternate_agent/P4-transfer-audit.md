# Package 4 — independent transfer and baseline audit

## Accounting and implementation checks

The existing user plan accounts for 44 runs: 24 primary μP proxy trials, eight
standard-parameterization proxy comparisons, nine transfers (two intermediate
widths and the target, three seeds each), and three naive SP target runs. The
22 GPU-hour figure is a combined upper budget, not a runtime estimate; substantive
full training is a separate authorization.

The primary grid remains learning rate 0.0001/0.0003/0.001, readout multiplier
0.5/1 and attention-key multiplier 0.5/1, with seeds 2026110201/2026110202.
The existing primary constructor/update code applies μP conventions. Independent
checks of actual optimizer groups at three tiny widths confirmed all 48 unique
trainable groups per width: matrices with two width axes receive inverse last-
width-ratio LR and ratio-scaled decay under the default decay policy; other
parameters retain LR. This does not prove every full-owner custom operator or
coordinate trajectory is qualified.

**The SP comparison path is not qualified.** The current native update primitive
requires an exact MuAdamW. An SP baseline cannot be obtained by changing a label
on the same μP readout/attention/initialization. The audit proposes four explicit
baseline learning rates (0.00003/0.0001/0.0003/0.001) and the two primary seeds,
standard fan-in initialization, ordinary AdamW, standard readout and inverse-
square-root head-dimension attention. This proposal is UNFROZEN and requires the
coordinator's separate implementation/release. Eight trials are a limited search,
not an equal exhaustive optimization budget claim.

Transfer seed candidates are recorded separately, but no actual transfer width,
chosen hyperparameter or source commitment is invented. Registered fixture model
counts elsewhere in this handoff do not freeze the final expanded architecture.

## Native validation-only selection

The independent wrapper reuses the actual native `select_proxy_transfer`, not a
second competing selector. All 24 distinct terminal primary artifacts must be
present. A hyperparameter point is eligible only when both fixed seeds provide
matched, predeclared validation source commitments and observed-target counts at
the same comparison exposure. Failed/OOM/interrupted dispositions remain visible
and cannot contribute a successful single seed as a complete point.

Per task, normalize `(loss - initial reference loss)` by
`max(abs(initial reference loss), declared minimum scale)`. The declared source-
family aggregation is applied before the fixed-seed mean. Ties are broken by
summed GPU seconds and then HP identifier. Target/test results cannot select the
winner or justify target retuning under a zero-shot label.

New synthetic outcome tests exercise the complete 24-artifact selector, failure
retention, and refusals for test contamination, changed denominators/exposure,
duplicate artifacts and incomplete campaigns. They are algorithm tests: **zero
actual comparison runs have completed in this work**.

## Recovery, state and reporting contract

The replayable checklist covers exact train/evaluation commitments, source
rights/groups/masks, complete ordered native axes, initialization/readout/
attention/decay declarations, parameter aliases, numerical coordinate witnesses,
model/optimizer/scheduler/RNG/reader state, fresh-process equivalence, atomic
publication, corrupted-alias refusal, finite-loss/OOM dispositions, comparable
observed counts, per-width CPU/GPU/disk resources and exclusive-worker release.

Run IDs, trial/arm/attempt/segment identities, logs, checkpoint hashes, all metric
history and failures remain immutable. Planned, launched, terminal, measured and
selected are distinct states; a terminal receipt alone is not a comparison or a
selection. Selection alone is never a launch approval. Large weight recovery
has been exercised, but large optimizer recovery and GPU resource qualification
remain open.

## Receipts and next gate

`outputs/alternate-20261108/campaign-audit-v1/campaign-audit.json` records exact
current plan and implementation digests, grid, baseline proposal, selection rule,
record formats and open bindings. `blocker-ownership.json` classifies all 57
original extension preflight blockers: 39 source-owned, 13 construction/model
admissions and five remaining coordination/model/integration gates. This is
ownership reconciliation, not a claim that those old blockers constitute the
entire current expanded source universe.

Next: freeze the actual common train/validation snapshot and complete model,
qualify the separate SP implementation and every active native operator, finish
large optimizer/recovery/GPU/exclusive-worker evidence, and obtain distinct
matching owner attestations before the coordinator releases any campaign.
