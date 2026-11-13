# Original SciPlex study-group exposure: complete train source audit

2026-11-11. Isolated 24+2 Mora-3 campaign continuation. This is an original-source
study-group **qualification finding**, not a training release, validation outcome,
causal pairing, or a substitute for global split qualification.

## Exact authority and operations

The frozen common-source witness was
`79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`.
The native family inventory had 4 `sciplex_native_admitted` and 158
`sciplex_native_expanded` **train** source shards. Every original shard was
SHA256-streamed and compared to its pinned source hash, with an exact full9
filesystem identity; original effective-group sidecars were byte authenticated
against their independent pins and decoded. Original source task *values*,
heldout targets and model/optimizer state were not decoded or modified.

All 162 original source shards and group sidecars conserved 41,286 train tasks:
1,024 in admitted, 40,262 in expanded. All were retained by the existing
view mask. Original task IDs, source-file binding, evidence-scope declaration,
source group IDs and effective-group overlays were checked at each ordinal;
the source groups are subsets of the original effective groups.

## Scientific result and exclusion boundary

The 41,286 retained directional task records all connect into **one
study-group component**, via **one shared source-reported study token**. Both
source populations contribute to that component (1,024 / 40,262), and no other
component exists under the explicitly scoped study-token equivalence policy.

This is a strong **group-exposure warning**, *not* evidence that 41,286 task
records are independent cells, independently measured interventions, matched
molecular responses, or replications. A study/GSM identifier represents a
sampling/batch scope; neither an exact cell identity nor causal equivalence
follows. Chemical and other non-study tokens cannot create same-cell or response
pairings in this audit. Dataset admission must separately establish biological
and experimental grouping, donor/replicate/perturbation roles and valid global
evaluation partitions. A naïve row-wise random split of these records is **not
qualified as independent** by this source evidence.

There is no original validation or test population for these SciPlex
descriptors in the scoped complete native inventory. Study-level
heldout independence therefore remains **unproved**, rather than reported as
a zero-overlap finding. The audit intentionally does not manufacture new splits.

## Reproducibility, verification and next release boundary

Executable code: `src/mora3/research/sciplex_original_group_exposure.py`.
Pure plus actual-original tests:
`tests/research/test_native_sciplex_group_exposure.py`. The original-source
readback is `outputs/common-source-sciplex-group-20261111/actual-original-sciplex-group-exposure.json`,
with independent source-resource and integration-test logs in the same directory.
The regression catches missing/placeholder study groups, duplicate IDs/tokens,
attempts to equate a shared chemical token with same-cell identity and
overstating heldout readiness. Run using the existing tokengt Python,
one CPU/BLAS thread and CUDA hidden; no training budget consumed.

Required next: construction-owned global donor/study/perturbation and heldout
reconciliation across eligible native sources, correct model adapter and
source/owner/snapshot bindings, exact 29-head model/operator/optimizer and
checkpoint/RNG/cursor qualification, independent same-snapshot construction
and model attestations, and then reviewed counted-run admission. The exact
24 native μP proxy slots and at most two native approximately80M target runs
remain the only permitted substantive training. Zero optimizer updates,
counted proxy slots or target runs were executed in this task. The separate
read-only readiness watch and active checkout were not changed.
