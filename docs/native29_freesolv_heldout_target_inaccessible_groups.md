# Original FreeSolv heldout group exposure, target-inaccessible source audit

Date: 2026-11-11. This is **one bounded source qualification**, not validation
selection, complete chemical identity proof, 62-route corpus admission, or any
training run. Only the *original SHA256-authenticated group sidecars* were
read, together with the two previously authenticated complete **training**
source readbacks (AZ 4,184; FreeSolv 556). The candidate and common 29-head
source snapshot identifiers remain unchanged.

## What was actually inspected

The original FreeSolv **42 validation + 44 test** records have one molecule and
one scaffold hash in their original effective grouping sidecars. We compared
these to the original ChEMBL37-AZ 4,184 training and FreeSolv 556 training
sidecar records, and compared validation directly with test. In all **five**
scoped comparisons, **zero identical `molecule:` and zero identical `scaffold:`
hashes** occurred. All 22 sidecars (17 AZ train, 3 FreeSolv train, one FreeSolv
validation, one FreeSolv test) were re-read using their original exact-byte
SHA256 commitments. Each row retains its original task ID and role. FreeSolv
sidecar source-file pointers were checked, and the original shard *sizes* were
checked without opening the source shards. The source file SHA values in the
audit are explicitly *prior-receipt references*, not newly recomputed hashes.

The curated chemical grouping recipe hashes stereo-insensitive canonical
connectivity for `molecule:` and Murcko scaffold (or fallback molecular string)
for `scaffold:`. These are **guard identities**, not experimental observations
or full chemical identities: they do not resolve tautomer, charge-state, salt,
structure-standardization or assay equivalence. Equal scaffold hashes would be
a conservative leakage warning, not proof of identical chemistry; the absence
of a hash collision is scoped to these exact curated source representations and
policies. The AZ source has no original heldout role in the audited route.

The original source-shard target columns were **not opened**. In particular,
`native_payload`, measured labels, heldout coordinates, original target bytes
and source response distributions were not decoded or used in this check. The
only projected columns for FreeSolv heldout were `task_id`, `source_file`,
`source_group_ids_json` and `effective_groups_json`. AZ sidecars contain only
`task_id` and `effective_groups_json`. These separate columns and the sidecar
files are not a substitute for an independently justified global chemical /
biological leakage policy. Previously established training-side original
chemical overlap (10 exact isomeric AZ–FreeSolv training structures) remains
untouched; this audit introduces no heldout target exposure.

## Limits and next authorization boundary

A zero collision for these group keys does not establish global train/validation
independence, exact isomeric heldout structures, correct stereochemistry,
complete all-source cross-family grouping or permission to use FreeSolv test in
selection. The complete eligible expanded corpus, independent distinct
construction/model owner attestations, a matched validation baseline for the
80,283,064-parameter STAGED29-head owner (plus SCRI/trajectory qualification),
resources, restart/cursor correctness and source-bound executable release still
require independent verification. `SourceBoundOwner.require_training_admission`
must not be bypassed. If exact stereochemical heldout structure linkage is
needed, acquire/prove a **separate original condition-only structural identity
view** with no target access, rather than decoding the heldout native payload.

The implementation is `src/mora3/research/heldout_chemical_group_exposure.py`
and its tests are in `tests/research/test_heldout_chemical_group_exposure.py`.
The original auditable readback is under
`outputs/common-source-freesolv-heldout-group-20261111/`. No campaign budget,
optimizer, checkpoint or training job is touched; the at-most-26 cap remains
24 native μP proxy trials plus at most two full-model μP target runs. The
independent READ-ONLY readiness watch is unchanged.
