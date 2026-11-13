# Four-source native chemical group exposure: original sidecars, no heldout targets

2026-11-11. One bounded real-source qualification in the isolated
`implementation/campaign26-20261110` worktree. **No training admission, optimizer
updates, campaign runs, hyperparameter selection or target transfer.**

The scoped source registry contains **197 pinned original group sidecars**
spanning ChEMBL37-AZ logD7.4, FreeSolv, Tox21 original source-call prior, and
ExpansionRx native ADMET readouts. Every sidecar was read and checked against
its original immutable SHA256 commitment and registered row count. The original
source shard paths were checked by role, existence and byte length only; **no
numerical source shard, heldout label/target, or heldout condition was opened**.
The source factory's candidate and common 29-head snapshot metadata were
checked against `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`
and `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`,
respectively. These are metadata identities, not empirical model qualification.

| Source and role | Original effective-group rows |
|---|---:|
| AZ train | 4,184 |
| FreeSolv train / validation / test | 556 / 42 / 44 |
| Tox21 train / validation / test | 12,892 / 228 / 480 |
| ExpansionRx train / test | 38,442 / 22,984 |

Total: **79,852 original task-sidecar rows** in nine source/role populations.
The existing registry recorded no ordinal exclusions in this scoped set.
All 20 heldout-versus-train pairings have zero direct original molecule-hash and
scaffold-hash intersections. This is a limited **literal native hash** comparison,
not a negative assertion about all stereochemical/tautomeric/salt identities,
source-wide structural exposure or biological independence.

A necessary positive control is that training-source hashes do intersect:
AZ–FreeSolv has **11 molecule and 16 scaffold hashes**;
AZ–Tox21 **201 and 293**; FreeSolv–Tox21 **161 and 160**.
AZ–ExpansionRx has **zero molecule and two scaffold** hashes;
Tox21–ExpansionRx **zero molecule and one scaffold** hash;
FreeSolv–ExpansionRx has zero of either. Source-local hashing is not an
independent universal chemical equivalence adjudication. Prior independent
AZ/FreeSolv RDKit work found ten exact isomeric train/train structures, which
must not be substituted for these differing native group counts.

Of the ten comparisons among heldout populations, **two have intersections**:
FreeSolv validation and Tox21 validation share one molecule hash and one
scaffold hash; FreeSolv test and Tox21 test share three scaffold hashes and
zero molecule hashes. These are cross-benchmark exposure warnings; they are not
train/test leakage discoveries, proof of same assay or an independent heldout
qualification. Every comparison's counts, task multiplicity and collision
commitment are in the readback, without decoding heldout numerical targets.

Evidence:
`outputs/common-source-four-chemical-role-exposure-20261111/actual-original-four-family-group-exposure-v2.json`.
The earlier v1 source readback was retained unchanged; v2 adds six
train/train source-domain controls. The original-source replay finished in
9.17 s with 3,320,608 KiB peak RSS. The nine relevant focused pure regression
tests passed (0.82 s pytest; 1.40 s external elapsed). Two combined test
attempts were deliberately preserved as failures: an initial isolation import
failure and, after adding test paths, active-root-only manifest assumptions in
older unrelated cases. None was repaired by modifying another owner's source
or bypassing a gate. No CUDA allocation or optimizer step was performed.

Limits and next task: the sidecars already carry established source-native
molecule/scaffold hashes, not independently rederived standardized structures
from inaccessible heldout originals. Registered source bytes were stat-only,
not independently rehashed in this audit. This is a scoped four-source exposure
diagnostic, not complete chemical, protein, gene, assay/study, trajectory or
other expanded-corpus grouping. The construction owner must reconcile these
and other source groups with an independently adjudicated, globally
leakage-safe split; the exact final model owner still needs its same-owner
reference, operator, recovery and resource evidence and **distinct** owner
attestations. Keep the 24 native μP proxy slots and at most two native ~80M
target runs untouched until genuine admission.
