# First original native train-window common-source replay — 10 November 2026

Scope: one bounded, actual original-source row, not a synthetic receipt or
metadata-only route rebinding. This is an additive continuation of the
29-head/BETSE/raw-cell source-factory integration at implementation commit
`9bd561a9a8108cbae3c1349a17b9eac7c202659a`. The active source data were
read from `/home/iska/Documents/amelie/bio/Mora-3`; the new code was developed
in the existing isolated `Mora-3-campaign26-20261110` worktree.

## Actual source and executable route identities

Imported the existing previously reviewed inactive common-source metadata
checker, byte-identical to the retained
`outputs/primary-campaign-common-source-rebind-20261108-v1/candidate/src/mora3/research/primary_campaign_source_rebind.py`
(SHA256 `2c9034fcadf13fa140313fcd0b0b9e13b1e9e31839977ed2e149d5f33b02e3b4`).
The preserved active rebind manifest is
`outputs/primary-campaign-common-source-rebind-20261108-v1/common-source-rebind.json`
(SHA256 `9da9041ab90959d15a7395a21c3cd90d8a1b55648898b300559b37feb99d6c0b`).

The new `src/mora3/research/common_source_native_replay.py` reads the **actual**
`effective-study-v2` native view through the preexisting real source constructor,
recreates two native `NativePriorCampaignAdapter` instances for `native-prior:dna`,
and binds both against the immutable source/route metadata. Their different source
snapshots are `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`
and `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`.
Their exact adapter and initial-cursor hashes agree with the reviewed candidate
and common descriptors, despite being different between the two snapshots.

The bounded diagnostic then opens just the **first eligible original train
window** with each adapter. It verifies full Arrow-native row identity, source
shard, original row ordinal, task identity, native-payload SHA256, source-role
`train`, and exact independently derived source-row digests. It calls the
original native adapter `prepare` to count observed training targets but never
calls its returned model-evaluation closure. No model is allocated and no
optimizer, scheduler, trainer, native run budget, held-out evaluator or generator
is executed.

**Observed actual source outcome:**

- Original source ordinal: **0**, one authentic native training row.
- Source shard SHA256: `670d533aab75ad28305cd25e5e9cb91a8d8bc63c5940c5385f79e609875ca79f`.
- Native payload: **5,433 source bytes**, SHA256 `459e565690a4796ea4a98ac9cd09957e4111203876d3654f8fdcbd2c260`.
- Exact native prepared supervision: **3,980 observed train tokens**.
- Independently hashed original and common source row digests both
  `b9c9e44482d352fc414340204574e63bf4eb65a75dfa59d5e7472e90910df957`.
- Source grouping/view policy SHA256 `c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.
- **Zero optimizer updates; zero proxy/target training runs.** No source writes.

These are real native source bytes and target positions, but they do not prove
source-level scientific eligibility, independent train/validation leakage closure,
all62 native routes, or an80M target owner/transfer. They do not qualify any
of the missing owner signatures. No biological, mechanistic or clinical quality
claim follows from the byte-level correspondence.

## Verification and controls

Initial missing-module test fails as expected:
`outputs/common-source-native-replay-20261110/expected-red.json`.
Real-source five-test package passed on the original-root checkout:
`outputs/common-source-native-replay-20261110/second-real-test.json`.
The independent checksum implementation and byte-alteration/role-refusal tests
were added afterward. A separate observed readback with independent native
checksums passed: `outputs/common-source-native-replay-20261110/independent-double-readback.json`.

The separate isolated-worktree fixture regression uses that worktree as cwd,
which the existing native optimizer code requires to verify executable adapter
file paths; it yielded **91 passed, 53 skipped, one deliberately deselected**.
The skips were due to the unmaterialized qualified LINCS component under that
isolated root and must not be called successful full-owner qualification.
The earlier mixed-root command produced143 passes and two path-context failures
because it imported modules from the isolated worktree but ran source-registry
fixtures using the different active checkout as cwd. This is a harness mismatch,
not a mutation of the native reader or an accepted skipped result. The separate
current commands use the correct root for their specific scope.

Exact Python interpreter: `/home/iska/miniconda3/envs/tokengt/bin/python`.
No package installation, environment changes, GPU work, worker restart, network,
source mutation, checkpoint overwrite or extra training run.

## Next single bounded critical-path task

Extend this validated real source read to the remaining original train windows
of this *same* route and independently prove complete cursor/order/target
conservation, preserving exact fresh-process state. Then advance to the other
61 source route identities and the five actual matched validation evaluator
scopes under the SAME `79d328f5...` source owner. The complete replay and a new
owner-specific measured validation baseline must be established before the
24-trial run-budget genesis, fixed target blueprint, capacity and data/model
owner attestations can become an executable campaign release. Do **not** reuse
historical reference losses from the old family or treat the metadata-only
`common-source-rebind.json` as an admission receipt.
