# Two actual native DNA prior windows: common-source and cursor conservation

Date: 2026-11-10. This is one bounded source-fidelity implementation/verification step after `6cebcdd03c0efe68789e3f08e0b8f16a9f91ff1e`. The active original source is `/home/iska/Documents/amelie/bio/Mora-3`, and the implementation remains in the isolated `Mora-3-campaign26-20261110` worktree. No training run, optimizer update, source rewrite, credential access, model construction, checkpoint write, worker restart or environment modification occurred.

## Real original data, not fixtures

`audit_common_source_two_prior_windows` reconstructs the original and common-snapshot `native-prior:dna` adapters using the actual immutable source view and native fitted codec. The original candidate snapshot is `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`; the common 29-head owner snapshot is `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`. The exact adapter and initial-cursor hashes are checked against the retained inactive source rebind. Both use `effective-study-v2` and source-view digest `c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.

The function consumes *exactly two* original one-row train windows from each independently bound route. This matches the reviewed per-update maximum window scan budget of two; **it does not assert that the original DNA corpus contains only two rows** or that the total 5,000-update campaign has been replayed. Raw Arrow rows, types, parent shards, original row ordinals, task roles, payload digests, independently computed source fingerprints, full native target-byte counts, token counts, bounded target-preparation dispositions and abstract route cursor positions are compared across both snapshots. A second independent iterator is started from the recorded first-window cursor under each snapshot, and its next row, cursor and target disposition must exactly match the sequential second window. Source coordinate cursor positions agree even though serialized cursor hashes necessarily differ by route identity. No target is decoded by a model; no returned neural evaluation closure is executed.

## Actual result: retained second row is over the frozen target-byte bound

The original DNA source shard SHA256 is `670d533aab75ad28305cd25e5e9cb91a8d8bc63c5940c5385f79e609875ca79f`.

| Original row ordinal | Native payload bytes | Original target bytes | Native target tokens | Preparation under the frozen 4,096-byte cap | Cursor after |
|---|---:|---:|---:|---|---|
| 0 | 5,433 | 3,979 | 3,980 | Observed/admissible to the source-only adapter | shard0, ordinal1 |
| 1 | 5,681 | 4,227 | 4,228 | **Budget-refused**, not truncated and not admitted to training | shard0, ordinal2 |

Both source snapshots produced the same values, roles and dispositions. The original source contains **8,208** native target tokens across the two inspected rows, of which **3,980** are within the current route's fixed preparation-byte budget. The remaining 4,228 are retained original source values, *not* accepted training labels under this configuration. The first row was already checked in the previous iteration; the second is a new real-data finding. The refusal is the correct preexisting native adapter behavior, not a signal to silently enlarge the budget, change the frozen descriptor, truncate the target or mark the row missing. One planned `native-prior:dna` update is present in the inherited 5,000-step schedule; this diagnostic does not certify the complete schedule nor how other routes behave.

The source-row identities differ, and the routes' cursor positions advance strictly `(0,0) -> (0,1) -> (0,2)`. A fresh cursor restart independently reproduces the second original source row and its over-budget disposition under both snapshot bindings. This proves only source-read/cursor reproducibility, **not** full optimizer, CUDA RNG, owner checkpoint or task-scheduler recovery.

## Tests and retained evidence

Tests were written before the implementation: `outputs/common-source-second-window-20261110/expected-red.json` records the missing API. The first real source execution discovered the over-budget second row; `real-native-two-window-v1.json` preserves that deliberate failure. The test contract and code were corrected to preserve an explicit `budget_refused` disposition instead of changing the source or finite budget. `real-native-two-window-v2.json` then passed **8 checks** on the original source root.

`actual-two-window-readback.json` records a separate successful original-source readback, including both original row SHA256s, cursor hashes and dispositions. Peak process resident set was about 2.85 GiB, with one CPU thread and no CUDA visibility. The final scoped regression executes tests from their correct code/source working directories: `real_source_original_cwd.json` records **14 passed** (includes real-source checks), and `native_fixtures_worktree_cwd.json` records **13 passed, one skipped** (the qualified LINCS component was not materialized in that fixture worktree). `combined-focused-regression.json` retains two earlier failures caused by mixing code from the isolated worktree with the original root in executable-adapter hashing; splitting the working directories resolves both without changing production code. These counts overlap and are not summed into one unique test total.

The source-read tests and independent readback are not training. They establish only two-window source fidelity, configured target refusal and cursor continuity. No complete 62-route replay, five owner-specific matched validation baselines, eligible source/master closure, global split qualification, resource/optimizer recovery or genuine construction/model owner attestations are established. `SourceBoundOwner.require_training_admission()` remains unchanged and closed. No actual24 proxy outcomes exist and no transfer winner is selected.

## Next bounded source-critical-path task

Use the same immutable common owner/source and frozen route metadata to qualify the next source-native route (for example the next `native-prior` family) with its exact planned per-update window bound, source role, target capacity and ordered cursor. Extend only where actual tested evidence allows. Keep over-budget original values and explicit refusals visible; do not silently skip them in the campaign's source cursors. After meaningful route-by-route progress, the complete 62-route/5-validation-scope same-owner replay, independently accepted groups/splits and genuine distinct owner admissions remain mandatory before initiating any of the24 counted μP trials. The user run limit remains24 proxies plus at most two approximately80M targets, with no extra retries.
