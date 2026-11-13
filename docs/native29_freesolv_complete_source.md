# Complete FreeSolv native train replay and declared-group census

Date: 2026-11-11. This bounded source-qualification operation extends the earlier
two-row FreeSolv check to the **complete retained 556-row native training route**.
It does not run a model, reserve a campaign slot, select hyperparameters, measure
a validation baseline, issue source admission or modify the frozen campaign.

## Actual source readback

The implementation extends `src/mora3/research/common_source_freesolv.py` rather
than introducing another runner. Both existing native source adapters are bound
to their exact retained candidate and common N29 snapshot identities:

- Candidate: `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`.
- Common: `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`.

Both original iterators exhaust all three training shards, containing 256, 256
and 44 rows. Every original source coordinate, complete Arrow record, native
payload, condition, target, audit metadata and cursor position matches across
the two bindings. Independent restarts at both shard boundaries succeed under
both snapshots. The final native cursor position is `(3, 0)`. The frozen
**two-window per-update scan limit remains unchanged**; the 556-row operation
is a finite source audit, not an enlarged training run.

The production scalar projection is independently checked against a standard-
library parser of the same immutable native JSON. All 556 experimental target
Float64 bit patterns, native numeric tokens, uncertainty bit patterns and
uncertainty evidence classes agree. Targets remain reported experimental `expt`
values in kcal/mol. Source uncertainty may be a default; it is not relabeled as
a measured standard deviation or substituted for learned predictive variance.
SMILES conditions do not gain experimental targets or computed predictions.

The input is the retained source-native corpus shard set, including embedded
original-source audit rows. This does **not** re-download or independently
reconstruct the upstream FreeSolv release or establish new licensing evidence.
"Independent reference" means a separate numerical parser/check, not an
independent construction/model owner attestation.

## Complete local source-role and group accounting

All 642 original record identities and group sidecars are checked: 556 training,
42 validation and 44 test rows. The 86 held-out rows are decoded only through
seven identity/provenance columns, never their numerical target payloads.
Whole-file digests are checked as opaque bytes before and after the operation.
The tests intercept held-out Parquet projections and reject any native held-out
batch validation, while forwarding all permitted calls to the real readers.

The declared training molecule IDs contain **553 distinct molecule groups for
556 retained rows**. The three extra same-molecule rows remain present; no
silent deduplication, weighting change or assertion of independent experiments
is made. Validation has 42 distinct molecule groups and test has 44, for 639
across all roles. The co-occurring component/molecule/scaffold groups yield
379 transitive components, none spanning the original source roles.

This establishes separation only within these **existing declared FreeSolv
sidecars**. It neither computes new chemical similarity nor establishes global
cross-source chemical-group closure, scaffold-independent generalization or a
clean accepted benchmark. Cross-source exposure and source eligibility remain
separate qualifications. The original roles and permission flags are unchanged.

## Evidence and verification

The complete 556-row readback is:

`outputs/common-source-freesolv-complete-20261111/original-complete-freesolv-readback.json`

SHA256: `f0820098583bd37155b55f36716b8734649613faa1d897250c2d89759d7e51a9`.

The first two rows also reproduce the retained earlier receipt's source,
condition, numeric and cursor commitments. That receipt has SHA256
`bf1f6b8ef05188c89396e1219b83daa82b631969d989ea24801a594931c7d75c`.
No historical receipt or source file was rewritten.

A missing-interface failure was retained before implementation. The initial
focused run passed 18 tests. The final focused integration run passed **33 tests,
no skips**, including those 18 tests, the earlier seven FreeSolv route tests and
eight existing scalar-semantics tests; these are overlapping, not 51 unique tests.
The tests include transitive cross-role bridge detection, repeated-molecule row
retention, duplicate identity refusal, malformed group rejection, exact numeric
lexemes, nonfinite/bool rejection, condition-target separation and source-backed
full-route exhaustion. This was not the full project test suite.

The separately saved actual-source readback took 16.505 seconds and measured
3,318,044 KiB peak process RSS. The 33-test run took 24.742 seconds outer wall
time and measured 3,389,400 KiB peak child RSS. Both used one CPU thread, CUDA
hidden, no optimizer updates, no checkpoint writes and no training processes.
Detailed commands and results are retained in the same output namespace.

## Remaining campaign boundary

This closes one complete original native train route and its source-local role
census, not the full expanded corpus, all 62 routes, current-owner validation,
operator/resource/recovery qualification or distinct same-snapshot owner review.
The approximately 80M target and real 24-trial selection are not completed by
this audit. The user cap remains 24 proxy slots and at most two target runs.
The changes stay in the isolated worktree; active model code, other owners'
ledgers, worker/supervisor configuration and the read-only watch are unchanged.

Next work should use this complete source inventory when checking existing
cross-source chemical exposure and immutable quarantine masks, or advance the
next complete native source route. Do not repeat the two-row demonstration as
new progress, relabel held-out targets or promote this local group census to
full training admission.
