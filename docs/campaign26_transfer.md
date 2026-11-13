# Counted 24-trial sweep and native hyperparameter-transfer continuation

Date: 2026-11-10. This package repairs and completes the **post-sweep software
handoff**. It does not contain results from an actual24-run campaign. No source
admission, new model architecture, optimizer update or substantive run is issued
by these changes. The maximum remains24 proxy trials plus two ordinary full-model
approximately80M runs retaining the native muP parameterization.

## Reproduced defects and corrections

The old target-spec helper hashed the inner selection result, while the native
publisher and persistent run budget bound the outer publication (including the
terminal pin-chain commitment). A complete synthetic24-result replay reproduced
the disagreement. Target specifications now carry THREE explicitly different
identities: selection-publication SHA256, inner-result SHA256 and terminal-prefix
SHA256. They verify the exact canonical published wrapper against a fresh replay
of the native validation-only selector rather than treating these hashes as
interchangeable.

The active-directive reader previously checked24+2 counts and target seeds but
not the primary grid. A changed master learning-rate candidate0.1 instead of0.001
was incorrectly accepted with a legitimately rehashed directive pointer. The
reader now requires the exact native3x2x2 grid and both proxy seeds.

The target specification must be frozen before collecting proxy results. The
versioned `Campaign26/v2` genesis includes a complete target-blueprint hash. Old
v1 wire representations remain unchanged and readable, but cannot be described
as predeclared target settings. The existing counted proxy worker requires the
blueprint BEFORE reserving a slot and checks its width-family/task-cycle/clipping/
decay policy against the frozen native proxy campaign. Merely storing a new
configuration file does not allocate or start a run.

## Native data path

`campaign26_transfer.py` provides:

1. `TargetBlueprint`: native target factory configuration template, all-head
   parameter count, source/family/owner/resource commitments, update/time budget,
   task cycle, clipping and decay-alias policy. Readout and attention slots stay
   neutral until the selector supplies their values. The target horizon must
   satisfy the existing native schedule's exact10-percent warmup policy.
2. `verify_counted_selection`: rereads all24 native terminal receipts and metric
   histories through the existing immutable pin store; cross-checks counted IDs,
   statuses and artifact hashes; reruns the original paired-seed validation-only
   selector; and verifies the published winner. Target outcomes cannot influence
   that winner. Partial histories, mismatched artifacts, modified prefixes and
   absent paired-seed candidates refuse.
3. `publish_target_transfer` / `load_target_transfer`: exclusive publication and
   complete readback of both target configurations. Publication holds the existing
   run-budget lock, preventing a concurrent target reservation at that boundary.
   These functions do not reserve a run, construct a model, take a step, or issue
   a launch authorization. Readback remains possible after a target is started;
   the publication's `target_training_started=false` describes its nonexecuting
   publication operation, not a live progress monitor.
4. `transferred_configuration`: exact native `PrimaryConfiguration` and `MuAdamW`
   inputs. Every non-tuned target setting is checked against the predeclared
   blueprint, including count, source, seed, updates, schedule and no-target-tuning
   disposition. A caller must first verify the pinned publication and histories;
   a self-consistent dictionary alone is not external source authority.
5. `require_transfer_target`: checks an already-admitted real native target update
   engine against the transferred settings. It calls the ORIGINAL owner admission
   method first, checks the actual count/aliases through native inventory, native
   family membership, optimizer settings, schedule, task cycle and fresh state.
   It creates no optimizer, checkpoint, lease or training process.

The target factory's stored owner representation differs from its constructor
input: source identity is outside `primary_configuration`, and the constructor
adds heads, layers and vocabulary identity. A dedicated comparison now uses the
actual native representation, tested on a real small factory assembly. It does
not mistake a dictionary-layout difference for a model mismatch or weaken the
owner's training refusal.

## muTransfer semantics

The selected master LR is not manually divided by model width. Existing MuAdamW
applies the role-dependent width multiplier exactly once. The generic real
optimizer fixture checks a hidden4->8 last-width-axis expansion: its group uses
masterLR/2 while finite/input/readout groups retain the master LR. Tied aliases
remain one parameter owner. The configured weight-decay convention is unchanged.
There are no optimizer updates in this check and no copied proxy weights or
optimizer moments. This is configuration/parameterization correctness, not an
empirical claim that the final biological target has improved or that all of its
operators are already muP-qualified.

Primary method reference: Yang et al., Tensor Programs V: Tuning Large Neural
Networks via Zero-Shot Hyperparameter Transfer, arXiv:2203.03466v2. Existing native
optimizer metadata retains the project's pinned upstream/fork revision; no
packages, weights or dependencies were downloaded or changed.

## Verification

`outputs/campaign26-transfer-20261110/transfer-regression-v2.json` records
**113 passing focused tests**, no skips,20.18 seconds pytest time,21.25 seconds
outer wall time, one CPU thread and CUDA hidden. This is not the entire project
suite. The tests cover the two reproduced old defects, all24 synthetic histories,
paired-seed selection, immutable receipt replay, corruption refusal, target
preregistration, lock exclusion, a fresh-process readback, real MuAdamW group
scaling and the actual current native owner's refusal before execution effects.
Earlier failing/red receipts and the first error-message-case mismatch are kept.

All trial metrics/histories used in these tests are **synthetic test fixtures in
temporary test directories**. The fixture parameter count80,304,407 and target
budgets20,000updates/86,400seconds are NOT a current project target declaration.
The corrected code does not convert these fixtures into source admission or
actual campaign success. This continuation performed zero optimizer updates and
started zero actual sweep or target runs.

## Remaining live execution dependencies

The active source/model ledgers still report full_dataset_ready=false, missing
distinct same-snapshot construction/model attestations and unfinished expanded
source/operator qualification. `SourceBoundOwner.require_training_admission`
continues to refuse. This package does not edit that owner, the source ledgers,
master corpus, schedules of existing workers, global configuration or environments.

A later staged29-head constructor has retained counts10,145,408 /15,944,576 /
80,283,064 at widths24/36/130. These are a separate scoped source-family witness,
not the earlier optional-head-omitting fixture and not a qualification of the
complete SCRI/trajectory-extended training campaign. The common-source rebind
receipt explicitly remains metadata-only: storage-isolated source replay, a new
matched baseline for that same owner, independent split evidence and executable
release remain open. Old reference metrics from another owner family cannot be
reused merely by changing their snapshot hash.

The actual target blueprint is therefore still unresolved: do not fill it with
the unit-test dimensions or budgets. The winning HP values are also intentionally
unknown until the REAL24 outcomes exist. Unknown winning values do not block an
otherwise qualified proxy phase; missing real dataset/operator/execution evidence
and missing fixed target specifications do.

The existing supervisor was inspected but not restarted: its historical prompt
instructs reading the private original specification, and its completion test is
training-start rather than24-trial/transfer completion. No new agent/supervisor or
training loop was launched in this package. Any continuation must preserve the
latest user restrictions and existing ownership rather than blindly replaying
that old prompt or the superseded44-run plan.
