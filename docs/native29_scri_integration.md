# Native 29-head source-factory and SCRI integration — bounded implementation handoff

2026-11-10. Isolated checkout: `/home/iska/Documents/amelie/bio/Mora-3-campaign26-20261110`. This package integrates the previously retained N29 source-factory candidate into the existing SCRI/trajectory-aware implementation without deleting or rewriting the existing SCRI branches. It is **implementation integration**, not full source qualification, complete29+SCRI owner count, a training release, a campaign run or evidence of achieved MuTransfer.

## Reviewed inputs and direct results

Retained original candidate:
`/home/iska/Documents/amelie/bio/Mora-3/outputs/primary-campaign-n29-source-factory-20261108-v1/production-candidate.patch` (SHA256 `796422313e337d327576238960a777460f74a47ac169e3e017d074b5c2bf2e8e`). Direct `git apply --check` on the original candidate failed at `src/mora3/model/full_owner.py` because SCRI had already added another optional owner/source branch. Applying without resolving this conflict would silently omit a required native capability.

The other21 staged targets applied in the isolated worktree without conflict. The `full_owner.py` hunk was manually reconciled with SCRI preserved. The result supports optional BETSE membrane heads and raw scCello count heads with exact shared native annotation owner/source identities **and** retains optional SCRI. It serializes each optional branch in strict owner/checkpoint topology and retains exact identity-tied parameter alias handling. `full_owner_mup.py` includes exact parameter-role dispatch for BETSE and raw scCello. Source asset factory, native assay and extra-axis bindings, source replay helper and diagnostic launch-preflight code were carried over from the pre-reviewed staged candidate.

The staged owner admission change is fail-closed: `full_owner_training_admission._REVIEWED_RELEASES` remains empty. Its previously inconsistent comparison of `PrimaryConfiguration` inputs with stored owner state was corrected to use the already-tested native owner representation matcher. In particular, source snapshot identity is top-level in the owner, and heads/layers/vocabulary digests are constructor-added metadata. This correction does **not** mint any reviewed release. Existing SCRI+N29 owner combinations do not match an N29-only release inventory and are conservatively refused pending a separately qualified complete expanded owner.

## Tests and bounded source preflight

`outputs/native29-scri-integration-20261110/initial-integration-tests.json`:
108 focused tests passed. Coverage includes existing native full owner, shared trajectory policy, muP campaign worker, 26-run counted worker and target-transfer package; single-thread CPU; no substantive training.

`outputs/native29-scri-integration-20261110/native29-scri-tests.json`:
11 additional checks passed against SCRI, new N29/SCRI coexistence and original source-assembly tests. Checks bind the real small native SCRI owner, shared parameter inventory and strict checkpoint recovery, refuse invalid BETSE/scCello source bindings before RNG mutation, and verify absent release authority. The checked SCRI module cannot be silently erased to make an N29-only release pass.

`outputs/native29-scri-integration-20261110/source-preflight.json`:
The newly integrated N29 preflight returned exit status0 against the actual project metadata without model construction or optimizer updates. It authenticated the historical widths24/36/130, known scoped counts10,145,408/15,944,576/80,283,064, 51,282 raw fit gene keys and documented source identity witnesses. It explicitly returned `owner_and_routes_rebound=false` and `training_authorized=false`. These historical **scoped** counts do not include any newly attached SCRI owner, so they are **not** the full required expanded campaign counts.

## Remaining actual critical path

The owner snapshot `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4` does not match the original route snapshot `c47f7a2b55da718c64f9910167830e5667cf4806249889d57eae336747e34ec2` or the inactive successor `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`. A metadata-only common rebind already exists, but actual executable source replay, fresh same-owner matched validation reference, global heldout/group and all required source-eligibility verification remain uncompleted. Full native29+SCRI+trajectory parameter counts, optimizer/scaling/recovery and hardware qualification are also unverified. No run budget or training release is created here.

The full dataset remains unready, both distinct current construction/model attestations are absent, and actual training admission remains unissued. The current run cap continues to be24 genuine proxy slots followed by up to2 native ~80M training slots. There were zero new optimizer updates, zero actual proxy results, zero transferred winner values and zero target training runs in this integration task.

Preserve the active parent checkout, protected data and existing owner ledgers until this isolated package receives a fresh conflict-checked safe-boundary publication. No external pushes, downloads, installs, jobs, checkpoints or supervisor restarts were performed.
