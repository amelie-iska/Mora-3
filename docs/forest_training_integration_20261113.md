# Native complete-forest training integration

13 November 2026. This implementation connects all four existing native forest heads to an explicit source-bound objective and the existing native update engine. It adds no trainable parameters and does not issue dataset or full-model training admission.

## Executable path

`mora3.model.forest_training` contains the typed source identity, separately typed flow/token/quality/analogy supervision, complete preflight, fixed objective specification and weighted joint loss. `mora3.research.forest_training_routes` binds a finite ordered segment of those batches to an exact trial, source snapshot, native head configuration and repeat policy. Its production entry checks the actual `SourceBoundOwner` admission before reading a source window, then uses the engine-owned generator and calls the existing `NativeMuUpdateEngine.update` only after successful objective preparation.

The four heads must reference the same literal field object. The actual field still shares the native geometry/TokenGT owner. Every head's parameter/buffer dtype and device must agree, and the source codebook, criterion policy, source encoder, coordinate system and grouping policy are pinned. The source coordinate/encoder commitments are declarations requiring independently qualified source exports; merely constructing a Python object is not scientific qualification.

## Distinct supervision roles

* Flow supervision supplies a detached clean latent target and an explicit per-coordinate availability mask. Gaussian noise and uniform solver time come only from the checkpoint-owned generator. Known clamped coordinates are not targets.
* Token supervision supplies a separate detached source-encoder latent export, source-aligned token labels and their availability. This is decoder reconstruction supervision, not a claim that the flow generator has produced those states. A complete contiguous token prefix must end in exactly one allowed stop token. Missing tokens are not fabricated and oversized payloads are not truncated.
* Quality supervision supplies a separate complete candidate state and criterion-specific labels/masks. Criterion values are finite in [0,1] where present. Unknown labels, including masked NaNs, do not become negative examples. These predictions are not independent hard-constraint decisions or calibrated biological evidence.
* Analogy supervision supplies condition-only query evaluation, complete detached train-memory forests and explicit supported positive/negative pair labels. A group-overlapping pair is ineligible; unknown pairs do not enter the denominator. Every query requires at least one disjoint supported positive and negative. Retrieval similarity is not proof of directed semantic equivalence.

One batch uses one declared source-known skeleton and context; differently shaped or conditioned examples use separate batches. Hidden topology is not learned or licensed by this interface. The normalized unknown-topology policy and hybrid GFlowNet teacher/flow-student distribution remain separate requirements. Existing stochastic trajectory-GFN code and deterministic FoT/flow inference retain their different semantics.

## Reduction and missingness

The loss is the fixed weighted sum of available head means, in the declared order flow/token/quality/analogy. The flow mean is over available coordinates; token reconstruction is cross-entropy over available complete token slots; quality is summed binary cross-entropy divided by the number of available whole-forest and node criteria; analogy averages its explicit multi-positive contrastive loss over queries. A missing head contributes no term and does not renormalize the other coefficients. An entirely unavailable batch produces no objective, no random draw and no optimizer update.

Every enabled supervision path is checked before any neural forward or random draw. Source IDs, original ordinals, evidence class and hashes are retained as audit metadata, not added to predictor features. Changing a graph's hidden target does not alter its conditional context or decoder/quality input; flow denoising has its separately declared noisy-target role. The importer does not invent source pairs, rewards, source rights or complete cross-corpus leakage separation.

## Finite source segment and recovery

`ForestTrainingRoute` clones detached source tensors, commits the complete batch contents, rejects duplicate original identities/ordinals within the segment, and preserves an explicit ordered traversal and repeat count. A cursor binds the route and exact segment position. Payload capacity counts tensor bytes plus serialized metadata; it is not an operating-system RSS limit or a whole-corpus storage estimate. This is a bounded in-memory native segment interface, not a disk-backed full-corpus materializer. No source scripts, unsafe pickles, foreign model weights or network services are invoked.

The optional forest binding adds checkpoint schema `Mora3.NativeMuUpdateState/v6`; engines without it keep their previous v1-v5 formats. The binding includes the source, trial, task, route, objective, head configuration, method, known-skeleton policy and exact total window count. Restore rejects wrong-route, noncanonical, boolean or out-of-range positions before copying parameters or optimizer state. The engine's original complete model, optimizer, scheduler, RNG and cursor recovery remains in force.

A failure during pre-update preparation restores both the private sampler and global random states. Once the objective is handed to `engine.update`, the wrapper does not pretend that an optimizer-phase failure can be rolled back. Such failures belong to the existing counted-run terminal/recovery protocol and do not grant a replacement run. Source cursors commit only after a successful native update. Inactive-window scanning likewise does not commit a source position without an update.

## Verification boundary

The new checks compare the joint numerical loss and gradients against independent calls to each original head objective. They cover source/split/encoder/coordinate/group/policy refusal, complete token grammar, masked nonfinite labels, explicit analogy support, exact shared heads and dtype/device agreement, target exclusion, source-content sealing, original ordinals, repeat/exhaustion behavior, real owner admission before reads, sampler rollback and native checkpoint replay. Small native full-owner tests preserve actual module aliases and use the existing MuAdamW family without taking optimizer steps. Test-only admission stubs exist only inside explicitly synthetic test cases; production admission is unchanged.

The first recovery-negative tests demonstrated that invalid forest cursors could otherwise pass generic checkpoint validation. The implementation now checks their source and range before state copies. Earlier failed attempts and unavailable source-fixture paths remain in the local execution evidence; no synthetic receipt replaced an original source artifact.

These are CPU implementation checks with zero optimizer updates, not a completed-update recovery experiment, empirical width-transfer qualification, trained quality result, final approximately80M parameter count or finished campaign. Final test counts and code pins are recorded in the execution handoff. Native source preparation, final model/campaign admission, the 24+2 campaign, unknown-topology learning and post-training evaluations remain unfinished.

## Reproduction

Run the new tests with this checkout's `src` and test helper directories on `PYTHONPATH`. The existing full-owner fixtures require the unchanged original LINCS and annotation-axis receipts; point `MORA3_REPO_ROOT` at their active root and use that root as the working directory for the older relative-path annotation fixture. Keep the new source/tests isolated. No dependency changes are needed. Guard optimizer step methods during this implementation-only suite.

Primary method: Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747v2. Gaussian conditional paths are reused from the existing native field. The four-objective composition, finite source-segment protocol and native checkpoint integration are Mora-3 adaptations.
