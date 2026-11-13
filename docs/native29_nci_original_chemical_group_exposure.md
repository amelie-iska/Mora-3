# Original NCI-AIDS chemical-group exposure in the staged 29-head owner

This bounded source-qualification implementation extends the authenticated
ChEMBL37-AZ, FreeSolv, Tox21 and ExpansionRx group audit to three distinct
original NCI-AIDS task-view families: source conclusion, source endpoint and
paired assay profile. It is NOT a global evaluation split, a source/model
attestation or training admission.

## Source authority and protection

The source authority is the existing pinned common-source snapshot
`79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`,
not GitHub main or a synthetic dataset. The original four-family sidecar audit
is recomputed and authenticated first. The additional NCI metadata comes from
the same immutable finite native source view. The implementation verifies every
source file's exact full-nine stat identity WITHOUT reading its numerical
payload and verifies every original group sidecar's byte count and SHA256.

It reads only `task_id` and `effective_groups_json` from 114 original NCI
sidecars, preserving source train/validation/test codes and exact retained
ordinals. The NCI sidecars represent 115,786 retained task records:
69,632 conclusion train, 6,144 validation, 6,144 test; 29,696 endpoint
train, 2,048 validation, 2,048 test; and 74 paired-profile train.
No heldout native source, target, assay label or condition was opened.

Source-specific NSC and component identifiers remain intact in the original
sidecars but are NOT treated as independently standardized cross-family
chemical identifiers. The group closure uses only explicit, typed, literal
`molecule:SHA256` and `scaffold:SHA256` group keys, and does not infer
stereochemical equality, assay independence or causal pairing. A shared
scaffold is deliberately conservative and cannot be called identical chemistry.

## Measured cross-source group exposure

The full authenticated readback covers 195,638 original task records in
16 source-role populations (the original four families' 79,852 plus NCI's
115,786). The combined literal molecule/scaffold graph contains 28,753
connected components, of which 9,366 span distinct source task families.
There are **zero** connected components that mix training with validation
or test, and **zero** that mix validation with test. None of the 63 ordered
heldout-versus-training role pair comparisons share a literal molecule or
scaffold hash. There are 1,313 components containing multiple heldout
source-role populations, but these overlap within the SAME heldout partition,
not between validation and test. These are task-view multiplicities, NOT 1,313
independent biological observations.

This establishes no collisions under the checked original native hash policy
in this exact, restricted seven-family scope. It does not establish that
the group vocabularies were independently standardized, the global split is
biologically leakage-safe, or remaining families are qualified.

## Method verification

The new test module first failed to import before the production implementation
existed. An additional red regression established that prior heldout-summary
logic did not distinguish validation/test collisions from overlaps within the
same heldout partition. Both were repaired. Five new focused tests, including
the real immutable sidecar readback, passed along with eight existing original
four-family group and component tests: 13 integrated passes in 26.51 s.
The independently persisted real-source execution passed in 17.93 s,
with peak resident memory 3,374,024 KiB. No GPU training, optimizer update,
counted proxy trial, target run or hyperparameter selection occurred.

Immutable evidence is
`outputs/common-source-nci-seven-chemical-20261111/actual-original-nci-seven-chemical-group-exposure-v2.json`,
SHA256 `ba7d32f6bab727110839b4aa46bc7adfeb87e660479b1bae83e6f27e1d669123`.
The earlier v1 result and both red-test diagnostic logs are preserved.
All new fixtures exist only under tests.

## Remaining prerequisites

Complete original expanded-source eligibility and cross-family grouping,
independent global split/identity adjudication, source-owner and model-owner
same-snapshot qualification, exact full 29-head native factory and matched
validation evidence, native operators/checkpoint/RNG/cursor recovery and
exclusive runner admission are still required. The strict 24 existing-grid
native muP proxy plus at most two ordinary ~80M native target run cap is
unchanged. This code neither relaxes admission nor writes campaign accounting.
