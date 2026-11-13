# Native29 source replay: original protein-prior homology quarantine

## Verified scope (November 10, 2026)

The seventh native prior, `native-prior:protein`, **does not use** the same
training source-view digest as the preceding six. Its frozen common29-head view
is `2888d9f11038f70f9b25a049c8a006d87ca9918de6e94adcb71ec5ee1c9fd34b`,
not the unquarantined effective-study-v2 digest
`c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.
The difference is an original, pre-existing v6 one-row source-context/homology
exclusion: `native-common-chemical-quarantine-v6-20261108/bound-route-00-quarantine.json`
SHA256 `b463ee189a3a39e916bf9ea46c3684496e498f81864ea9e914bfb8f3ff3ecd00`.
It excludes original UniProt train shard ordinal 24 (source 2026_03, protein
sequence modeling), without deleting, relabeling, changing or truncating any
source data or altering any existing evaluation group. This is a train-only
selection control, not a proof of global homology leakage closure.

The implementation now authenticates the original v6 policy index and policy,
opens the original 256-row Parquet shard within an explicit source-byte budget,
verifies the full shard SHA256
`9eb37df985160d2d7db5d49e458a96a737e35ae2ea092ac57d9c0e46a8c27326`,
and independently checks that ordinal 24 is the original train UniProt protein
sequence task with the original pinned native payload digest
`53437fa2c97550f993fdefd6e29f47f96afbeaf3855d77775931d2d5e2fb687c`.
The original policy's opaque selection-identity and chain digests are retained
verbatim; hashes alone do not prove biological or causal equivalence.

Exactly one extra exclusion is applied to the original base227 study exclusions.
The candidate and common source views thereby both have228 excluded original
coordinates. Their exact independently computed view-policy digests match the
previously frozen candidate `4f144c456c46a393e678d640cc96c99e4d4b7af793174b50b49971088fb8f227`
and common `2888d9f1...` commitments. Original descriptors, source manifest,
shard/ordinal layout and all unaffected route views retain their identity.

## Important pre-existing owner distinction

The original reviewed quarantine rebind changes the native **iterator** view;
it does not replace the already pinned **target-preparation callback**. Creating
a new `NativePriorCampaignAdapter` directly on the quarantined view would retain
cursor identity but change its executable-adapter SHA256, making it incompatible
with the reviewed campaign. This difference was reproduced in a failing test.
The corrected code uses the original base-view adapter/codec/target preparation
and the separately reconstructed quarantine-view route, preserving the original
binding's cursor/adapter commitments exactly. The train reader itself enforces
the exclusion. Future worker qualification must independently ensure that a
source window supplied to `prepare` always originated from that same protected
route; construction receipts alone are not execution authority.

A direct original-source iterator test starts at ordinal24. The unquarantined
reference returns original source ordinal24, while both candidate and common
protected iterators return ordinal25. This is a demonstrated **actual row
selection difference**, not merely a metadata-hash agreement. No row is rewritten
or moved. The original source remains present for reproducible auditing.

## Actual source-window and conservation results

The exact candidate and common bound original train routes read two genuine
protein sequence windows. They retain source ordinals0 and1, with original
full target lengths594 and691 UTF-8 bytes, and516 and595 fitted source-native
byte/EOS tokens: total**1,111 observed training-token positions**. The
original raw Parquet tables, record/payload SHA256, target counts, source roles,
source positions, and independent next-window cursor restarts matched across
both distinct snapshot bindings. Both original targets satisfy the unchanged
4096-byte per-window preparation budget; no over-budget rejection was needed
for these two rows. This does NOT establish source exhaustion or all82 scheduled
protein updates, complete62-route replay, a fresh same-owner matched validation
reference, or any global heldout group/split admission.

The machine-readable source result is
`outputs/common-source-protein-quarantine-20261110/actual-protein-quarantine-two-window-readback.json`.
The focused test and six previously qualified native-prior regression receipts
are adjacent. No optimizer updates, model weights, checkpoints, GPU training or
real26-run campaign reservations were created.

## Next bounded critical path

The seven native-prior route types now each have scoped two-window original-source
and restart checks under their proper distinct view policies. The rest of the
62-route train/validation mixture remains unqualified. Next, move to a distinct
real original source route with its own exact filter/cursor/codec and record
actual bounded source-target conservation. Continue towards a complete common
29-head source replay and a **new matched reference for that exact owner**.
Do not reuse another owner's old validation-loss baseline, remove the original
protein quarantine, or substitute unit-test fixtures for real training evidence.
Full expanded-data readiness, current owner admission, actual24 μP sweeps and
post-sweep hyperparameter transfer remain closed until independently qualified.
The maximum is unchanged at24 proxy trials and two native80M target runs.
