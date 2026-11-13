# Native29 DeepBacs original bidirectional source qualification — 2026-11-11

## Exact bounded scope

This continuation added `src/mora3/research/common_source_deepbacs_pair.py`
and `tests/research/test_common_source_deepbacs_pair_real.py`. It opened
**two genuine original DeepBacs-Staph training rasters**, with both original
`image_to_mask` and `mask_to_image` task records, under separately constructed
candidate and common29-head source bindings. It made **no changes** to the
original reader, source store, protected views, active owner, model weights,
checkpoint/optimizer, supervision schedule, or the read-only readiness watch.

The original source is `Zenodo5550933.v1`, https://zenodo.org/records/5550933,
with the source-reported `CC-BY-4.0` label. The reader independently pins
the original grayscale-envelope proof, effective view and complete train-role
raster payloads. Its original **test split was development-exposed**, so this
replay makes **no claim of a clean independent held-out validation benchmark**.

The inactive candidate source SHA256 is
`1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`.
The common29 source SHA256 is
`79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`.
The reviewed finite source view is
`c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.
Both direction-specific adapter hashes and initial cursor hashes are
authenticated against the immutable 62-route common-source rebind receipt,
not derived from an invented equivalent source identity.

## Actual native readback

For each of the two acquisitions, a train-role `image_to_mask` record at
original shard ordinals **0 and 2** is paired with a `mask_to_image` record
at original ordinals **1 and 3**. Across candidate and common snapshot
identities, all original Arrow fields, source payloads, original ordinals,
typed source-native raster bytes, masks, shapes, group names, task roles and
next positions matched. The second row of each of the four routes was
reproduced independently from its exact previous-window cursor. The distinct
snapshot identifiers correctly retain distinct serialized cursor hashes.

The original frames and labels are complete **512 × 512 unsigned 16-bit**
arrays (`>u2`), or **524,288 decoded bytes each**. The forward mask objective
prepares **785,408 supervised positions per image** (all pixels plus valid
right and down affinities). The reverse detector flow prepares **262,144
positions per image**. Across the two original acquisitions the total is
**1,570,816 forward plus 524,288 reverse positions = 2,095,104**. This counts
two task directions over two source images, not four independent microscopy
experiments. No objective closure was invoked and no optimizer steps occurred.

The source's `image_sha256` and `mask_sha256` are hashes of original
provider **image/mask members**, not necessarily decoded pixel arrays.
An initial failing implementation incorrectly compared those hashes with
the decoded raster hashes. This was corrected by verifying both independently:
provider-member digests through source `Provenance`, decoded pixel and
label digests against the complete original native byte arrays. Both types
of hashes are retained distinctly in the immutable readback. Raw arbitrary
mask label IDs are preserved; any canonical partition mapping is not treated
as new biological evidence.

The native source relationships and source parent group are conserved.
Forward conditions contain only the image, with labels as the target;
reverse conditions contain the observed labels, with the image as target.
The directions are explicitly different original task records sharing one
raster acquisition. No segmentation quality, biological causal perturbation,
cross-domain pairing, independent group/split acceptance or phenotype prediction
was established by a source replay.

## Verification and evidence

Red-first module-import failure:
`outputs/common-source-deepbacs-paired-20261111/expected-red.log`.
The first integration attempt retained two failing assertions: an incorrect
provider-vs-decoded-hash assumption and a malformed mutation-test case;
`first-green-attempt.log` retains both failures. The amended focused run
passed **8 tests** (`focused-v2.log`). Native morphology/raster regressions
passed **6 tests** (`native-raster-regression.log`). Combined genuine
DeepBacs plus existing GPerturb source regressions passed **17 tests**
(`original-source-regression.log`), before the final type-exact adapter
refinement, which is covered by the subsequent rerun.

The readback is
`outputs/common-source-deepbacs-paired-20261111/actual-original-deepbacs-two-direction-raster-pairs.json`,
SHA256
`179fe5de78039f4e18fa9aa6dd1ee77097650ddd46280d3390d72520c06d5ca6`.
Source rows are verified, **not** campaign trials. Current user limit:
**24 predeclared native muP proxy slots plus at most two ordinary ~80M
native muP target runs**, with no added retries/comparisons. In this task:
**zero optimizer updates, zero counted runs, no winner, no target transfer**.

## Remaining critical path

This two-window source audit does not exhaust the DeepBacs source, replay all
62 routes, qualify global grouping, create a new exact-owner matched
validation baseline or satisfy distinct construction/model owner attestations.
The complete expanded corpus remains unready and actual
`SourceBoundOwner.require_training_admission` continues to refuse.
Preserve the fixed source/model/grid and owner ownership while completing
genuine remaining source eligibility and measured reference/recovery checks.
Do not turn the result into a validation/test selection or training launch.
