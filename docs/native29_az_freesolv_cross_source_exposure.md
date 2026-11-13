# Original AZ / FreeSolv cross-source training chemical exposure

Date: 2026-11-11. This is one bounded **source-only** qualification for the
STAGED29-head Mora-3 campaign, not a training run or an approved global split.
It compares **all** verified original ChEMBL37-AZ logD7.4 training records
(4,184 in 17 original shards) with **all** 556 FreeSolv training records (three
original shards). Both original full-source replay receipts were authenticated
against their previously pinned SHA256 values, then every current original
training Arrow payload and effective-group sidecar was reopened using its
existing exact-byte digest; record identity, task, original ordinal, role,
source-shard SHA, group identity and payload SHA were cross-checked. No protected
source or model file was changed.

## Actual chemical exposure result

With existing `tokengt` RDKit 2026.03.1, native source SMILES were parsed and
converted separately into canonical **isomeric** and **non-isomeric** SMILES,
without salt stripping, charge-state normalization, tautomer conversion or
experimental-assay equivalence assumptions. This is an exact, deliberately
bounded chemical-representation comparison, not a global structure ontology.

- **10 distinct isomeric chemical structures** occur in both original training
  populations: ten exact AZ–FreeSolv record pairs, involving ten FreeSolv
  training records.
- **11 connectivity structures** occur in both sources. Three additional
  cross-source record-pair candidates arise from unspecified versus specified
  stereochemistry, covering one further FreeSolv training record. These are
  `connectivity_only_matches`, **not** confirmed exact isomeric identities.
  Their original stereo representations are retained side by side.
- The initial original **source-local** grouping has 300 components (the AZ
  assay/document sidecar connects all 4,184 AZ records; FreeSolv supplies its
  declared local component/molecule/scaffold links). Source names are kept as
  explicit namespaces so unrelated group-key hashes never merge implicitly.
  Adding **only ten exact isomeric cross-source edges** reduces this to 293
  two-source components. One cross-source connected component contains 4,369
  training records: the 4,184 AZ records and **185 FreeSolv records**. Only ten
  of those 185 FreeSolv records have a direct confirmed exact-structure link;
  the other 175 are connected **transitively by existing FreeSolv declared
  grouping**. They are not asserted to be the same molecule or independent
  experiments.

These are real original *training* exposure collisions. They constrain future
combined source-group closure and selection of an independent validation
reference; they do **not** prove that cross-source molecular measurements are
replicates, that hydration free energy equals logD7.4, or that all 4,369 records
are chemically identical. The original AZ records already share one source
assay/document; the connected-component size is consequently very conservative.
The measured scalar units and epistemic categories remain separate. Original
source-local group hashes are not silently interpreted as cross-source aliases.

## Data roles and release boundary

Only 20 original **training** shards were scanned. FreeSolv's 42 validation and
44 test targets, their source payload files, and all other possible held-out
objective targets were untouched. In particular this operation cannot assert
whether AZ training records expose FreeSolv's held-out chemicals. No test data
were used for selection. This is partial two-source closure, not complete global
chemical/biological grouping over the other 60 native route types, homology,
assay/publication roles or all mandated modalities. It supplies no new matched
validation loss for the final approximately 80M owner and no construction/model
same-snapshot attestations. Training admission remains closed.

The historical candidate source SHA256 is
`1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`;
the common 29-head source SHA256 is
`79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`.
Those binding identities are attested by the **prior real original-source
readbacks**, not by a new model construction or factory training release here.
The original shared source-view policy remains
`c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.

## Evidence, tests and next execution boundary

The independent original-source exposure artifact is
`outputs/common-source-az-freesolv-exposure-20261111/actual-original-az-freesolv-training-exposure.json`,
SHA256 `f560c172b43ef62bec86bb8aca088939e9942882acfceafa7251bdcb2c405e3d`.
The original readback references are retained, not rewritten:

- AZ `actual-original-az-logd-complete.json` SHA256
  `1a6e1dfdb26ccfe2693e974b79870ea3d1763771206dd1a366eba8a02f8151d7`.
- FreeSolv `original-complete-freesolv-readback.json` SHA256
  `f0820098583bd37155b55f36716b8734649613faa1d897250c2d89759d7e51a9`.

An expected missing-module red was retained, followed by a corrected-runner
root failure (the original source must be accessed from its exact checkout),
and a corrected real-source expectation distinguishing three connectivity-only
record pairs from one additional FreeSolv molecule. Final integrated execution:
**42 tests passed**, zero skips, including original AZ and FreeSolv complete
source replays and both earlier source/regression suites. The CPU-only combined
suite took 79.90 seconds with maximum RSS 3,420,060 KiB; the separate actual
cross-source readback took 0.849 seconds with peak RSS 186,504 KiB. Disk free
was 856,245,817,344 bytes, comfortably above the 64 GiB reserve.

No optimizer updates, campaign reservations, proxy trials, target-training runs,
checkpoints or hyperparameter selection occurred. The existing 24 native μP
proxy slots and at most two approximately 80M μP target runs remain intact.
The next qualification should close genuine train/held-out chemical identities
with an explicitly authorized metadata-only/target-inaccessible reader, then
continue global biological group and multi-source route coverage before a
matched same-owner reference and independent admission. Do not override the
training gate or promote these two-source components to globally accepted roles.
