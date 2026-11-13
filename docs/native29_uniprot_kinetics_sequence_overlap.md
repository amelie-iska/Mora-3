# Original UniProt kinetic/sequence-prior cross-source train exposure

2026-11-11. This is one **real-source grouping-qualification task** against the
isolated campaign26 implementation and protected active Iska originals; **not**
a training run, model evaluation, globally admitted split or hyperparameter
selection.

The audit independently authenticated 410 original training Parquet shards and
their original effective biological-group sidecars under the retained
`effective-study-v2` source view. It compared all **622 original nonrandom
UniProt2026_03 kinetic records (618 retained, four excluded)** with all **104,176
original UniProt2026_03 protein-sequence-prior records (103,171 retained, 1,005
excluded)**. It selected only original train rows, preserved source ordinals,
read source group identities and typed metadata but **did not decode native
protein or kinetic target payloads**, open held-out source shards, or call model
objectives. All 618 retained kinetic source coordinates and group hashes
revalidated against the independently completed prior full-original kinetic
readback SHA256
`6d9922b1af68973836cac5dac63c10b2e9ab656014156219b4ea3dcbf4ec1c0d`.
Source revisions and rights remain bound to existing immutable original
inventories, not inferred from upstream availability.

The cross-source join uses ONLY exact literal original group tokens. It treats
`accession:`, `sequence:`, and `detected_protein_guard:` identifiers as
different evidence categories and `pubmed:` as a possible shared-study split
risk, not exact protein identity. No new homology cluster, transitive chemistry
equivalence, experimental independence or measured kinetic evidence was created.
Direct shared identifiers comprise **93 accessions, 93 sequence-group hashes and
89 detected-homology guard groups**. Including existing source-local grouping,
the two training source families contain **87 cross-source connected
components**, exposing **all 618 retained kinetics records and 1,123
retained sequence-prior records** to cross-source train-group co-membership.
The largest resulting mixed component contains 143 records. The complete
union graph has 39,431 components. Identity-only edges produce 89 mixed
components; adding all declared study/source grouping can merge those
components. A source-local count or a direct shared ID is not a declaration
that two tasks are independent experiments or identical measurement protocols.

The canonical audited readback is
`outputs/common-source-uniprot-cross-exposure-20261111/original-uniprot-kinetics-sequence-exposure.json`
(SHA256 `34c6953bcdcabc5a16c85b458b27e0c048772462eb1eccf7361dd176d1ddce03`).
The first source run refused an incorrectly estimated 700–900-shard inventory;
the actual original families contain 407 sequence and three kinetic shards. The
guard was corrected to that independently checked count, and the successful
real-source scan retained both execution logs. The successful scan took **10.71
seconds**, approximately **3.17 GiB peak RSS** (3,320,464 KiB), CPU only, with
zero model forwards and zero optimizer updates. Available disk remained over
855 billion bytes, exceeding the 64 GiB reserve.

This is **train-to-train group exposure**, not demonstrated evaluation leakage.
It establishes a concrete constraint for any future independent global
train/validation/test partition: original kinetic groups must not be split
independently of these protein-prior groups. No held-out/test group closure,
full expanded corpus/source release, distinct actual construction/model
attestations, matched same-owner 29-head validation reference, numeric
operator/optimizer recovery qualification, or counted μP run is established.
The exact 24 proxy + at most two μP 80,283,064-parameter targets remains
unchanged. The existing read-only watch and active checkout are untouched.
