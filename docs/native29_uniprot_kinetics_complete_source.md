# Original UniProt kinetics: complete retained source replay

Date: 2026-11-11. One bounded **source-qualification** operation, not a
training run or validation reference. This extends, rather than repeats, the
two-row source fidelity at commit `994fb14`.

The original three UniProt2026_03 nonrandom-first100 train shards have **622
original task records**. The immutable current view retains **618** and
quarantines **four** via native ordinal dispositions; neither excluded values
nor source roles were rewritten. An actual readback exhausted **all 618
eligible train records** with separately reconstructed candidate and common
N29-head adapters. Every Arrow row/ordinal/payload and typed scalar projection
matched, with an independent stdlib reference for each source JSON numeric
value, lexeme and literal unit. Both native cursors reached (3,0). Four
independent restart checks covered the two cross-shard boundaries under both
source bindings. The frozen campaign limit of TWO source windows per native
training update was not changed: this is a separate source audit.

The retained measurements are **433 Km** and **185 Vmax** source-reported
values. Km uses original mM (107), nM (3), and uM (323). Vmax retains 11
distinct source-literal unit strings, ranging from pmol/min/ug to
umol/min/mg; unlike Km, it is not a concentration. No unit conversion,
inferred kcat, comparable assay standardization, target-to-condition leak,
noise calibration, independent experiment declaration, or causal assertion is
performed. The original data do not specify measurement-linked assay method,
pH, temperature, construct or parent molecule context. The source is a
nonrandom first100 subset, not representative of the complete UniProt
proteome. Evidence codes or citations remain reported annotations, not
guarantees of independent measurement validity.

The original sidecars and disposition bytes were authenticated separately.
Each retained row's literal original group set is compared to its immutable
protein-threshold effective group set; transitive closures span accession,
sequence, PubMed and detected homology identifiers. They yield **87 declared
source-local components**, the largest containing **71 records**. This
grouping is a leakage precaution, not a proof of global source-independent
biological train/validation/test separation or complete homology coverage.
There is no original validation/test kinetic role in this selected source
route. No target values from other roles are decoded.

Verification is in
`outputs/common-source-uniprot-kinetics-complete-20261111/`, including the
SHA-pinned full 618-row native readback and actual-execution resource receipt.
The readback took 13.10s at roughly 3.16GiB peak RSS on a CPU-only run;
856.3GB disk remained free, above the 64GiB reserve. All source fields and
protected ledgers remain in place; no caches or data were reindexed.

This establishes **one** complete native kinetic train route under the
candidate/common N29 source identities, not all 62 routes, expanded dataset
readiness, distinct construction/model owner attestations, a same-owner
validation baseline, comprehensive source/model qualification or training
admission. The 24 native proxy slots plus two native ~80M full-target slots
remain unchanged, with zero optimization updates or actual runs here.
