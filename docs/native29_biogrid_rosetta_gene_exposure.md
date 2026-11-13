# Original BioGRID/Rosetta gene and study grouping exposure

2026-11-11. One bounded **actual-original-source** qualification of the
train-side grouping records from BioGRID, Rosetta-L1000 and
Rosetta-CellPainting. Not an executable source admission, positive model loss,
independent benchmark, training run or μP hyperparameter selection.

The audit binds current pinned candidate and common 29-head source snapshots
(1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733
and 79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4),
checks all **173 original Parquet source shards** and original source-effective
group sidecars, and validates every retained original source-disposition count.
It reads only original provenance/task/group columns; no expression,
phenotype or other target value and no heldout/test payload was decoded.
Source/sidecar SHA256 manifest commitment:
c704c9c49874e9b66cc25d19f2168c6ddf2acab8e408fdd0c96454ac80e8b352.

| Source family | Original train rows | Original excluded | Retained |
|---|---:|---:|---:|
| BioGRID chemical-gene annotation | 2,916 | 210 | 2,706 |
| Rosetta-L1000 | 11,862 | 7,770 | 4,092 |
| Rosetta-CellPainting | 29,060 | 23,934 | 5,126 |

Exact literal gene:9606 overlaps, NOT sequence/protein-equivalence inference:
**21** BioGRID/L1000 gene tokens, **13** BioGRID/CellPainting and **142**
L1000/CellPainting. The two Rosetta sources also share **two** literal study
tokens. Other original cross-source matches include ncbi_gene,
perturbation_key and source_intervention. Such matches are group-exposure
evidence, NOT proof that an expression profile and CellPainting phenotype
form a paired intervention-response observation. No gene ID was invented
from a UniProt accession, nor any chemical structure from BioGRID IDs.

**2,188 retained BioGRID records have the lexical token
chemical_identifier:BIOGRID:-**. This appears to be a missing-value
placeholder, not a validated molecular identity. Under the literal
original source grouping one giant cross-source component has **11,711**
records: 2,493 BioGRID, 4,092 L1000 and 5,126 CellPainting. In a separate
**non-admitted sensitivity** omitting ONLY this placeholder, the largest
cross-source component still has **11,613** records: 2,395 BioGRID,
4,092 L1000 and 5,126 CellPainting. The original literal grouping has 46
components; the sensitivity has 69. The placeholder alone adds 98 BioGRID
rows to the giant component; study/gene/intervention exposure already joins
nearly all retained Rosetta records. **No source-policy changes were made.**
Neither diagnostic proves all component members are the same chemical,
independent experiments or validation leakage. No heldout split was admitted.

Actual original readback:
outputs/common-source-biogrid-rosetta-group-20261111/original-bioGRID-rosetta-gene-group-audit.json
SHA256 28e487a4b6b637758c68df1ab9bbebac4ad6498766a700c854b12e2d539cc687.
The original audit took **9.58 seconds**, peak **3,320,060 KiB RSS**, CPU only,
zero optimizer updates. **22 focused and native source regression tests passed**
in **24.08 seconds**, peak **3,673,104 KiB RSS**. An additional existing
Rosetta registry test failed because its executable module resides in the
isolated worktree rather than its required active-root factory path; its
diagnostic failure was preserved, without editing that other owner's test.
The initial combined run had 27 passes and this one root-binding failure.

The next critical path is independent construction-owner reconciliation
of global biological, study, chemical and source-split groups, retaining
honest sentinel semantics, exact same-owner matched validation,
complete eligible expanded source/model/operator qualification, distinct
genuine owner attestations and native recovery/resource admission.
There were zero actual 24-grid μP proxy trials and zero target runs in this
invocation, no chosen hyperparameters and no completed transfer.
