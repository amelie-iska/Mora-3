# Package 3 — GRH intake and expanded-source reconciliation

## Pinned GRH source and original denominator

Release `AmelieSchreiber/GRH-Mora-ConformerResponse-20261104`, revision
`53a6b66acaa69ef79584648f3f6dc068363c4837`, was reused read-only. Final-manifest
SHA256: `3610fcb330ee2b2b89037a38a39bf8c0509190b52f61a92313422f07ea9c5cad`.
No repository scripts or pickle artifacts were executed and no data downloaded.

| Configuration | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Static data | 22,067 | 434 | 450 | 22,951 |
| Source anchors | 266 | Absent | Absent | 266 |
| Electronic dynamics | 84 | 69 | 88 | 241 |
| Nuclear MD | 4 | 4 | 4 | 12 |
| **Total original records** | **22,421** | **507** | **542** | **23,470** |

All ten original Parquet files were independently hashed and streamed. Native
counts match their footers and declared splits. Of these rows, 23,025 are valid
static native MMFF94 energy/force candidates for the implemented view; 445 have
other methods or dynamic representations. There were zero candidate-schema
rejections and no duplicate record IDs within a configuration. **New training
admissions are zero**; candidate counts do not establish source rights, global
heldout safety, unique chemical entities or independent experiments.

## Native fidelity and computed-target handling

`grh_intake.py` keeps original condition/target JSON separate. IDs, groups,
source origins, lineage, method, units and payload digests are audit data, not
shortcut predictors. Static MMFF94 input coordinates and force targets preserve
the same ordered atom frame, angstrom, kcal/mol and kcal/mol/angstrom; absolute
energies are not relabeled cross-molecule atomization energies. Computed targets
never pass through an experimental-only admission bypass. Electronic and nuclear
payloads remain native; the reader does not invent a universal objective for
uncalibrated electronic outputs or treat trajectory frames as independent cells.

`grh_conservation.py` independently compared **300,327 duplicate native column
values** against source JSON, validated bond endpoint/order/stereo-map structure,
and checked exact JSON/payload agreement. Zero rows failed those checks. Exact
within-release parent closure resolved **253/253 parent-record links** into
**71 connected components**, with zero conflicting official split components.
This includes the declared groups and explicit data-parent relationships; it
does not add inferred chemical similarity, PDB homology or external corpus links.

The full scan used about 452 MB peak process RSS; the distinct conservation scan
used about 437 MB. Neither emitted protected coordinate values in reports.

## Other lanes and ownership

`audit_alternate_sources.py` verifies the pinned existing expanded witness and
its lane evidence without repeating source acquisition or changing owner ledgers.
It carries original evidence hashes and explicit unknown full eligible
numerators/denominators. Known historical counts are not promoted to admission.

| Lane | Existing scoped evidence | Remaining source admission issue |
|---|---|---|
| X-Pert | Norman 111,445 x 33,694 and L1000 678,401 x 978 native stores | Complete source/control/physical-context masks and official/global groups |
| Chemical retrosynthesis | 103-source census, 3,224,294 nonadditive occurrences | Recorded role disagreement/UNKNOWN dispositions, chemical/route closure and materialized native tasks; evaluation-only references excluded |
| Peptide retrosynthesis | Limited candidate/reference releases | Complete route/synthesis/yield authority and source-specific rights; no invented eligible denominator |
| MINT | Large retained source reconstruction and partial native work | Original release/build/filter/order/permutation/split authority and guarded-stop continuation |
| Transition-state enzyme | Existing source/constraint contract and separately owned reader work | Complete original source/frame/role/units/heldout and constrained export qualification |
| ProCyon/scCello/LINCS/ADMET/UMA-eSEN | Existing owner inventories and staging retained | Existing source/model owners retain acquisition, curation and operator acceptance; no duplicate jobs |

The active RGD1-v6 energy quarantine remains an override irrespective of old
admitted flags. Source anchors, calculated properties, evaluation references,
unknown labels and missing physical units do not become observed train labels.

## Authoritative receipts and next gate

Under `outputs/alternate-20261108/`: `source-audit-v1/grh-candidate-audit.json`,
`source-audit-v1/lane-reconciliation.json`, `source-audit-v1/receipt.json`,
`grh-conservation-v1.json` and execution receipts. The handoff seal supplies
input/output SHA256 values and code bindings.

Next: source-owner review of original anchor rights and provenance; construction
owner cross-corpus exposure/similarity/homology closure and authoritative masks;
model-owner review of each remaining method-specific objective and native
registration; complete common snapshot and distinct matching owner attestations.
The within-release audit is not a substitute for any of these gates.
