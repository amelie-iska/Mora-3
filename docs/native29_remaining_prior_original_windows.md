# Common 29-head source replay: four additional native-prior routes

Date: November 10, 2026 (PST). Scope: four distinct, genuinely original train-source routes, **exactly two original one-row windows each**. This is one bounded source-fidelity qualification, not μP optimizer training, a whole original-corpus pass or a validation baseline.

## Exact frozen source bindings

The original effective-study-v2 native train source-view policy is `c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`. The inactive candidate snapshot `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733` and native 29-head common snapshot `79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4` are distinct. Each route reconstructed its actual fitted native codec and typed source adapter, compared its current executable hash and initial-cursor hash with the frozen common-source rebind, then read the SAME original Arrow source rows under both snapshot labels. The full row schema/bytes, source shard/ordinal and both resulting cursor coordinates agreed; a separate read from the first recorded cursor reproduced the second row and preparation disposition. Source inputs were never re-split, truncated or re-labeled.

## Actual original data observations

| Source task | Original ordinals | Original target bytes | Source-native target tokens incl. EOS | Budget-refused rows |
| --- | --- | --- | ---: | ---: |
| `native-prior:rfam_rna` | 747, 748 | 104, 104 | 210 | 0 |
| `native-prior:rnacentral_rna` | 0, 1 | 200, 363 | 565 | 0 |
| `native-prior:smiles_to_selfies` | 0, 2 | 195, 125 | 322 | 0 |
| `native-prior:selfies_to_smiles` | 1, 3 | 37, 33 | 72 | 0 |

**Eight real original rows; 1,169 native tokens permitted by the existing 4,096-byte per-window preparation cap.** These eight source windows are not eight training runs. The Rfam source's starting ordinal 747 reflects existing original retained source-role filtering, not a newly assumed sequence beginning at zero. The two directional SELFIES/SMILES tasks use alternating original ordinals in the same source shard; they are not considered interchangeable molecular examples or an authorized cross-task join. RNA tasks retain the native nucleotide-symbol codec; conversion tasks retain the fitted shortest-path byte codec.

The SHA-bound fresh-process evidence is `outputs/common-source-remaining-priors-20261110/actual-four-prior-readback.json` (SHA256 `d17769dba14c2139a0350e92435de7b5992663efe216a6838d9671d65124c1fa`). This process took 31.005 seconds for four routes, peak resident size 3,145,356 KiB, one CPU thread and no CUDA. It did not advance a model, issue a training or source admission or perform a target evaluation. The preceding new interface test first failed at collection; its record is `expected-red.json`. Initial real tests: `first-real-test.json` (11 passing checks,32.14 seconds). No modifications to the source view, fitted vocabulary, immutable rebind or native prior adapter occurred.

## Explicitly unresolved source-specific protein policy

`native-prior:protein` is **not** included in this common-view audit. Its frozen common-source rebind demands view-policy SHA256 `2888d9f11038f70f9b25a049c8a006d87ca9918de6e94adcb71ec5ee1c9fd34b`, distinct from the witnessed effective-study-v2 base policy. The reference-digest difference is a concrete source-policy binding mismatch, not evidence that protein observations are invalid. The current audit refuses a protein key before reading source data. A separately authenticated source-specific view, original split/group semantics and recreated exact adapter/cursor commitments are required; silently substituting the base view or mutating the frozen rebind would invalidate provenance and selection integrity.

## Remaining native execution qualification

Previously completed two-window native-prior audits cover DNA and native SMILES. Together with this task, six of seven native-prior **route types** have scoped two-original-window provenance checks; this is NOT full source-corpus coverage, not 62/62 route replay and not proof of global group/split safety. The original 5,000-update/5,940-window schedule and its per-route scan and cursor rules remain frozen, and no historical reference losses may be transplanted between owner snapshots. Required next work is the protein source-view reconciliation followed by complete registered route/window validation and a fresh matched baseline for the actual 29-head owner. Other required modalities, SCRI and trajectory reasoning, both distinct same-snapshot owner attestations, target configuration/recovery and runtime resources remain separate admission gates. Actual μP trials begun: **zero**; optimizer updates: **zero**; target runs: **zero**. Maxima unchanged: **24 proxy + 2 approximately80M native target runs**.
