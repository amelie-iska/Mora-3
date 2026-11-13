# UniProt source-reported kinetic values: two original windows under common 29-head bindings

## Scope and evidence

This bounded task replays two actual original train windows for the distinct
`curated_kinetic_annotation_value` route of the pinned
`UniProt-kinetics` source. It compares the exact original Arrow row, typed
numeric source projection, executable scalar adapter, source/split roles,
source positions, cursor and independently restarted second window across the
inactive candidate source binding and the common 29-head binding. It does
**not** admit training or assess model predictions, full corpus completeness,
sweep results or biological/clinical efficacy.

The installed original-source view policy is
`c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393`.
The unchanged original input shard has SHA256
`fc2c05d7e09952daa0cbefaa1e9b45c26d5dcfb5e989123d0238f661aa46354b`.
The candidate executable adapter hash is
`015321a2faabd098dec5ae6f2f68d3ca18aed8597e5f2c8ae23a7faaffd20ab6`;
the corresponding common-snapshot adapter hash is
`8bb5a4a1b4065ee01ccf890a2342b43f007a57b37c2aa9cac9578bbfbc83aa26`.
These intentionally differ by source snapshot binding. The independent
original cursor hashes also differ as required, while actual row identities,
payloads, finite measurements and cursor positions remain the same.

The historical source/route selection provenance is the exact retained
`outputs/primary-campaign-common-source-rebind-20261108-v1/common-source-rebind.json`,
and native scalar and iterator operations are the installed
`mora3.research.remaining_eight_registry`,
`TrainOnlyScalarCampaignAdapter` and
`project_native_scalar_batch`. All source datasets are read-only.

## Two genuine original observations

| Original ordinal | Source-reported quantity | Literal source unit | Original numeric lexeme |
| --- | --- | --- | --- |
| 0 | Km | uM | 2.6 |
| 1 | Km | uM | 11.7 |

Both original records are source-reported **Km** concentrations; they are
not Vmax observations, kcat values, clinical measurements, or predictions
generated during this work. Their exact Float64 value bits, native numeric
lexemes, source target envelopes, protein identity digests, substrate alias,
source coordinate, task/condition/audit metadata, original units and partition
are retained. These literal units have not been silently converted or
normalised.

The existing native projector distinguishes `Km` (substrate alias and
source `constant`) from `Vmax` (enzyme alias and source `velocity`).
Its separate existing fixture test verifies both interpretations, but **no
actual Vmax source row was included in the two selected windows**. In
particular, Vmax cannot be converted to kcat without suitable enzyme
concentration, construct and assay information. The original selected
records do not specify method, pH, temperature, assayed construct or
parent comment-molecule scope. Missing metadata are preserved as missing,
not guessed from the protein or substrate identity. Two distinct original
rows do not prove independent experiments or leakage-safe group separation.

Both candidate and common iterators independently reproduced the exact
second original row after restarting at the first recorded cursor. This
verifies the bounded two-window cursor transition, not full exhaustion of
this route's original training data or its 81 scheduled occurrences.

Machine-readable genuine-source readback:
`outputs/common-source-uniprot-kinetics-20261110/actual-original-uniprot-kinetics-two-window-readback.json`
SHA256 `241534e4944ac5fa4a3c9696b090e26af091b2b82b9c341b6abd5c0b32c48738`.

## Tests, operational constraints and limitations

A missing-module failing test preceded implementation. The first live
implementation test detected a mismatch between a presumed entity key and
the actual source schema, `molecule_class`; a second test detected a
test-only mismatched receipt key. Both were corrected without editing
original source bytes, the underlying scalar projector, effective filters,
original native adapter, source ledgers or training gate.

The focused regression ran 29 passing tests in 23.68 seconds of pytest
time, against the original installed source with the new isolated code
modules, using existing `tokengt` Python, a single CPU thread and CUDA
hidden. The command, stderr, source hashes and result are retained in
`outputs/common-source-uniprot-kinetics-20261110/focused-regression.json`.
The suite includes the seven new scoped real-source tests, original
FreeSolv/AZ source regressions and existing Km/Vmax scalar semantics
fixtures. The trained native model, owner evaluator and optimizer were
not invoked. Zero optimizer updates and zero counted sweep runs occurred.

This does **not** satisfy all 62 source routes, complete original-source
eligibility, independent global grouped split checks, both distinct
same-snapshot owner attestations, complete29-head/SCRI/trajectory operator
qualification, full model and checkpoint recovery, measured target runtime,
or a fresh same-owner matched validation reference. The source-bound
training admission remains closed. Do not reuse historical reference
losses from another source/model identity.

The latest direct user limit remains **24 actual native μP proxy trials
plus at most two μP approximately80M target runs**; none was consumed
by these source-only checks. The next bounded work should proceed to
a different, still-unqualified nonprior route or the actual same-owner
matched validation baseline once its source admission can be shown,
without turning these two-window checks into a false completion count.
