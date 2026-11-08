# SCRI extension: source contracts, shared objectives, and sweep preparation

Status: **additive implementation for review; full-corpus training is not admitted**.

This extension prepares spatial cell-state, perturbation, bioelectric, and
localized-design work in a namespace separate from the active Mora-3 campaign
implementation. It is not an alternative primary training owner. The existing
native owner, complete-corpus requirements, operator qualification, and campaign
recovery and hardware gates remain authoritative.

## Scope and isolation

The implementation lives in `src/mora3/extensions/scri/`, with an independent
CLI in `scripts/scri.py`, source catalog in `configs/scri_sources.json`, and
focused tests under `tests/extensions/`. Importing the package does not acquire
data, import external model repositories, run BETSE, start training, or modify
the primary owner. New reports and curation outputs use exclusive creation so
an earlier evidence artifact cannot be overwritten accidentally.

The 2026-11-08 branch is based on published Mora-3 commit
`f74ff02d88bb9e12b02e835b541c27486298f580`. The active checkout contains newer
campaign changes and remains the source of truth for ongoing development. Do
not replace its working tree with this clone. Cherry-pick or otherwise review
only the additive extension after the active owner reconciles its interfaces.

## Scientific data contract

`contracts.StateRecord` retains native feature identity and order, measurement
units, explicit missingness, organism, time and event reference, interventions,
spatial coordinate frame, and source provenance. Numeric values must be finite;
counts cannot be negative or fractional. Missing values are `None` with
`observed=False`; a measured zero remains observed. Source digests and identifiers
are audit coordinates, not learned predictive features.

Every record labels its evidence as experimental, simulation, model prediction,
or synthetic fixture. The schema records claims supplied by the source adapter;
passing its validation does not establish original-data fidelity, rights, or
biological validity. Those require independent source and owner review.

`TransitionPair` requires explicit pairing evidence, matched known donor,
study, experiment, organism, and strictly ordered native times. A same-cell
claim additionally needs declared longitudinal lineage evidence. That declaration
must be checked against the provider's experimental design; lineage membership
alone is not sufficient biological proof of repeated observation of one cell.
Matched population data remain population-level evidence. Independent snapshots
cannot be relabeled as observed single-cell trajectories.

`compile_transition` keeps outcome values outside conditions, preserves the full
native feature axis and units, and records the pairing evidence separately. It
does not infer a causal effect from an observational pair.

### Leakage controls

Splits are assigned to connected components of declared study, donor, sample,
experiment, lineage, intervention, and explicit cross-source identity links.
Repeated use of the same original observation always creates a linkage.
Requested holdout identities must be known. Official provider splits are
preserved; contradictory official assignments in one component are rejected.

The default study-and-donor split is intentionally conservative. It can produce
an empty validation or test partition for a small collection. The implementation
does not split a donor or study merely to fill the requested percentages.
Training admission must check complete, scientifically appropriate partitions
and native target denominators for each task.

### Bounded curation

The supported interchange formats are exact native-record JSONL and a documented
long CSV table. They are not universal readers for provider archives. H5AD,
Zarr, matrices, imaging files, and provider-specific metadata still require
source adapters and independent semantic verification before use.

Long CSV has the following exact column order:

```text
record_id,study_id,sample_id,donor_id,experiment_id,lineage_id,organism,cell_type,time_value,time_unit,time_reference,coordinates,coordinate_unit,coordinate_frame,intervention_ids,official_split,modality,feature_id,value,unit,observed
```

Coordinates and intervention IDs are JSON arrays. `observed` is the literal
`true` or `false`; a missing value is empty and requires `false`. Categorical
values use the `category` unit. Source bytes are hashed before and after readback;
source row locations and feature order are retained. Limits reject the complete
operation rather than silently accepting a source prefix as full coverage.

Candidate curation creates `records.jsonl`, `splits.json`, and `receipt.json` in
a new output directory. The receipt reports evidence kinds, measured/missing
values, partition sizes and hashes. Both readers return an immutable
`ParsedSource` sequence bound to the exact consumed bytes and parser. Curation
rejects changed input or an unrelated record tuple before creating outputs.
It does not promote candidates to training
admission, and it does not claim an arbitrary reformatted input is the original
provider artifact.

## Required task families and evidence

| Family | Required native evidence | Key limitation |
| --- | --- | --- |
| Identity and state | Native cell identity/state labels and full feature axes | An atlas label does not establish reprogramming success |
| State transition | Explicit longitudinal or matched-population pairing, time and donor | Independent single-cell snapshots are unpaired |
| Perturbation response | Intervention, dose/time, controls, experimental unit and response | Treatment associations are not automatically causal effects |
| Spatial neighborhood | Actual coordinates, tissue frame, units and neighborhood construction | No invented coordinates or cross-section geometry |
| Temporal trajectory | Native time reference and experimentally supported links | Pseudotime is not elapsed time |
| Bioelectric state | Measured or explicitly simulated voltage/channel/junction variables | Simulation outputs remain a separate evidence class |
| Functional phenotype | Source-defined functional endpoint and denominator | Morphology or transcript changes alone do not prove function |
| Localized inverse design | Admissible actions, region, budget and measured outside-region outcomes | Action constraints do not prove physical effect containment |

Existing Mora-3 TokenGT, ConvexTok, tropical-attention, FoT/GFlowNet, GraphCG and
native biological objectives retain their own implementation and qualification
requirements. External STATE/Nicheformer/BETSE references do not replace the
joint Mora-3 owner or imply that pretrained weights have been incorporated.

## Shared model and spatial constraints

`spatial.build_neighborhood_query` builds actual Mora-3 `GraphRecord` and
`ConditionalTask` objects from a coherent baseline specimen. It selects the
complete native-radius neighborhood, preserves missing baseline measurements,
and creates reciprocal directed proximity edges for the existing payload
encoder. The direction is an encoding convention, not a causal relationship.
Only the query node has missing target placeholders. Source IDs and audit
metadata are excluded from predictive field values.

`model.SharedSCRIModel` accepts the existing `SharedAnnotationModel` or
`SharedPhysicalModel` by identity and uses the same payload, TokenGT backbone,
and readout convention for transition and spatial prediction. It adds a fixed
native output axis with explicit Gaussian, Poisson-count, or per-feature
categorical likelihoods. Masks retain observed-target denominators; all-missing
supervision is connected zero with denominator zero. Checkpoint preflight checks
all configuration/axis/tensor bindings before copying any shared weight.

The full owner currently fixes the orthogonal node-identifier capacity at two.
The spatial bridge therefore accepts at most two cells with that owner. Larger
neighborhoods require an explicitly constructed and qualified larger identifier
capacity, held fixed across the width family. The bridge never substitutes
nonorthogonal identifiers, truncates a neighborhood to fit, or changes an
existing backbone. The data-side builder has a separate ceiling of 128 cells.
This capacity restriction is a material integration gate, not a claim of
large-neighborhood support on the current full owner.

The shared payload retains its existing lossy positional pooling. Complete
conditions are subject to a 65,536-byte default bound and existing token bounds;
the wrapper does not silently truncate a full gene axis. Provider-native sparse
or array inputs and full-scale contexts need their own existing or newly
qualified native route. In particular, scCello's retained ranked values are not
raw counts and must not be passed to the Poisson objective as if they were.

`localization.Region` requires a native coordinate frame and units. A candidate
containing an unsupported or off-region action is rejected in its entirety;
silently clipping it would change the proposed intervention. The finite action
vocabulary and budget must come from the task, not be inferred from a desired
outcome.

`localization_metrics` reports measured absolute change inside and outside the
region, with actual observed/missing denominators. Region membership is defined
by baseline coordinates. It requires the same longitudinal IDs, explicit
lineage/donor/experiment/time evidence, compatible units, and one evidence class.
The complete metric cohort must also share study, experiment, donor, specimens,
organism, native time interval, and intervention context; unrelated cohorts
must be evaluated separately. Duplicate IDs cannot be collapsed. Missing outside-region measurements and a
zero total change produce an undefined ratio, not a claim of perfect
localization. These metrics do not establish clinical or causal validation.

## tasmorpho compatibility

The adapter reads the exact CSV interface at
`ab09864214ca36787a2144bb89425c8b9610bd6f` without importing the upstream scripts
or loading dill/pickle artifacts. Other revisions require a new schema review.
It verifies the supplied artifact SHA256, exact schema, seven cases, finite
values, zero control, energy-component sums, and positive-definite quadratic
cost before deriving `Q`, `R`, and `K` with explicit units.

The committed 869-byte summary has SHA256
`657aec89b9f828c355dc7af968a38bd3e1332accdc18f71d512b9d0e18552734`.
Its seven cases share one simulation-run partition. They are aggregate
simulation outputs, not independent biological replicates, cell graphs,
experimentally measured membrane voltages, or morphology targets. Cost is a
normalized squared-flux time integral, not joules; response coordinates mix
voltage with a native gap-junction proxy. No coordinate-dependent eigenvector
is exported as a validated morphology axis.

A separate parser accepts the exact five-row count reconstruction emitted by
upstream script 03. That CSV is not committed upstream and was not generated
here. Primary experimental provenance and rights remain unverified. Conditional
single-headed and double-headed challenge denominators remain separate, and
nigericin/monensin challenge predictions cannot enter this observed-count
schema.

The README's expansion of `8OH` conflicts with the primary planarian literature,
which identifies 8-OH as 1-octanol. The adapter retains the opaque source label
and an explicit unresolved mapping; it does not alter upstream development or
silently assign a drug ontology identifier. Primary evidence:
https://pmc.ncbi.nlm.nih.gov/articles/PMC2823934/ and
https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973/.

## Sweep plan and release gates

`readiness.build_sweep_manifest` expands the existing registered user plan:

* 24 primary proxy trials: 3 learning rates, 2 readout multipliers,
  2 attention-key multipliers and 2 fixed seeds.
* 8 standard-parameterization proxy comparisons.
* 9 transfer runs: three seeds at each of two intermediate widths and the target.
* 3 naive standard-parameterization target comparisons.

The 44 runs have a combined **22 GPU-hour upper budget**, not a measured duration
estimate. The full training run is separate. Primary trials retain the original
5,000-update/1,800-second caps; transfer trials retain their original
1,000-update/1,800-second caps. Target width values, actual all-head parameter
counts, baseline details, transfer seeds, and selected hyperparameters are not
invented by the preparatory manifest.

The complete architecture must pass its native finite/infinite-axis and alias
ledger checks, coordinate checks for every active operator, optimizer/recovery
qualification, complete validation protocol, source/corpus/model admission, and
exclusive bounded hardware scheduling. Selecting proxy hyperparameters from
test results or retuning at the target under a zero-shot label is prohibited.

The preflight is an evidence-preparation check and always declines to confer
training permission itself. It reads current ledgers and the actual primary
owner source without importing or executing that owner. In the inspected
revision, `SourceBoundOwner.require_training_admission()` raises unconditionally.
No extension flag or JSON label bypasses that refusal. Only a properly reviewed
native owner implementation and completed admission evidence can clear it.

## Commands

Run these from the isolated clone with the already provisioned interpreter.
Set `MORA3_ACTIVE_ROOT` to the reviewed checkout and `TASMORPHO_SUMMARY_CSV` to a separately obtained pinned source file. Use new output names for each invocation.

```bash
python scripts/scri.py audit \
  --active-root "$MORA3_ACTIVE_ROOT" \
  --inventory --output outputs/scri-20261108/active-audit-next.json

python scripts/scri.py catalog \
  --catalog configs/scri_sources.json \
  --output outputs/scri-20261108/catalog-next.json

python scripts/scri.py plan \
  --active-root "$MORA3_ACTIVE_ROOT" \
  --catalog configs/scri_sources.json \
  --output outputs/scri-20261108/sweep-plan-next.json

python scripts/scri.py preflight \
  --active-root "$MORA3_ACTIVE_ROOT" \
  --catalog configs/scri_sources.json \
  --tasmorpho-revision ab09864214ca36787a2144bb89425c8b9610bd6f \
  --output outputs/scri-20261108/preflight-next.json

python scripts/scri.py tasmorpho-summary \
  --input "$TASMORPHO_SUMMARY_CSV" \
  --source-commit ab09864214ca36787a2144bb89425c8b9610bd6f \
  --run-id tasmorpho-committed-example-ab098642 \
  --expected-sha256 657aec89b9f828c355dc7af968a38bd3e1332accdc18f71d512b9d0e18552734 \
  --output outputs/scri-20261108/tasmorpho-summary-next.json
```

Blocked preflight exits with status 2. The CLI has no training-launch command.
This prevents an audit or catalog operation from being mistaken for permission
to use the GPU or consume the active worker's resources.

## Owner handoff

1. Source owner: freeze the original releases, rights, native units and feature
   axes; verify pairing and metadata; distinguish acquisition from admission.
2. Construction owner: curate complete source components, reconcile identity
   graphs and official splits, prove original-to-task conservation, and expose
   native task denominators without inventing unavailable modalities.
3. Model owner: reconcile the additive shared objectives with current native
   routes, bind all heads and finite axes, and validate every parameter/alias role
   and target-free input policy against the live owner.
4. Campaign owner: obtain distinct matching readiness signatures, qualify
   operator/optimizer/recovery/runtime evidence and complete width profiles,
   finish baseline and transfer run bindings, then use the existing native
   campaign runner and validation-only selection policy.

The active owner's progress files and source modules are not updated by this
extension. A successful tiny fixture test establishes software behavior only;
it is not data readiness, scaling validation, a trained model, or held-out
biological performance.

## Method references

* Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
  Hyperparameter Transfer*, https://arxiv.org/abs/2203.03466.
* Official μP implementation and coordinate-check guidance,
  https://github.com/microsoft/mup.
* Pinned tasmorpho source,
  https://github.com/amelie-iska/tasmorpho/tree/ab09864214ca36787a2144bb89425c8b9610bd6f.

Dataset-specific original sources and unresolved rights are recorded in the
source catalog. A code repository's license is not assumed to license the
associated dataset, paper, or third-party measurements.
