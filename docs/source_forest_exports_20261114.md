# Complete source-forest exports and explicit payload capacity

Status: implemented research component, locally verified on Iska,13 November2026
Pacific (14 NovemberUTC). No training run or public push is established here.

## What this implements

`mora3.data.forest_source_exports` compiles an original complete `ReasoningTrace`
into the existing native forest supervision format. It retains every supported
tree node and branch, original lexical fact JSON, source revision/ordinal/groups,
direction category and original train/validation/test role. Every latent is computed
from its entire root-to-node native graph prefix. Future and sibling facts do not
enter that prefix; targets/evaluation metadata remain outside predictor inputs.
Merged DAG histories, nontrivial relation squares, censored or invalid branches
refuse rather than being silently flattened, duplicated, or discarded. This is a
known-source-tree reconstruction contract, not an unknown-topology policy.

`FrozenForestSourceEncoder` clones the supplied native GeometryConditioner and
extended AnnotationPayloadEncoder without freezing or altering the live owner.
Weights, buffers, alias-sharing configuration, actual token axis, implementation
hashes and runtime versions bind the export. Source coordinates and decoder
codebook rows come from those exact native modules. Neither a random codebook nor
old metadata rows can masquerade as the40 appended forest controls.

`ForestSourceExport` and `load_source_codebook` provide plain-JSON/tensor-byte
save/readback without pickle, external code execution or overwrite. All tensor
bytes and original scalar lexical representations are preserved. The source
encoder itself is lossy: retaining export bytes does not make latent embeddings
invertible or validate scientific reasoning. A separate retained qualified encoder
checkpoint remains necessary for production regeneration.

The train-batch adapter connects to the implemented joint flow/token objective
and finite native route. It only accepts original train exports and exact native
encoder/codebook/source/group identities. Quality labels and analogy pair labels
remain absent unless independently supplied; they are not fabricated here.
A real-source export from an explicitly untrained diagnostic encoder cannot be
used as production training supervision. Full-owner admission remains separate.

## The interrupted capacity gap is fixed

The default GeometryConditioner per-entity payload bound remains8192. Existing
legacy and native-v2 configuration wires, parameter counts and tensor equations
are unchanged. A nondefault explicit `payload_max_tokens` requires applied native
muP and uses version3 (`complete-explicit-token-cap-v1`). Checkpoint configuration
mismatches refuse before parameter copying. Source-prefix identifier rank is a
separate capacity and is not expanded by this payload option.

The retained real source census measured maximum entity streams74561 and81761.
The complete diagnostic uses131072 per entity and40000000 aggregate prefix-token
work, within an explicit64M exporter envelope. Default aggregate capacity remains
2M. No token truncation, source role reassignment, normalization change, fabricated
signature or resource/admission bypass is used. Aggregate prefix work counts repeated
serialization across full history inputs; it is NOT a32M-token attention context.
These settings apply to the separately frozen CPU reference encoder, not an
implicitly changed full-model source factory or frozen training campaign.

## Actual retained-source export

All three original references are pinned to GRH-research revision
`dbd954eede479a5129831e4e7c30537389ae973d` and manifest
`f4d2e685a423cfb0bc99c29019f0edabb3b26157a50309e5b9706f953d1e4904`.
All remain computed, evaluation-only (`test`) references. Their given execution
schedules are not claimed to be sampled branching reasoning searches.

| Retained reference | Original dataflow nodes | Complete prefix states | Aggregate prefix tokens | Observed seconds |
|---|---:|---:|---:|---:|
| photonic-0 | 6 | 4 | 35,153 | 0.055 |
| tropical-attention | 59 | 28 | 20,702,036 | 17.944 |
| causal-soft-moe | 66 | 34 | 32,596,980 | 27.602 |

Every export has exact independent disk readback,8 FP64 coordinates per prefix,
and preserved original file hashes. The untrained width8 diagnostic used one CPU
thread and peaked at692191232 bytes of cumulative process RSS. This includes
runtime/import overhead, not isolated model memory, training memory or an80M
GPU qualification. Source programs were not executed and no optimizer ran.

The first earlier run correctly refused the two larger records under its8192-token
bound. Those refusals, the measured token census and subsequent complete export
are all retained; the failure history has not been replaced with a success label.

## Verification scope

The isolated final zero-update suite passed271 tests, with no skips or optimizer
step attempts. It covers40 source-export cases,18 new capacity/CLI cases, existing
native heads/owner/trajectory integration and CPU geometry parameter/alias checks.
FP32/FP64 direct loss/gradient references, complete branch visibility, source/axis
corruption refusals, fresh-process asset loading, default/extended capacity output
and gradient equality, and pre-copy checkpoint refusal are included.

An initial expanded verification command mistakenly selected14 legacy recovery
tests which explicitly call Adam.step. The guard refused all14 before any update;
that command's failure and38 skipped cases remain in its receipt. The final scope
excludes update-bearing recovery tests rather than mocking an optimizer or weakening
a production check, and uses the active original source root so all selected native
owner checks actually run. This is not a full repository or completed-update
recovery suite. CUDA and substantive training remain separately unqualified.

The offline [source-export inspector](project/source-forest-exports.html) draws
actual prefix coordinates, not invented reasoning-quality scores. Its complete
8-coordinate table and source digests accompany the lossy2D projection. Browser
checks cover all three sources at1360px and390px, prefix visibility, exact decimal
coordinates, axes, URL reload/back/forward, keyboard operation, evidence export,
invalid URL recovery, zero page errors and no horizontal overflow. A matching
[vector figure](project/figures/source-forest-exports.svg) is included in the paper.

## Remaining campaign dependencies

This package does not fill missing real train-source trajectories, trained encoder
exports, quality criterion labels or supported analogy correspondences. Large token
payloads must still satisfy the native decoder's explicit4096-slot bound; this change
does not silently expand its learned slot axis or drop token supervision. The full
source/model counts and context qualification must include whatever final task
representation is chosen. Normalized unknown-topology generation, hybrid GFN/flow
teacher–student training, complete eligible-source/group/rights qualification and
genuine distinct construction/model results remain separate unfinished work.

The maximum remains24 counted proxy trials plus at most two approximately80M
full-model runs. No training or push is authorized by these test results. The latest user instruction authorizes reviewed source publication at progress
checkpoints independently of training start. This supersedes the earlier
publication delay, not the unchanged requirements for qualified training.
