# Mora-3

Mora-3 is a research implementation for multimodal modeling across molecular structures, sequences, cellular profiles, imaging and source-grounded text. The architecture uses a shared TokenGT backbone with native task heads and modality-specific generation objectives. The training code includes source-aware loaders, explicit role and observation checks, and a width-transfer workflow using μP.

This repository is a source snapshot. Dataset payloads, checkpoints, private environments and experiment logs are distributed separately when release and training gates are met. Some source-specific checks require separately provisioned source artifacts. The affinity feedback component provides teacher-specific electronic interaction targets alongside distinct experimental affinity endpoints. Production complex-data enrollment and benchmark results are pending.

[Interactive research site](https://amelie-iska.github.io/Mora-3/) · [Local checkpoint playground](tools/playground/README.md) · [SCRI and tasmorpho extension](docs/scri_extension.md)

## Repository layout

- `src/mora3/` contains model, data, geometry, reasoning and research code.
- `scripts/` contains data intake and workflow scripts. Source-specific scripts may need the corresponding source manifests and local resources.
- `tests/` contains source-independent tests and tests that also require local artifacts.
- `environment.yaml` describes the project environment.

Use `PYTHONPATH=src` when running the source tree directly.

## Complete-forest reasoning and current campaign

The [interactive flow-forest explorer](https://amelie-iska.github.io/Mora-3/project/flow-forest.html)
connects the proposed flow head, whole-trajectory analogy retrieval, directed topology,
typed final decoding, few-step flow maps and context-dependent quality adaptation.
The [live paper](https://amelie-iska.github.io/Mora-3/paper.html#flow-forest-methods)
embeds the interactive explorer and eight matching vector figures.
The [working manuscript](paper/mora3_preprint.md),
[methods supplement](paper/flow_forest_methods.md), and
[iterative implementation plan](planning/flow_forest_extension_20261111.md)
distinguish implemented components, proposed neural extensions, mathematical examples
and empirical results. No trained quality or biomedical validity is inferred from a diagram.

The governing allocation is **24 native μP proxy trials plus two approximately80M
full-model runs**, with validation-only hyperparameter selection and no fresh retries.
See [the current directive](planning/current_mup_campaign.json) and
[the native transfer handoff](docs/campaign26_transfer.md). Source/model admission,
all-head width/alias qualification, recovery and measured resource evidence remain
required. Additional distillation, preference adaptation and trained QLoRA comparisons
have no allocation under the unchanged26-run cap.

Dataset payloads, original private specifications, credentials, environments, model
weights and private progress logs are not included in this source integration.
Local artifact-dependent tests are not expected to pass without their pinned inputs.


## Vectorized topology and exact retention

The [persistence laboratory](docs/project/persistence-vectors.html) and matching paper figure demonstrate batched landscapes, Gaussian-integrated images, explicit L2 metrics and sampling counterexamples. The [numerical contract](planning/persistence_vectorization_20261111.md) distinguishes complete finite landscapes from lossy fixed-grid features and preserves original diagrams and directed evidence. Native topology-memory retrieval uses exact finite-landscape L2 by default; structural certification and training admission remain separate.


## Native forest-head implementation — 13 November 2026

The optional complete-forest velocity, analogy, token decoder and quality heads
are implemented and source-bound in the native owner. The concrete velocity field
and subtree editor connect to multi-output FoT refinement. The learned decoder
uses a frozen versioned codebook,40 appended controls and explicit incomplete-output
refusals. All modules remain untrained by this implementation package.

[Interactive native-head computations](docs/project/native-forest-heads.html) ·
[Methods and qualification limits](paper/native_forest_heads_methods.md) ·
[Implementation contract](planning/native_forest_heads_20261113.md).

Full source/model release, actual24+2 trial results and checkpoint-based held-out
evaluation are still required. Software test success is not empirical μTransfer
or biological-quality evidence.


## GRH-research connection

The [GRH research atlas](https://amelie-iska.github.io/GRH-research/) is linked through a
[source-pinned complete numerical-trace bridge](docs/grh_research_integration_20261113.md),
with an [interactive native-prefix companion](docs/project/grh-research-bridge.html).
Original numerical references remain evaluation-only and distinct from the GRH
computed molecular dataset, whose cross-source admission is still incomplete.
No new Mora training or GRH proof is implied.
