# Mora-3

Mora-3 is a research implementation for multimodal modeling across molecular structures, sequences, cellular profiles, imaging and source-grounded text. The architecture uses a shared TokenGT backbone with native task heads and modality-specific generation objectives. The training code includes source-aware loaders, explicit role and observation checks, and a width-transfer workflow using μP.

This repository is a source snapshot. Dataset payloads, checkpoints, private environments and experiment logs are distributed separately when release and training gates are met. Some source-specific checks require separately provisioned source artifacts. The affinity feedback component provides teacher-specific electronic interaction targets alongside distinct experimental affinity endpoints. Production complex-data enrollment and benchmark results are pending.

[Interactive research site](https://mora-3-research.wolvieworld.chatgpt.site)

## Repository layout

- `src/mora3/` contains model, data, geometry, reasoning and research code.
- `scripts/` contains data intake and workflow scripts. Source-specific scripts may need the corresponding source manifests and local resources.
- `tests/` contains source-independent tests and tests that also require local artifacts.
- `environment.yaml` describes the project environment.

Use `PYTHONPATH=src` when running the source tree directly.
