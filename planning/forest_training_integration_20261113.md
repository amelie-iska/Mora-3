# Native complete-forest training integration — 13 November 2026

One coherent implementation: connect frozen source-aligned forest supervision, all four existing native heads and the checkpointed native MuAdamW engine. No source-master/owner ledger or training admission is changed.

Use fixed source-known skeletons only. Flow targets remain supervision. Decoder reconstruction uses a separate detached source-encoder export; quality uses separate complete candidate states, not the flow targets by assumption. Analogy queries are condition-only and require explicit train-only disjoint positive/negative correspondence labels. Unknown labels remain masks; inactive objectives do not renormalize fixed mixture coefficients.

Preflight every enabled objective before RNG/neural calls. Freeze source/split/group/encoder/codebook/skeleton/quality-policy and route contents. One finite ordered route provides exact source ordinals and immutable cursors. The engine owns noise/time RNG; checkpoint v6 preserves the new optional binding without changing legacy formats. Failures before optimizer execution restore random state; an optimizer-phase failure is not claimed rolled back. Existing full-owner admission precedes any source-window read.

Tests first: four-loss equations/all-head shared gradients; target exclusion; missing labels; wrong split/group/source/coordinate/codebook/policy; all-input preflight; cursor conservation/exhaustion; immutable route/config; native checkpoint replay. Zero optimizer steps. Full source preparation, normalized unknown topology, final owner/operator/runtime qualification and actual24+2training remain separate work.

Gaussian conditional flow matching is reused from the existing field (Lipman et al., arXiv:2210.02747v2). The joint objective and native scheduler integration are Mora-3 adaptations. No new packages, workers, credentials or services.
