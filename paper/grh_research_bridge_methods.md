# GRH-research integration with Mora-3 — 13 November 2026

## What is actually combined

The independent GRH-research project is pinned at `dbd954eede479a5129831e4e7c30537389ae973d`. Its [research atlas](https://amelie-iska.github.io/GRH-research/) distinguishes mathematical results, open analytic obligations, finite numerical references and saved learned-checkpoint experiments. Mora-3 does not reinterpret those scoped results as a complete GRH proof, new biomedical evidence, or its own completed training.

The numerical bridge is executable code in `src/mora3/data/grh_research_traces.py`. It imports complete original computation DAGs, preserving source bytes, tensor dimensions/axes/dtypes, complex parts, original dependency edges and provenance categories. It feeds the existing `ReasoningTrace`, `prefix_record` and `prefix_native_graph` interfaces rather than creating a separate incompatible reasoning representation.

There are two explicit views. The complete known-program execution schedule retains every source node and records shared dependency edges as fact dependencies, without counting repeated ancestors as independent evidence. The operation-specific view retains the selected operation's entire ancestor DAG and excludes unrelated branches and output heads. Its native predictor context stops before the requested output. Neither view is a claim that a deterministic computation DAG is a searched tree, an intervention graph, or an on-policy GFlowNet sample.

All imported references are conservatively test-only, grouped by source repository revision and checkpoint when applicable. A saved source-heldout example cannot be promoted to training merely by calling the importer. Source-generated numerical data remain `computed`, including deterministic replay of trained weights. Scalar arrays are carried as exact typed JSON; physical units are not inferred from axis labels. Native finite payload/depth bounds refuse overflow instead of truncating a trajectory.

## Actual pinned references

| Original export | Source kind | Source DAG nodes | Complete native execution steps | Native facts |
|---|---|---:|---:|---:|
| photonic-0 | trained-checkpoint-replay | 6 | 3 | 7 |
| tropical-attention | fixed-weight-reference | 59 | 27 | 60 |
| causal-soft-moe | fixed-weight-reference | 66 | 33 | 67 |

The original manifest SHA256 is `f4d2e685a423cfb0bc99c29019f0edabb3b26157a50309e5b9706f953d1e4904`. Three listed exports are retained byte-for-byte as evaluation fixtures. The source manifest lists22 references; the other19 are not fabricated or described as imported. Source code and foreign checkpoint files are not executed or unpickled.

Original trace SHA256 values:

- photonic-0: `3c386adb2352ef78b58d46a6b879d3bc68b67f5d200dd6b9a4f41c52f6aade4a`.
- tropical-attention: `cca65cf0b295848cc55d893accf260705c3baf4a7a16d795d060eba343aa5869`.
- causal-soft-moe: `3f861e5df9d3491b207f0191e4b87644678846fe4b6143a9e4ed45f550289680`.

An independent masked max-plus calculation on the actual tropical export reproduced the full context with maximum absolute error0 and the complete tied-winner mask exactly, over batch/head/query/key/channel shape[1,2,7,7,3]. Illegal keys are excluded and empty queries produce zero context. This checks that specific numerical operator and export; it is not full-model training, empirical width-transfer qualification or a proof of source mathematical claims. Other source-declared checks are preserved as declarations rather than silently promoted to independently reproduced evidence.

The29 focused bridge tests include full dependency preservation, exact source integrity, tensor/complex/dtype validity, oversized-input refusal, future/target mutation invariance, ancestor-only sibling exclusion and a real differentiable call through the existing native geometry/TokenGT prefix representation. The native gradient check uses the small photonic prefix, not a claimed qualification of every full-context biological graph. No optimizer update is performed. The final combined native/source/GRH regression passed541 tests and60 subtests with no skips and zero optimizer-step attempts. This is a focused software/source suite, not a completed μTransfer campaign.

## Interactive page and manuscript figure

The [interactive source-prefix companion](../docs/project/grh-research-bridge.html) renders actual importer outputs. Select a source and completed-operation prefix, inspect exact visible tensor names/axes/dtypes, and separately reveal the retained reference answer. Revealing a reference in the browser does not change the predictor's inputs. The long timeline scrolls without changing the original number of operations. It is an execution-state timeline, not a physical causal diagram or learned forest visualization.

The same SVG renderer supplies `paper/figures/grh-native-prefix.svg` and `docs/project/figures/grh-native-prefix.svg`. Five actual installed-Chrome tests cover every prefix of all three selected targets, exact native fact visibility, reference hiding, pinned links, URL state, SVG export and mobile/keyboard behavior. These browser tests perform no network inference or training.

## Separate biological GRH source boundary

The GRH-Mora-ConformerResponse biological auxiliary source remains separately pinned to HF revision `53a6b66acaa69ef79584648f3f6dc068363c4837`, manifest SHA256 `3610fcb330ee2b2b89037a38a39bf8c0509190b52f61a92313422f07ea9c5cad`. It contains22,951 static computed rows,266 source anchors,241 electronic trajectories and12 nuclear trajectories. Original experimental/computed/simulated labels, units, atom order, masks and source-split lineage are not interchangeable with the numerical architecture references above.

Already retained independent evidence verifies the22,067 static training MMFF94 rows,84 electronic training trajectories and4 nuclear training trajectories. It also records important limits:21 of74 MMFF94 source-anchor rows failed graph reconstruction; four original static validation rows are incident to current native test identities;21,474 GRH train coordinates are withheld in an inactive cross-corpus candidate mask;1454 native train direct overlap coordinates require full group closure. None of those inactive masks becomes authoritative admission through this integration.

The previous zero-cross-role-key scan of the80,656,029 registered Mora task views did not include every unregistered GRH/source relation. Therefore these GRH conflicts cannot be dismissed by citing that earlier scan. Full computed-auxiliary objective enrollment, true cross-corpus closure, source rights and distinct same-snapshot construction/model acceptance remain required. Experimental-only target rules are not weakened and computed values are not relabeled as observations.

## Publication, training and reproducibility

The source-only integration preserves all reviewed older worktrees and private active planning. Source publishing is distinct from dataset publishing and training. A hash-verified Git reproduction bundle may accompany the GRH dataset as a source-code supplement; that bundle is not a new set of training examples and does not modify original data shards or their revision pins. The private Mora training repository remains a partial FreeSolv release until additional payload components are independently cleared and verified.

The authorized26-run maximum remains24 native μP proxy slots plus at most two approximately80M targets. This integration has started zero substantive training runs. The full eligible data/model/operator/recovery/resource admission remains incomplete, and no global biological leakage theorem, chosen proxy winner or trained target performance is asserted.

Primary source: [GRH-research numerical trace manifest at the exact revision](https://github.com/amelie-iska/GRH-research/blob/dbd954eede479a5129831e4e7c30537389ae973d/web/data/architecture/manifest.json). All new conversion/validation logic is a Mora-3 adaptation, not an undocumented claim attributed to that source.
