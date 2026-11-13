# Mora-3: flow-generated reasoning forests and quality-aware adaptation

**Design revision FRF-00 · 11 November 2026 · private research plan.**

This document specifies the user's five additions and the linked interactive/print figures before production implementation. It is not a training report. The initial deliverable is the documentation, mathematical reference explorer and publication-ready vector figure set. Neural components below are proposed unless a later immutable implementation receipt names their native source, tests and measured behavior.

The authoritative local inspection began at 2026-11-11T12:08:44Z: active main `255d986e8938c90123e0050f538748e547719d97`; isolated implementation `98c75212cfd708a9b3a07cb7f0f1e19df8a3c6aa`. The latter contains actual bounded source-exposure/group qualifications added after the previous transfer package. Those source qualifications are not completed training. Complete-expanded-dataset readiness was false; distinct construction/model attestations were absent; the native training owner still refused admission. Existing work on UMA node1 inverse rotation/aggregation, all source families, SCRI, tasmorpho and trajectory integration is preserved.

## 1. Governing experiment boundary

The current authorization is **24 native μP proxy trials plus two approximately80M full-model runs, 26 substantive runs in total**. The current directive pointer and historical budgets are not rewritten by this design. Neither failed starts nor new training phases disappear from accounting because they use an old checkpoint.

The user also requests later few-step distillation, GraphCG quality exploration, a preference-adaptation run and a QLoRA comparison. A genuine new optimizer run for each is additional substantive training. These stages are designed now but remain `budget_blocked` until the user explicitly revises the global cap or approves a different complete allocation. They must not consume proxy slots, masquerade as diagnostics, be called a resumption of completed training, or run as uncounted “fine-tuning.” Non-updating evaluations at 1/2/4 solver steps may share an existing checkpoint.

The new head, vocabulary, retrieval semantics and initial objective mixture change the architecture. They must be qualified and frozen before the first applicable proxy trial. If any real campaign has already begun when a later worker reads this document, it must preserve that campaign unchanged and stage the addition separately. Never claim old μTransfer results qualify newly added operators or objectives.

## 2. Scientific object: a complete, typed forest

Let a generated forest be F=(T,Z,A), with discrete ordered topology T, continuous latent node/token-slot values Z, and typed action/evidence attributes A. Context c contains only source-available conditions and explicitly desired outcomes. A construction history h records which prefixes have actually been executed. A forest is not a bag of endpoint embeddings, and a root shared by four leaves is one state, not four independently sampled observations.

A fixed-dimensional ODE cannot by itself choose an arbitrary number of nodes, parents, branches or terminal events. The first native implementation will condition a continuous flow on an explicitly declared skeleton. A separately normalized topology/length/action policy is required for unknown topology. Parent indices must refer to existing nodes; generated topology must not read the held-out target skeleton as conditioning. Canonical construction order or a fully specified backward kernel must account for the multiplicity of histories that produce the same forest.

Two independent notions of time are retained: **reasoning depth d** indexes computational dependencies; **flow-solver time s in [0,1]** indexes transport from base noise to the entire forest embedding. A one-step flow map may output many reasoning nodes. It does not make a long argument a one-step argument.

Known conditioning nodes are clamped. Source/target/desired/evaluation roles and missing coordinates have separate masks. A noisy target is a legitimate input to a denoising training task, but is never relabeled a clean observed condition. Every branch and censored leaf is retained. A depth, memory or search-limit termination is not a completed solution.

Each edge carries a source-supported direction class: computational, temporal, observational or interventional. Procedural precedence is not biological causation. An intervention artifact must describe actual intervention semantics; a digest only binds that artifact. No automatic reverse edge, commutative square, biological mechanism or state-equivalence merge is inferred from embedding proximity. AND-combining results from two branches requires an explicit valid execution history, not an implicit union of incompatible branch-local facts.

## 3. Shared conditional flow head

Reuse the actual native TokenGT condition encoder, ConvexTok payload axis, graph identifiers and configured attention operator. The new velocity head consumes a noisy complete latent forest, solver time, known skeleton/masks, biological context and eligible retrieved exemplars. It must expose its parameter aliases and μP roles, fixed versus width-scaled axes, readout scaling, optimizer ownership and complete checkpoint state. Count every new parameter in all three widths and the actual approximately80M target; do not retain a historical parameter count after adding the head.

For a fixed skeleton and available continuous coordinates, a baseline straight conditional path is

    Z_s = (1-s) epsilon + s Z_1,
    u_s = Z_1 - epsilon,
    L_FM = E[ sum_available ||v_theta(Z_s,s,c,T)-u_s||^2 / count_available ].

The existing Gaussian path with nonzero sigma_min is a separate named variant and retains its residual terminal noise. Frozen/clamped coordinates are not optimized; unavailable targets are not filled with zero. A frozen, source-bound encoder or a separately specified joint alignment objective is needed to prevent a moving latent coordinate system from invalidating targets, codebooks and retrieval indices.

The native tropical operator is preserved. Ordinary attention is not substituted under its name. The current project has stricter numerical/μP qualifications for ordinary versus tropical attention; attaching a tested head does not automatically establish an extended tropical width family. Long graph/context support remains bounded by actual identifier rank, attention cost and representative device measurements.

### Domain geometry

The representation distinguishes physical geometry from generated reasoning coordinates. Molecules/proteins require atom/residue identity, coordinate units, frames and symmetry-aware transforms; pairwise distances do not distinguish enantiomers, so ordered stereochemical/chiral constraints remain separate. Torsions use periodic geometry. Cells preserve physical scale or explicit pixel units, morphology, donor/cell-type composition and spatial context. Gene-regulatory networks use directed, typed relations and intervention evidence, not invented Cartesian positions. An ODE on a Euclidean reasoning latent is not automatically a normalized flow on SE(3), an angular torus or a discrete molecular graph.

## 4. GFlowNet feedback without a false probability model

The desired distribution is a measure on complete forests, not merely a scalar best-answer objective:

    pi*(F|c) = R(F,c) / integral R(F',c) dmu(F').

The hybrid reference measure includes counting mass over valid finite topologies/actions and a declared continuous base measure per skeleton. One conservative integrable reward is R=q0(T,Z,A|c) exp[-E(F,c)] with E>=0 and normalized q0. Hard-infeasible outcomes may receive zero mass, but then support, exploration and log-reward handling need explicit policy; do not silently clip zeros into an undocumented reward floor.

**Initial integration choice:** use a separately qualified stochastic native GFlowNet to produce source-bound forest examples, then distill that distribution into the conditional flow. This separates exact transition probability accounting from ODE sampling. Finite-support tests can verify mass ratios; actual reward-proportional learning still requires empirical distribution, coverage and calibration evidence. Training a teacher on a reward does not prove the teacher samples its target exactly.

For prefix-tree terminal objects, the existing complete-path objective uses deterministic prefix deletion as its backward construction. If the terminal object is instead an unordered forest equivalence class with multiple constructions, reuse of that backward rule without multiplicity correction changes the target distribution. The design must specify which object is sampled. Whole-forest rewards are not multiplied independently over root-to-leaf paths when those paths share ancestors.

A deterministic flow ODE does not supply Gaussian transition log probabilities. Exact flow density would require a justified change-of-variables/divergence computation under the actual solver and support. Stochastic flow-policy training, such as Flow-GRPO, is a distinct option with explicit SDE kernels and normalized probabilities, not proof of GFN reward proportionality. Negative-aware flow fine-tuning is another separately named objective. FoT beam search, top-k pruning and rejection change the proposal distribution and must not masquerade as unmodified on-policy samples.

All caches and replay items bind code, context, source, skeleton, reward, teacher, solver, RNG and encoder/codebook revisions. Sampler generator state and source cursor join optimizer/model state in tested fresh-process recovery. No replay from a changed reward or moving latent gauge is accepted implicitly.

## 5. Whole-forest analogical retrieval and training

A dedicated query/readout may learn a problem-conditioned retrieval key. The first implementation should share existing native features and minimize new learned parameters, but automatic model discovery of correspondence is a separate requirement from the existing supplied-functor verifier.

The retrieval index contains train-only source problems and complete reasoning forests, with source rights, native evidence classes, original splits, biological group closure and encoder revision. Query features use only currently available conditions. Held-out answers, target-derived topology, reward labels and unavailable measurements cannot enter the query or shortlist key. Refit an index only in a new versioned namespace when the encoder changes.

Use approximate retrieval for a bounded shortlist, then evaluate complete directed relational mappings, intermediate subgoals and branch coverage. Report approximate recall against an exact reference, correspondence accuracy, query latency and abstention coverage. The exact reference and candidate sizes must be scientifically representative; a two-concept toy is not whole-trajectory support.

Training pairs preserve same-problem correspondence where genuinely available, while hard negatives include deceptively similar endpoints with incompatible action order, missing prerequisites, different cell/assay context or absent defining relations. Unknown analogies are not fabricated negatives. Partial analogies map explicit subpresentations, with uncovered branches recorded. A source step may map to a target subtrajectory rather than one target vector.

Persistent H0/H1 descriptors can aid retrieval and geometry regularization, but equal barcodes do not imply isometry, identical reasoning, causal transfer or semantic equivalence. Grandis's finite directed path presentations and positive-chain information supply complementary order-sensitive checks. The direction of a path is not the orientation of a physical coordinate frame. No continuum reconstruction theorem is claimed.

## 6. Final decoding and finite typed vocabulary

The existing payload pooling is lossy and does not establish an invertible token representation. Therefore the final decoder requires explicit source-aligned token slots, length/role conditioning, a learned latent-to-token projection and a versioned codebook. Nearest-codebook projection is only an initialization or local refinement rule. A constrained decoder must enforce forest serialization, node references, evidence types and modality grammars without silently discarding invalid branches.

A joint training objective may combine native reconstruction/conditional token loss, commitment or round-trip loss, geometry/direction preservation and supported modality constraints. Discrete nearest projection needs an explicit straight-through/surrogate policy if used in training; it is not differentiable simply because the upstream flow is. Frozen versus jointly learned codebooks are distinct experimental settings.

Decoding is many-to-one. The probability of an output token forest is the latent probability integrated over its decoding region. Thus a reward-proportional latent distribution does not automatically remain reward-proportional after rounding, repair or constrained projection. Measure pre/post-decoding validity, reward and group diversity, token error, structural edit distance, branch preservation, abstention and coverage. An output reasoning forest is an inspectable generated artifact, not evidence that it is a uniquely faithful explanation of the model's internal computation.

### Proposed finite controls: 40 symbols, no IDs allocated yet

| Family | Proposed symbols (each prefixed MORA and bracketed in the registry) |
|---|---|
| Structure | FOREST_BEGIN, FOREST_END, TREE_BEGIN, TREE_END, NODE_BEGIN, NODE_END, EDGE_BEGIN, EDGE_END |
| Actions | PARENT, BRANCH, JOIN, SUBGOAL, PROPOSE, VERIFY, REVISE, FINISH |
| Evidence | CONDITION, DESIRED, PREDICTED, SOURCE_OBSERVED, SOURCE_COMPUTED, SIMULATED, HYPOTHESIS, UNKNOWN |
| Relations | ANALOGY, MAPPING, RELATION, CONSTRAINT, UNIT, FRAME, MISSING, ABSTAIN |
| Decisions | SUPPORTED, UNSUPPORTED, PASS, FAIL, UNCERTAIN, CONFLICT, INFEASIBLE, BUDGET_EXHAUSTED |

These are a Mora-3 design proposal inspired by typed output interfaces, **not a recovered Jev vocabulary**. TypeSafe's public Jev documentation establishes predefined typed probabilistic outputs and parallel generation; it does not publish this special-token list. A SOURCE_OBSERVED token never itself certifies source evidence.

Reserve these controls outside ConvexTok's learned byte-substring budget; retain all256 byte tokens. Version vocabulary, embedding/output rows, metadata offsets, finite-axis identities, token masks and serialized checkpoints together. The current PayloadEncoder metadata namespace depends on tokenizer size; blindly appending controls would change offsets. Old checkpoints and vocabulary artifacts remain immutable and must not be auto-migrated. Ordinary output text and biological identifiers remain representable; a finite control list is not a finite list of all answers.

## 7. One-to-four-step flow maps

**Selected investigation:** DMF-informed conversion of a trained velocity head into an interval-conditioned flow map, with an initial native teacher-pair distillation path and named reference alternatives. Teacher trajectories and codebooks are frozen. A map M(x,s,t) must satisfy M(x,s,s)=x and approximately compose as M(M(x,s,t),t,u)=M(x,s,u). Learned average velocity permits large solver intervals; merely decreasing Euler steps is not distillation.

MeanFlow, Decoupled MeanFlow and Align Your Flow motivate this direction. Their image-model results do not establish biological forest quality. The inspected DMF implementation uses JVPs and particular attention kernels; Mora-3's first-order tropical kernels are not presumed JVP-compatible. The initial teacher-pair endpoint regression avoids importing that derivative path and is explicitly a local adaptation, not a faithful reproduction of the DMF training objective. A JVP-based variant needs independent derivative/tie tests before adoption.

Evaluate 1,2,4 and a reference larger-step setting on the same held-out problems, reporting full forest/task accuracy, topology and branch preservation, decoder validity, calibration, diversity and wall time. Count **all** velocity/map evaluations, guidance passes, topology-policy calls, retrieval and decoding costs; one solver step is not one total model call. Distillation must not collapse reward modes or erase rare legitimate modalities.

This is an additional training stage, blocked under the unchanged26-run authorization. No1–4-step quality or speedup is asserted by the analytic explorer.

## 8. GraphCG exploration and quality-aware adapter training

A learned GraphCG direction is not intrinsically “bad.” Its signed effect depends on task, scale, dose/context, modality and current latent location. The original project's numerical direction diagnostics do not prove established semantic factors. Explore matched conditions and bounded signed amplitudes without updates, retaining the unedited control and complete forest. Record changes in all applicable objectives, uncertainty and constraints; preserve legitimate rare modes and task-specific trade-offs.

Create pairs only when source-supported evidence and a predeclared evaluator distinguish the outcomes under the **same** problem/context. Store favored, disfavored, neutral, conflicting and unknown dispositions; not every pair has an ordering. Hard invalidity is separate from soft preference. Evaluator disagreement and unavailable measurements are not negative labels. Use condition/source/scaffold/protein-family/donor/study-aware splits and hold the final test set outside selection.

For a native stochastic policy with computable complete conditional log probability, an explicit pairwise preference objective may compare logp(F+|c)-logp(F-|c) against a frozen reference with trust-region/retention terms. An ODE velocity MSE is not logp and cannot be substituted into DPO. A flow-specific negative-aware objective is a separately documented candidate. Preserve calibrated uncertainty, constraint satisfaction, minority contexts and multimodal diversity rather than optimizing a single self-referential scalar.

### Adapter decision after current-source review

**Quality-first engineering candidate: BF16 DoRA; quantized candidate: QDoRA; comparison reference: QLoRA.** This is a conditional selection for native qualification, not a claim of universal superiority. At80M parameters, low-bit storage should be justified by actual end-to-end memory/throughput measurements rather than assumed necessary. QLoRA quantizes the frozen base; adapter training is not thereby all4-bit.

The current PEFT stable tag0.21.0 resolves to commit `9dc6fa2d2093853dec3889b790d4f251efe7fcac`. The inspected DoRA implementation separates magnitude/direction, handles row/column conventions and detaches the evolving norm in its forward derivative. Native aliases, biases, tied embeddings/readouts and finite biological output axes must be audited before replacing modules. Do not quantize sensitive density/reward reductions, geometry constraints or tropical winner comparisons by default. Revalidate width scaling, initialization, checkpoint loading and mixed-precision behavior for the adapter objective; base-model μTransfer is not a proof for arbitrary post-training changes.

Newer AQLoRA was screened, not adopted as the default: its mixed-layer protection trade-offs and evaluated model families do not establish a quality advantage for this native graph/tropical80M owner. LoRA-GA² (2608.19800v2) was withdrawn on 1 September 2026 and is excluded from implementation selection; its historical abstract is not accepted quality evidence. Gradient probes that update or train are not free diagnostics under the cap. Installed tokengt currently has neither PEFT nor bitsandbytes; no dependency installation or environment change is made by this plan.

### Multimodal quality evidence requirements

| Domain | Evaluate in its actual context | Do not treat as established by itself |
|---|---|---|
| Protein sequence/backbone | Residue identity/frame, stereochemistry and clashes; independent sequence–structure compatibility; relevant fold/assembly; experimentally supported stability/solubility or calibrated predictors; task function | High pLDDT, self-refolding agreement or low aligned RMSD as thermodynamic stability or function. Preserve legitimate disorder, flexible loops and membrane contexts. |
| Molecules and complexes | Valence/stereochemistry/protonation, geometric strain and pose context; native assay units; supported solubility, reactivity and ADMET uncertainty | A valid string as synthesizability, affinity, safety or correct protonation under every assay. |
| Cells, mixtures and GRNs | Cell type/donor/dose/time; baseline and control; composition versus within-cell changes; supported spatial/intercellular effects; batch/doublet artifacts and off-target context | Ligand/receptor co-expression as causal communication, a desired state as an observed state, or a rare cell type as an error. |
| Reactions/structures/TS | Exact identity/maps, conservation, constraints/frames, separate bound-pose and template requirements, source-supported transition-state evidence | An apparently plausible3D shape as a transition state, mechanism or catalytic outcome. |
| Images and segmentation | Source channel/scale, alignment, instance/compartment validity, image–mask consistency, held-out morphology measures | A convincing generated image plus its own segmentation as independent experimental response validation. |
| Reasoning and analogy | Complete prerequisites, valid directed mappings, all branch outcomes, grounded citations/units, task constraints, honest abstention | Fluent token forests, equal barcodes or a high model-derived reward as truth. |

No fixed universal clinical thresholds are introduced. Domain reviewers and qualified sources determine applicability and thresholds before labels are built. Model-generated annotations stay model-generated. No laboratory protocols or biological efficacy claims arise from these criteria.

## 9. Data preparation and acceptance

Required new artifacts are: complete source problem/forest records with rights and evidence classes; original group/split map; context-only query corpus; explicit analogy correspondences and uncertainty; versioned frozen latent/codebook exports; complete teacher trajectories and solver lineage; native preference/evaluator records; and all rejected/censored/missing cases. Actual rows, bytes and coverage must be counted from admitted sources, not generated from this specification. No new corpus has been acquired or admitted in FRF-00.

Tests before production code cover: schema/role leakage; branch enumeration and shared-prefix accounting; normalized finite topology kernels; density/reference-measure identities; noise/time/RNG replay; dimensional and μP role scaling; aliases/all-head counts; nearest-decoder grammar and versioning; current-source token roundtrips; geometry/periodic/chiral distinctions; directed-relation preservation; exact-vs-approximate retrieval recall; and fresh-process recovery. Real held-out performance and representative4090 resource evidence remain separate acceptance requirements.

## 10. Ordered implementation work packages

| ID | Bounded deliverable | Completion evidence |
|---|---|---|
| FRF-00 | This researched plan, updated manuscript methods, interactive mathematical explorer and matched vector figures | Equation/unit/browser/export checks; no learned-quality claim |
| FRF-01 | Complete forest skeleton/source contract and lossless native compiler | All branches and masks preserved; no hidden topology/target leakage; actual eligible source rows |
| FRF-02 | Native shared conditional velocity head | Independent equations/gradients, complete finite/width axes, alias ledger and head count |
| FRF-03 | Context-only learned analogy retrieval plus whole-path reranking | Group closure, exact retrieval reference, complete multi-branch transfer and abstention |
| FRF-04 | Versioned40-control vocabulary and token-slot refinement decoder | Byte roundtrip, role grammar, old-checkpoint refusal, output distribution/validity audit |
| FRF-05 | Hybrid forest GFN/flow distribution integration and task schedule | Explicit measure/backward semantics, reward support, current-source numerical tests and RNG/cursor recovery |
| FRF-06 | Existing REAL24 proxy campaign and two target runs | Qualified native source/model/family/resources;24 actual receipts; validation-only selection and actual transfer |
| FRF-07 | Teacher flow-map distillation and1/2/4-step evaluation | Revised run budget first; fixed teacher and matched held-out quality/diversity/latency |
| FRF-08 | GraphCG quality exploration and DoRA/QDoRA preference adaptation | Revised budget for training/comparison; matched-context labels, uncertainty and retention tests |
| FRF-09 | Final paper/site evidence reconciliation | All results traced to accepted manifests/checkpoints; no schematic shown as empirical data |

Every work package is one coherent task per invocation and is performed in an isolated namespace first. Source-owner and model-owner ledgers are not silently overwritten. The read-only readiness watch is unchanged. The existing source-qualification critical path remains mandatory; a feature request is not permission to train an incomplete corpus.

## 11. Visualization and manuscript contract

The local project-page explorer is `docs/project/flow-forest.html`. It is offline, dependency-free and paired with a tested mathematical module. It contains seven explanatory views: architecture, whole-forest interpolation/reward mass, directed analogy plus metric persistence, typed final decoding, few-step maps, contextual quality/adapter decisions, and the24+2 campaign boundary. Each view shows whether it is an existing component, proposed method, exact mathematical fixture or real evidence snapshot. Interaction state is URL-backed; keyboard/touch access and a text interpretation accompany graphics. No external data is uploaded.

The figure set in `paper/figures/` is exported from these same diagrams. Static vector figures belong in the paper; interactive controls belong in the linked HTML companion. A PDF is not mislabeled interactive. The explorer is not a deployed update to the pre-existing hosted ChatGPT site; that site's editable source/deployment binding has not been established. No public publication is performed.

All source claims are separated from local design choices. The analytic Euler and persistence examples are exact small fixtures, not trained speedup curves or biological results. Geometry cartoons are not molecule coordinates. Visual QA checks numerical invariants, selections, URL roundtrip, SVG export, mobile layout, keyboard operation, absence of network traffic and clipping. Pure mathematical demonstrations execute zero optimizer updates and consume no training slots.

## Selected sources and review boundaries

Detailed source and code pins are in `docs/research/flow_forest_sources_20261111.json`.

1. Lipman et al. Flow Matching for Generative Modeling. arXiv:2210.02747v2. https://arxiv.org/abs/2210.02747v2
2. Lahlou et al. A Theory of Continuous Generative Flow Networks. arXiv:2301.12594v2. https://arxiv.org/abs/2301.12594v2
3. Geng et al. Mean Flows for One-step Generative Modeling. arXiv:2505.13447. https://arxiv.org/abs/2505.13447
4. Lee, Yu and Shin. Decoupled MeanFlow. arXiv:2510.24474. https://arxiv.org/abs/2510.24474
5. Sabour et al. Align Your Flow. Official research page. https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/
6. Mean Flow Distillation. arXiv:2606.11155. Abstract-screened newer alternative. https://arxiv.org/abs/2606.11155
7. Flow-GRPO. arXiv:2505.05470v3. https://arxiv.org/abs/2505.05470v3
8. Diffusion Negative-aware FineTuning. arXiv:2509.16117. https://arxiv.org/abs/2509.16117
9. Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. arXiv:2402.09353v6. https://arxiv.org/abs/2402.09353v6
10. Dettmers et al. QLoRA. arXiv:2305.14314. https://arxiv.org/abs/2305.14314
11. PEFT0.21.0 source and documentation. https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora
12. AQLoRA. arXiv:2608.23816v2. Screened alternative, not a native-model winner. https://arxiv.org/abs/2608.23816v2
13. LoRA-GA². arXiv:2608.19800v2. WITHDRAWN 1 September 2026; excluded. https://arxiv.org/abs/2608.19800v2
14. TypeSafe. Introducing System One Models & Jev,15 September2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev
15. Grandis. Directed Algebraic Topology: Models of Non-Reversible Worlds. Cambridge,2009. User-provided book; selected sections1.4,2.1,3.2,6.3, not an exhaustive proof review.
16. Moor et al. Topological Autoencoders. ICML2020. https://proceedings.mlr.press/v119/moor20a.html
17. ProteinBench. arXiv:2409.06744. Benchmark scope reference. https://arxiv.org/abs/2409.06744
18. Limitations of the refolding pipeline for de novo protein design. Protein Science,2026,doi:10.1002/pro.70613. https://pmc.ncbi.nlm.nih.gov/articles/PMC13147959/

The earlier four analogy-paper reviews remain in `docs/topology_analogy/README.md`:2505.18651,2605.11258,2602.01992 and2310.01714. None establishes trained Mora-3 performance. Third-party code was inspected, not installed or run, and no foreign checkpoint was acquired.
