# Rfam route / native owner snapshot binding — 11 November 2026

The existing native Rfam prediction/generation adapters are implemented. Their
source rows' historical model_adapter_supported=false flag is not proof that no
adapter code exists, and must not be rewritten to claim campaign admission.

A direct prepared evaluator previously checked owner type, generic training
admission and Rfam branch identity, but did not explicitly require the route's
source snapshot to match the owner. The counted update dispatcher already checks
its source/route inventory. This correction additionally binds the direct adapter
boundary; it is not evidence that an unauthorized training run happened.

The evaluator now captures its source snapshot when preparing targets and compares
it with the native owner's verified extra state before reaching the admission call.
A valid snapshot still must pass the unchanged full-owner admission check. Family
axes, exact shared branch, source provenance and positive-only supervision stay
unchanged. No source flags, owner signatures or training release are manufactured.

Two failing regressions reproduced the old boundary behavior, one per direction.
After the correction,40 focused tests passed and two CUDA cases were skipped with
CUDA hidden. All implementation tests were guarded against optimizer steps and
recorded zero attempted optimizer steps. The native owners in the boundary tests
are explicitly synthetic width-eight fixtures, not current trained checkpoints.

Separate real-source readback reconstructed candidate and common snapshot adapters
for each direction and read two consecutive windows from each: eight reads of four
unique original task rows. Original payloads, conditions, targets and counts agree
across snapshots. Prediction has one positive family target per sampled row;
generation has137/195 native nucleotide-plus-EOS positions. A cursor from the
other snapshot is rejected. Prediction inputs remain sequence-only; generation
inputs remain desired family/clan only. No model objective was evaluated by this
source-only replay and no heldout targets were opened.

The executable adapter hash changes with this correction. Historical metadata
pins are preserved rather than silently reused; complete runtime release metadata
must be constructed for the final current code. The bounded source replay is not
complete62-route replay, a matched validation baseline, global split acceptance,
full dataset readiness or a training run. Those requirements remain unfinished.

Local evidence: outputs/rfam-route-owner-binding-20261111T1738/ in the isolated
main-release worktree. The existing24-proxy-plus-two-target campaign is unchanged.
GitHub publication is independent of these local implementation qualifications.
