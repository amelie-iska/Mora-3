# Ancestor-only forest context for token continuation

Implementation checkpoint (unadmitted): the opt-in
NativeForestWindowContinuation.teacher_logits_prefix and
teacher_loss_prefix methods accept an explicit KnownForestCausalPrefix and
construct an ancestor-only native forest view of the selected free node.
The complete target/forest state tensor is **not** supplied to this route.
Prior ancestor states are required from an external source-bound, frozen
representation, in strict root-to-parent order, with SHA256-pinned bytes.
There is no automatic substitution of the supplied full candidate states
for missing ancestor states.

The encoder receives the ordered ancestor chain, its declared edge directions,
an explicitly zero unknown target embedding, and the native biological
condition selected by the existing task. Descendant and sibling nodes,
their states, dispositions and directed edges are excluded. The local
forest skeleton is a root-to-target chain; its parent states are treated
as known inputs and the selected target is not clamped. Within-node token
labels remain shifted by one in bounded recurrent windows, and only the
preceding label tokens influence each next-token score.

A changed suffix or sibling/descendant in a complete forest does not alter
the selected node's ancestor-only teacher logits in the scoped regression
tests. Finite backward gradients are verified without optimizer updates.
Strict source/coordinate/context/token-axis and ancestor identity checks
fail closed, and a mutated externally supplied ancestor tensor is rejected.

**Research and qualification limitations:** this does not prove that the
selected path, parents, or embeddings were actually available before the
prediction event in a real source trajectory. The evidence SHA pin alone is
not owner authentication, corpus admission or leakage-proof global splitting.
Original-train complete trajectories, source-native criterion labels,
supported analogy correspondences, frozen encoder/codebook provenance, the
enabled native owner/factory path, full parameter/alias/μP roles, exact
checkpoint/cursor recovery and independent same-snapshot owner releases
remain prerequisites. The legacy joint-forest methods retain their documented
noncausal semantics; these new methods are **not** a production dispatch.
No counted μP run, target run or empirical result is asserted.
