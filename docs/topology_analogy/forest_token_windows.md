# Exact long-payload node-token window representation

This source-only, untrained continuation implements a reversible **target-data
representation** for node payloads longer than the existing neural decoder's
4,096 learned slot positions. It does **not** implement a trained or qualified
multiwindow autoregressive decoder, extend the native trainer, or grant corpus
admission. The current native `forest_token_loss` still requires one terminal
`NODE_END` per supervised node within the decoder's finite slot axis and
correctly refuses oversized payloads.

Let `S <= 4096` be the frozen decoder slot count and let the exact native
tokenizer map source bytes `b` to a canonical token string
`t(b) = (t_0,...,t_{L-1})`. Define `q = floor(L/S)`. For indices
`0 <= j < q`, record the complete S-token segment
`t_{jS:(j+1)S}` with its explicit **external** `(node,j)` address.
The final segment is `(t_{qS:L}, NODE_END)`, possibly `(NODE_END,)`
when `L` is divisible by `S`. The existing `NODE_END` control
is the sole terminal marker; no new token or embedding row is introduced.

Each nonfinal segment has exactly S payload tokens and no controls. The final
segment contains exactly one terminal control at its end. Empty byte strings
produce one stop-only segment. Decoding checks source-forest fingerprint,
coordinate/context and exact vocabulary-axis digests, per-node monotonically
contiguous window addresses, the final marker, all budgets, canonical tokenizer
re-encoding, and byte SHA256 for each supervised payload. A lost, reordered,
forged, prematurely stopped, noncanonical or truncated segment is rejected.
Clamped condition nodes cannot become supervised outputs. Proof of reversible
packing follows immediately by concatenating the canonical payload-token
segments and applying the byte-exact native tokenizer inverse; the final marker
and canonical re-encoding ensure uniqueness under these rules.

**Essential unsolved neural step:** the same fixed forest state fed to the
unchanged learned `slot_embedding` and `readout` for multiple windows would
produce the same predictions. That is not a valid multiwindow decoder. An
explicitly versioned continuation model must condition each window on the
window address **and on the preceding generated/teacher-forced payload tokens**,
with causal masks, bounded state and exact checkpoint/RNG/cursor recovery.
Its μP parameter ledger and optimizer must be requalified; neither this
source representation nor the existing single-window decoder is a valid
substitute. Generated streams must distinguish sampled content from beam
pruning and source evidence, and must not expose future/sibling target tokens.

The independent reference exercise used three already authenticated, computed
**test-only** GRH source exports with an **untrained diagnostic encoder**.
They remain evaluation/development references, not eligible original-train
records. The photonic source required no oversize node window; tropical-attention
and causal-soft-MoE each contained numerous source payloads exceeding 4096
tokens and were reconstructed exactly through 360 and 478 bounded windows,
respectively. This proves data-representation readback, not learned output
quality, causal correctness or biological generalization. The original source
bytes remain in their protected worktree and are not redistributed here.

The unchanged 24-proxy + at most 2 target run allocation, same-snapshot source
and model admission, and genuine source/rights/split requirements remain closed.
