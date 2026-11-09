# Package 1 — native SCRI qualification proposal

## Implemented

`native_owner.NativeSCRIModel` adds a separately owned exact spatial incidence
projection. The existing annotation payload and TokenGT feature, role, graph
seed, transformer layers and final normalization remain shared by identity.
The original rank-two owner is unchanged. The spatial branch has a declared
finite identifier axis held fixed across widths; tested capacity is 64 cells.
The implementation ceiling is 128 cells, not a claim about arbitrary tissue size.

Spatial output reads the query-node token, not a hidden target or audit field.
Complete nodes/edges/payloads and padded-batch attention work are checked before
encoding. Malformed or target-bearing graphs, rank overflow and inconsistent
source axes refuse rather than truncate. Checkpoint preflight checks every
configuration and tensor, plus equality across tied aliases, before copying.

`full_owner.py` and `full_owner_mup.py` contain an opt-in constructor/registration
proposal. `native_routes.py` binds explicit transition and spatial envelopes to
the existing finite native reader/route abstraction. Unknown targets yield an
unavailable event without neural evaluation. Invalid count/category targets and
official heldouts refuse before forward execution. Production admission is
unchanged and the active checkout has not received this patch.

## Measured evidence

| Width | Registered cooperative parameters | Alias names | Unique tensors |
|---:|---:|---:|---:|
| 8 | 492,107 | 970 | 188 |
| 12 | 764,307 | 970 | 188 |
| 16 | 1,066,907 | 970 | 188 |
| 32 | 2,581,307 | 970 | 188 |
| 80 | 10,042,907 | 970 | 188 |
| 266 | 80,304,407 | 970 | 188 |

These counts include every registered head of the tested cooperative source
assembly, but not unbound optional expanded lanes. The omitted fields are
enumerated in `counts-large-v1/counts.json`. This is not the final all-source
80M model or a qualified zero-shot transfer result.

Single-thread CPU dense synthetic spatial graphs:

| Cells | Directed edges | Tokens | Forward+backward seconds | Cumulative process peak RSS |
|---:|---:|---:|---:|---:|
| 8 | 56 | 65 | 0.0294 | 625,868,800 B |
| 16 | 240 | 257 | 0.0643 | 632,160,256 B |
| 32 | 992 | 1,025 | 0.3401 | 692,121,600 B |
| 64 | 4,032 | 4,097 | 1.6441 | 1,273,655,296 B |

Every case had finite gradients and nonzero spatial-incidence gradients. These
are actual measured stress shapes, not measured biological neighborhood
statistics, uncontended throughput or GPU training memory. Dense node/edge
expansion is material; the attention budget applies to graph tokens, not cells.

The ~10M and ~80M assemblies passed CPU SCRI forward/backward and complete
registered-parameter checkpoint recovery in fresh processes initialized with a
different seed. Predictions and all parameter hashes matched exactly; corrupted
aliases refused before mutation. Large optimizer recovery and all-head gradients
were not exercised. The separate tiny component diagnostic used five total
optimizer updates and verified exact optimizer/RNG/cursor continuation.

## Receipts and coordinator gate

Under `outputs/alternate-20261108/`: `focused-final.json` (100 tests +56 subtests),
`P1-full-factory-2.json`, `P1-routes-green.json`, `tiny-v1/small-qualification.json`,
`contexts-v1/contexts.json`, both `counts-*/counts.json`, and both
`large-recovery-*/receipt.json`. Input/source/output digests are in the package
seal. No real source rows were optimized or newly admitted.

Next: review and merge the proposed factory/role and native route changes at a
safe boundary; bind actual source rights/counts/groups/axes and the current
train/validation snapshot; profile intended real contexts and larger capacities;
qualify the complete expanded owner, all active operators, optimizer/recovery
and GPU resources. The existing unconditional production refusal must not be
removed merely to make these tests launch a campaign.
