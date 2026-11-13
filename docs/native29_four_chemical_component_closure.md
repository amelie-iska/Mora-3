# Four-family original chemical group closure (non-admitted)

2026-11-11 — One bounded original-source qualification in the isolated
`implementation/campaign26-20261110` worktree. This is **not** a campaign run,
training-data release, heldout benchmark approval or an independently adjudicated
chemical identity table. It does not amend the source construction owner's
split policy. The main readback is
`outputs/common-source-four-chemical-component-closure-20261111/actual-original-four-family-component-closure.json`.

The new graph algorithm operates on the already-authenticated **197 original
Parquet group sidecars** covering **79,852 task/role observations** from AZ,
FreeSolv, Tox21 and ExpansionRx. Source paths and byte counts are checked, but
**native numerical source shards are never opened**: in particular, no original
validation or test values, conditions or targets are read. The root/candidate
source metadata remains bound to the two frozen 29-head snapshot hashes
`79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4`
and `1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733`.

The original 9-role reader has an opt-in in-memory typed-row result so the new
component algorithm can consume its **validated literal sidecars** without
reimplementing source parsing or changing its default result. Its ordinary
readback has exactly the same canonical digest as the retained prior evidence:
`725363b6823ff997c905a4d537e4626539a5940c573eb5b0b7e439f6db62e0ac`.
That binds prior `source_role_file_commitment` and ordered retained-group identity
as well as all original roles. The prior immutable JSON byte SHA256 remains
`f8208e69ce6bdd33e99b0edddf5920bddc57f7c2693b965e4b3933baa079dd83`.

| Literal connection policy | Components | Largest component (task rows) | Cross-source components | Train/heldout mixed components | Cross-source heldout components |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact source-native molecule hash | 15,175 | 70 | 374 | 0 | 1 |
| Exact source-native scaffold hash | 9,354 | 958 | 472 | 0 | 4 |
| Either axis (union of both kinds) | 9,354 | 958 | 472 | 0 | 4 |

The largest **training-only membership count** in a union component is 452
training task rows. For these particular source sidecars, adding molecule-hash
links to scaffold-hash links did not merge any further components. This is a
property of the observed original hash adjacency graph, not a universal claim
about canonical molecular identities. Multiple observations may come from the
same original molecule, scaffold or assay, and the graph's edges do not establish
independent experiments, equivalent stereochemistry, identical molecules or
causal biological relationships. The four cross-source heldout components are
relationships between heldout source roles, not evidence of train/heldout
leakage. The zero train/heldout cross-components hold **only** for this literal
molecule/scaffold policy; other chemical representations, assay/study relations,
protein/biological overlap, group transitivity across other sources, parent
identity and benchmark rules still require independent global reconciliation.

The graph commits exact sorted original task membership to three SHA256 hashes
(one per axis), not merely component totals. Synthetic red-first unit tests cover
multi-step transitivity, axis isolation, role separation, hash typing and
identity collisions. The real integration replay checks the complete original
sidecar inventory, matches the prior immutable readback and verifies that no
heldout values are read. The measured original audit took 9.94 s, about
3,319,948 KiB peak RSS. No GPU was reserved, no optimizer update occurred and
no real proxy or target training run was launched.

Before any μP trial the construction owner must validate and admit the **whole
expanded dataset** with independent global source/study/chemical/biological
splits; the model owner must independently attest to the exact same released
snapshot, qualify the complete 29-head architecture including SCRI and
trajectory behavior, reproduce the actual matched validation reference and
native optimizer/cursor/checkpoint/recovery/resource budgets, and obtain a
reviewed exclusive worker release. The counted campaign remains exactly 24
native μP proxy slots plus at most two ~80,283,064-parameter target runs;
failed slots are never replaced. No owner signatures, approvals or train/run
counts have been manufactured by this group diagnostic.
