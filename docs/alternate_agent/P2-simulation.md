# Package 2 — tasmorpho/BETSEE simulation-source preparation

## Verified inputs and code rights

The tasmorpho checkout is pinned to
`ab09864214ca36787a2144bb89425c8b9610bd6f`. The 869-byte summary has SHA256
`657aec89b9f828c355dc7af968a38bd3e1332accdc18f71d512b9d0e18552734`.
The source script, LICENSE, summary and all seven case configurations were
independently compared to that commit's Git blobs before preparation.

The tasmorpho code is MIT licensed with the repository's 2026 copyright notice.
The user's `amelie-iska/betsee` fork was verified at
`5cbe6d5f2d06f0752661d909f2eb0a32c35064d1`; its LICENSE Git blob is
`569485840ff49f2332c02f82cf0bd3ac6c493e95` and states BSD-2-Clause.
BETSEE is the GUI environment. BETSE is the numerical engine. Neither code
license independently establishes third-party source, calibration, observation
or source-run provenance and redistribution rights.

Primary fork references:
- https://github.com/amelie-iska/tasmorpho/tree/ab09864214ca36787a2144bb89425c8b9610bd6f
- https://github.com/amelie-iska/betsee/blob/5cbe6d5f2d06f0752661d909f2eb0a32c35064d1/LICENSE

## Implemented and tested

`simulation_contracts.py` defines explicit interventions, source/configuration/
geometry/seed bindings, simulation-only evidence, optional independent experiment
references, and conservative family grouping. Different seeds and nearby
intervention values cannot escape their shared simulation/control lineage.
Response comparisons require exact native columns and predeclared tolerances;
no rounding, clipping or automatic retry turns a failed comparison into a pass.

`prepare_tasmorpho_replay.py` creates a fresh, immutable packet with seven typed
candidate cases, original digests, declared simulator actions, control arm,
0–0.02-second declared window, response units and explicit unknown uncertainty,
species, historical engine version and seed. Na/K actions are membrane
permeability multipliers, not inferred chemical identities. Gap-junction block
percentage remains distinct. Normalized integrated costs have units of seconds,
not joules. These are aggregate simulated outcomes, not individual cell states,
experimental phenotypes or validated morphology axes.

The parser's seven cases share one source/control run. No random row split or
new independent biological sample count is claimed. Retry policy retains every
failed seed/configuration; successful-only conditioning would change the sampled
distribution and needs a separate declared protocol.

## Uncompleted mandatory execution

**No independent simulation regeneration was performed.** BETSE, BETSEE and
PySide2 are absent from the selected `tokengt` environment. No install was
performed. Historical seed/version authority is also absent. The upstream main
removes its output root and loads dill state; it was not executed or imported.
The preparation script intentionally returns blocked status rather than running
an unsafe historical pipeline or presenting parsed old output as a new run.

Counts: seven preserved candidate cases, one shared simulation-run group,
zero new simulations, zero experimental response observations, zero newly
admitted training rows. Rights/source-run qualification remains incomplete.

Authoritative receipts: `outputs/alternate-20261108/tasmorpho-v2/receipt.json`,
its `simulation-candidates.json`, and `P2-source-execution-v2.json`. The earlier
v1 packet is retained; v2 adds explicit response units and control/time context.

Next gate: coordinator-approved engine availability and exact versioned seeded
export adapter, fresh bounded simulation, independently checked outputs and
predeclared tolerances, followed by source-run/input rights and heldout grouping.
The current preparation must not be relabeled an executed replay.
