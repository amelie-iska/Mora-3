"""Exhaustive original GPerturb inverse-direction common native preparation.

Bounded train-only source task readback. Neither a model objective nor optimizer
is called. No full-source experimental/causal pairing or heldout validation claim.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import re

from mora3.data.admet import canonical
from mora3.research.common_source_gperturb_pair import (
    _CONTEXT, _GROUP, _INTERVENTIONS, _POLICY, _direction_signature,
)
from mora3.research.gperturb_campaign_routes import GPerturbNativeCampaignAdapter
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_native_routes import finite_view_policy_sha256

_PAIR = (
    "outputs/common-source-gperturb-paired-20261111/actual-original-two-direction-cell-pairs.json",
    "2af78b814b4f0d847ced638bc13b64fa6db5513d6d0fda1d76af95d7b5d9f94b",
)
_RESPONSE = (
    "outputs/common-source-gperturb-full-response-20261111/actual-original-common-response.json",
    "92a840288fd0bb8b888202eabd5d234fca55aba3adc228e296dcdaaec3d28b09",
)
_SNAPSHOT = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_SOURCE_CELLS = 20643


def _hex(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def audited_intervention_record(row):
    """Verify native inverse target, response-conditioned source and identity."""
    if (
        type(row) is not dict or row.get("direction") != "intervention"
        or type(row.get("source_index")) is not int or row["source_index"] < 0
        or type(row.get("cell_id")) is not str or not row["cell_id"]
        or any(not _hex(row.get(k)) for k in (
            "source_shard_sha256", "native_payload_sha256",
            "expression_sha256", "axis_sha256",
        ))
        or row.get("original_group") != _GROUP
        or row.get("source_context") != _CONTEXT
        or row.get("condition_keys") != ("context", "expression")
        or row.get("target_keys") != ("intervention",)
        or type(row.get("target_positions")) is not int
        or row["target_positions"] != 2
    ):
        raise ValueError("Original inverse task identity, target exclusion or native masks differ")
    treatment = row.get("intervention")
    if (
        type(treatment) is not dict or set(treatment) != {
            "identity", "source_coordinate", "units", "physical_concentration", "is_vehicle",
        }
        or treatment["identity"] not in _INTERVENTIONS
        or treatment["physical_concentration"] is not None
        or treatment["units"] != "source_scaled_coordinate"
        or type(treatment["is_vehicle"]) is not bool
        or treatment["is_vehicle"] != (treatment["identity"] == "Vehicle")
        or type(treatment["source_coordinate"]) not in (int, float)
        or not math.isfinite(treatment["source_coordinate"])
        or treatment["source_coordinate"] < 0
    ):
        raise ValueError("Native observational intervention or uncalibrated dose changed")
    return dict(
        source_index=row["source_index"],
        cell_id_sha256=sha256(row["cell_id"].encode()).hexdigest(),
        native_payload_sha256=row["native_payload_sha256"],
        source_shard_sha256=row["source_shard_sha256"],
        expression_sha256=row["expression_sha256"],
        axis_sha256=row["axis_sha256"],
        original_group=row["original_group"],
        intervention_identity=treatment["identity"],
        native_intervention_positions=2,
        expression_is_condition=True,
        physical_concentration_known=False,
    )


def _pinned_json(worktree, record):
    raw = (worktree / record[0]).read_bytes()
    if sha256(raw).hexdigest() != record[1]:
        raise ValueError("Immutable prior original source evidence differs")
    return json.loads(raw)


def audit_complete_common_intervention(root, *, worktree):
    """Consume all original inverse train cells under the exact common29 source."""
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    if Path.cwd().resolve() != root or root == worktree or not worktree.is_dir():
        raise ValueError("Original source checkout and distinct implementation worktree required")
    pair, prior = _pinned_json(worktree, _PAIR), _pinned_json(worktree, _RESPONSE)
    if (
        pair.get("source_cells") != 2
        or pair.get("source_family") != "GPerturb-sciPlex2"
        or pair.get("common_source_snapshot_sha256") != _SNAPSHOT
        or pair.get("training_admission") is not False
        or prior.get("source_cells") != _EXPECTED_SOURCE_CELLS
        or prior.get("source_snapshot_sha256") != _SNAPSHOT
        or prior.get("source_view_policy_sha256") != _POLICY
        or prior.get("native_target_preparation_executed") != _EXPECTED_SOURCE_CELLS
        or prior.get("terminal_cursor_exhausted") is not True
        or prior.get("training_admission") is not False
    ):
        raise ValueError("Exact prior common-source train-only source evidence required")
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != _SNAPSHOT:
        raise ValueError("Frozen common29 source family differs")
    view = load_assets(root)[0]
    if finite_view_policy_sha256(view) != _POLICY:
        raise ValueError("Original study-filtered native source view changed")
    adapter = GPerturbNativeCampaignAdapter(view, _SNAPSHOT, "intervention", 1)
    route = adapter.train_binding().route
    if route.task_key != "GPerturb-sciPlex2:intervention" or route._batch_size != 1:
        raise ValueError("Wrong original reverse native route or cursor semantics")
    seen = set()
    categories = Counter()
    commitment = sha256()
    prior_cursor = route.initial_cursor
    first = []
    count = 0
    observed = 0
    for window in route.iter_from(route.initial_cursor):
        if window.original_source_rows != 1:
            raise ValueError("Original inverse source task row dropped or batched")
        before = route._cursor_position(prior_cursor)
        after = route._cursor_position(window.cursor_after)
        if after == before:
            raise ValueError("Native inverse source cursor did not advance")
        proof = audited_intervention_record(_direction_signature(adapter, window))
        if proof["source_index"] != count:
            raise ValueError("Native source row ordering/skips differ from response original")
        key = (proof["source_index"], proof["cell_id_sha256"])
        if key in seen:
            raise ValueError("Original inverse cell visited more than once")
        seen.add(key)
        categories[proof["intervention_identity"]] += 1
        commitment.update(canonical([
            window.batch.source_shard["path"],
            list(window.batch.source_row_ordinals), proof,
        ]) + b"\n")
        if count < 2:
            witness = pair["paired_rows"][count]
            if (
                proof["source_index"] != witness["source_index"]
                or proof["cell_id_sha256"] != witness["cell_id_hash"]
                or proof["native_payload_sha256"] != witness["intervention_native_payload_sha256"]
                or proof["expression_sha256"] != witness["intervention_condition_expression_sha256"]
                or proof["axis_sha256"] != witness["expression_axis_sha256"]
                or sha256(window.cursor_after).hexdigest() != witness["common_intervention_cursor_sha256"]
            ):
                raise ValueError("Pinned original first-two inverse source identities or cursor changed")
            first.append(proof["cell_id_sha256"])
        count += 1
        observed += proof["native_intervention_positions"]
        prior_cursor = window.cursor_after
    if count != _EXPECTED_SOURCE_CELLS or len(seen) != count or observed != 2 * count:
        raise ValueError("Complete original inverse source denominator not preserved")
    if tuple(route.iter_from(prior_cursor)):
        raise ValueError("Terminal inverse cursor replays completed native source records")
    if dict(sorted(categories.items())) != prior["intervention_categories_original"]:
        raise ValueError("Inverse/response native intervention population counts differ")
    if first != prior["first_two_cell_sha256"]:
        raise ValueError("Inverse/response original first-two cell identities differ")
    return dict(
        schema="Mora3.GPerturbCompleteOriginalCommonIntervention/v1",
        source_snapshot_sha256=_SNAPSHOT,
        source_view_policy_sha256=_POLICY,
        prior_two_window_witness_sha256=_PAIR[1],
        prior_complete_response_sha256=_RESPONSE[1],
        source_cells=count, source_target_positions=observed,
        source_cell_commitment_sha256=commitment.hexdigest(),
        terminal_cursor_sha256=sha256(prior_cursor).hexdigest(),
        terminal_cursor_exhausted=True,
        intervention_categories_original=dict(sorted(categories.items())),
        first_two_cell_sha256=first,
        first_two_source_rows_preserved=True,
        native_target_preparation_executed=count,
        response_direction_aggregate_concordance=True,
        full_pairwise_source_identity_comparison=False,
        learned_objective_calls=0,
        heldout_targets_opened=False,
        physical_concentration_calibrated=False,
        independent_biological_split_approved=False,
        training_admission=False,
        optimizer_updates=0, counted_runs_started=0,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python -m mora3.research.gperturb_full_common_intervention ORIGINAL_ROOT WORKTREE OUTPUT.json"
        )
    result = audit_complete_common_intervention(sys.argv[1], worktree=sys.argv[2])
    output = Path(sys.argv[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(dict(
        source_cells=result["source_cells"],
        positions=result["source_target_positions"],
        terminal_exhausted=result["terminal_cursor_exhausted"],
    ), sort_keys=True))
