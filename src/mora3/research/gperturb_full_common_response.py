"""Exhaustive original GPerturb common-source response preparation; zero training.

Verify every native response train window and original sidecar roles, WITHOUT
calling a learned objective, opening heldout targets or creating a train run.
The existing finite cursor/reader remain the only source of task admission.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.research.common_source_gperturb_pair import (
    _CONTEXT, _GROUP, _INTERVENTIONS, _POLICY, _direction_signature,
)
from mora3.research.gperturb_campaign_routes import GPerturbNativeCampaignAdapter
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_native_routes import finite_view_policy_sha256

_TWO = ("outputs/common-source-gperturb-paired-20261111/actual-original-two-direction-cell-pairs.json",
        "2af78b814b4f0d847ced638bc13b64fa6db5513d6d0fda1d76af95d7b5d9f94b")
_SNAPSHOT = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_COUNT = 5000
_EXPECTED_SOURCE_CELLS = 20643


def _hex(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def audited_response_record(row):
    """Fail closed on direction, native target exclusion and observational status."""
    if (type(row) is not dict or row.get("direction") != "response"
            or type(row.get("source_index")) is not int or row["source_index"] < 0
            or type(row.get("cell_id")) is not str or not row["cell_id"]
            or any(not _hex(row.get(k)) for k in
                   ("source_shard_sha256", "native_payload_sha256",
                    "expression_sha256", "axis_sha256"))
            or row.get("original_group") != _GROUP
            or row.get("source_context") != _CONTEXT
            or row.get("condition_keys") != ("context", "intervention")
            or row.get("target_keys") != ("expression",)
            or type(row.get("target_positions")) is not int
            or row["target_positions"] != _TARGET_COUNT):
        raise ValueError("Original source-native response roles, identity or masks differ")
    drug = row.get("intervention")
    if (type(drug) is not dict or set(drug) !=
            {"identity", "source_coordinate", "units", "physical_concentration", "is_vehicle"}
            or drug["identity"] not in _INTERVENTIONS
            or drug["physical_concentration"] is not None
            or drug["units"] != "source_scaled_coordinate"
            or type(drug["is_vehicle"]) is not bool
            or drug["is_vehicle"] != (drug["identity"] == "Vehicle")
            or type(drug["source_coordinate"]) not in (int, float)
            or not __import__("math").isfinite(drug["source_coordinate"])
            or drug["source_coordinate"] < 0):
        raise ValueError("Original observational intervention or uncalibrated dose changed")
    return dict(
        source_index=row["source_index"],
        cell_id_sha256=sha256(row["cell_id"].encode()).hexdigest(),
        native_payload_sha256=row["native_payload_sha256"],
        source_shard_sha256=row["source_shard_sha256"],
        expression_sha256=row["expression_sha256"],
        axis_sha256=row["axis_sha256"],
        original_group=row["original_group"],
        intervention_identity=drug["identity"],
        native_response_positions=_TARGET_COUNT,
        physical_concentration_known=False,
    )


def audit_complete_common_response(root, *, worktree):
    """Consume the full original common-snapshot train response route exactly once."""
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    if Path.cwd().resolve() != root or root == worktree or not worktree.is_dir():
        raise ValueError("Original source checkout and distinct implementation worktree required")
    raw = (worktree / _TWO[0]).read_bytes()
    if sha256(raw).hexdigest() != _TWO[1]:
        raise ValueError("Original paired source two-window witness changed")
    previous = json.loads(raw)
    if (previous.get("source_cells") != 2 or previous.get("source_family") != "GPerturb-sciPlex2"
            or previous.get("common_source_snapshot_sha256") != _SNAPSHOT
            or previous.get("training_admission") is not False):
        raise ValueError("Historical original witness is not this source family")
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != _SNAPSHOT:
        raise ValueError("Frozen 29-head common source snapshot changed")
    view = load_assets(root)[0]
    if finite_view_policy_sha256(view) != _POLICY:
        raise ValueError("Original study-filtered view differs from pinned source")
    adapter = GPerturbNativeCampaignAdapter(view, _SNAPSHOT, "response", 1)
    route = adapter.train_binding().route
    if route.task_key != "GPerturb-sciPlex2:response" or route._batch_size != 1:
        raise ValueError("Unexpected original source reader or per-row cursor semantics")
    seen = set()
    interventions = Counter()
    commit = sha256()
    prev_cursor = route.initial_cursor
    first = []
    count = observed = 0
    for window in route.iter_from(route.initial_cursor):
        if window.original_source_rows != 1:
            raise ValueError("Full original source cursor dropped or batched cell")
        before = route._cursor_position(prev_cursor)
        after = route._cursor_position(window.cursor_after)
        if after == before:
            raise ValueError("Native source cursor did not advance")
        signature = _direction_signature(adapter, window)
        proof = audited_response_record(signature)
        key = (proof["source_index"], proof["cell_id_sha256"])
        if key in seen:
            raise ValueError("Original response task cell was visited twice")
        seen.add(key)
        interventions[proof["intervention_identity"]] += 1
        commit.update(canonical([window.batch.source_shard["path"],
                                 list(window.batch.source_row_ordinals), proof]) + b"\n")
        if count < 2:
            prior = previous["paired_rows"][count]
            if (proof["source_index"] != prior["source_index"]
                    or proof["cell_id_sha256"] != prior["cell_id_hash"]
                    or proof["native_payload_sha256"] != prior["response_native_payload_sha256"]
                    or proof["expression_sha256"] != prior["response_expression_sha256"]
                    or proof["axis_sha256"] != prior["expression_axis_sha256"]
                    or sha256(window.cursor_after).hexdigest() != prior["common_response_cursor_sha256"]):
                raise ValueError("Original first-two source/readout/cursor pin changed")
            first.append(proof["cell_id_sha256"])
        count += 1
        observed += proof["native_response_positions"]
        prev_cursor = window.cursor_after
    if count != _EXPECTED_SOURCE_CELLS or len(seen) != count or observed != count * _TARGET_COUNT:
        raise ValueError("Full original response source denominator not preserved")
    if tuple(route.iter_from(prev_cursor)):
        raise ValueError("Terminal native cursor replays original completed train records")
    if sum(interventions.values()) != count:
        raise ValueError("Intervention source-category census differs")
    return {
        "schema": "Mora3.GPerturbCompleteOriginalCommonResponse/v1",
        "source_snapshot_sha256": _SNAPSHOT,
        "source_view_policy_sha256": _POLICY,
        "prior_two_window_witness_sha256": _TWO[1],
        "source_cells": count, "source_target_positions": observed,
        "source_cell_commitment_sha256": commit.hexdigest(),
        "terminal_cursor_sha256": sha256(prev_cursor).hexdigest(),
        "terminal_cursor_exhausted": True,
        "intervention_categories_original": dict(sorted(interventions.items())),
        "first_two_cell_sha256": first,
        "first_two_source_rows_preserved": True,
        "native_target_preparation_executed": count,
        "learned_objective_calls": 0,
        "heldout_targets_opened": False,
        "physical_concentration_calibrated": False,
        "independent_biological_split_approved": False,
        "same_cell_reverse_direction_fully_replayed": False,
        "training_admission": False,
        "optimizer_updates": 0, "counted_runs_started": 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python -m mora3.research.gperturb_full_common_response ORIGINAL_ROOT WORKTREE OUTPUT.json")
    out = Path(sys.argv[3])
    result = audit_complete_common_response(sys.argv[1], worktree=sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"source_cells": result["source_cells"],
                      "positions": result["source_target_positions"],
                      "terminal_exhausted": result["terminal_cursor_exhausted"]}, sort_keys=True))
