"""Exhaustively prepare actual Rfam train targets with immutable grouped sources.

This is source/model-input qualification only: never evaluate a model, allocate an
optimizer, open heldout targets, reserve a counted run, or issue owner admission.
New executable adapter hash is recorded separately from older frozen receipts.
"""
from hashlib import sha256
from pathlib import Path
import json

from mora3.data.admet import canonical
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.data.finite_union_native_rows import read_exact
from mora3.research.rfam_goa_registry import FILTER
from mora3.research.rfam_campaign_routes import RfamCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.rfam_native_complete_replay import CENSUS, REBOUND, _original_positions

SOURCE_SNAPSHOT = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
VIEW_POLICY = "11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046"


def _full_prepare(binding, adapter):
    route = binding.route
    if route._batch_size != 1 or len(route._paths) != 72:
        raise ValueError("Exact original single-row Rfam route required")
    coverage = sha256()
    count = positions = 0
    last_cursor = None
    for window in route.iter_from(route.initial_cursor):
        if window.original_source_rows != 1:
            raise ValueError("An original Rfam task was skipped or batched")
        row, = window.batch.native_rows.to_pylist()
        direction = adapter.direction
        result = binding.prepare(window)
        expected = _original_positions(row, direction)
        if (result.unavailable_reason is not None or
                not callable(result.evaluate) or
                result.observed_positions != expected or expected <= 0):
            raise ValueError("One actual native target failed model-input preparation")
        path = window.batch.source_shard["path"]
        ordinal, = window.batch.source_row_ordinals
        coverage.update(canonical([path, ordinal, row["task_id"],
                                  row["native_payload_sha256"], expected]) + b"\n")
        count += 1
        positions += expected
        last_cursor = window.cursor_after
    if count != 18423 or positions < count or last_cursor is None:
        raise ValueError("Rfam original source task population is incomplete")
    if tuple(route.iter_from(last_cursor)):
        raise ValueError("Rfam terminal cursor repeats completed source records")
    if len(adapter._verified_group_cache) != 72:
        raise ValueError("Every original group sidecar must be consumed")
    for source_path, (identity, groups) in adapter._verified_group_cache.items():
        side = adapter._manifest["effective_group_lookup"][source_path]
        if len(groups) != side["rows"]:
            raise ValueError("Incomplete cached group source")
        # Rehash original sidecar at the terminal boundary, independently of
        # metadata change detection during preparation.
        read_exact(side, root=adapter.view.root)
    return dict(actual_prepared_windows=count, actual_native_target_positions=positions,
                original_group_sidecars_sha_verified=72,
                terminal_cursor_sha256=sha256(last_cursor).hexdigest(),
                terminal_restart_exhausted=True,
                prepared_original_population_sha256=coverage.hexdigest())


def prepare_all_original_rfam(root, *, worktree):
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    if root != Path.cwd().resolve() or root == worktree:
        raise ValueError("Run from original source root, isolated code worktree distinct")
    raw = (worktree / CENSUS[0]).read_bytes()
    if sha256(raw).hexdigest() != CENSUS[1]:
        raise ValueError("Previous original Rfam census has changed")
    census = json.loads(raw)
    if census["parent_observations"] != 18423 or census["source_rows"] != 36846:
        raise ValueError("Original Rfam parent/task census differs")
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != SOURCE_SNAPSHOT:
        raise ValueError("Frozen common source changed")
    old_metadata = document(root, Ref(*REBOUND))
    candidate = old_metadata["inactive_candidate_snapshot"]["sha256"]
    view = load_inverse_literal_filtered_native_view(FILTER[0], FILTER[1], root=root)
    if finite_view_policy_sha256(view) != VIEW_POLICY:
        raise ValueError("Source effective group/exclusion policy changed")
    results = {}
    for direction in ("prediction", "generation"):
        dual = {}
        for name, snapshot in (("candidate", candidate), ("common", SOURCE_SNAPSHOT)):
            adapter = RfamCampaignAdapter(view, direction, snapshot, 1)
            binding = adapter.train_binding()
            result = _full_prepare(binding, adapter)
            result["source_snapshot_sha256"] = snapshot
            result["current_executable_adapter_sha256"] = executable_adapter_sha256(
                binding.prepare, root=worktree)
            dual[name] = result
        if any(dual["candidate"][k] != dual["common"][k] for k in (
            "actual_prepared_windows", "actual_native_target_positions",
            "prepared_original_population_sha256", "original_group_sidecars_sha_verified")):
            raise ValueError("Candidate/common source preparation differs")
        results[direction] = dual
    if (results["prediction"]["common"]["actual_native_target_positions"] != 18423
            or results["generation"]["common"]["actual_native_target_positions"] != 2281489):
        raise ValueError("Complete original independent Rfam target counts changed")
    return dict(schema="Mora3.OriginalRfamNativeEveryTargetPrepared/v1",
                canonical_common_snapshot_sha256=SOURCE_SNAPSHOT,
                original_census_sha256=CENSUS[1],
                old_immutable_rebind_receipt_sha256=REBOUND[1],
                source_view_policy_sha256=VIEW_POLICY,
                exact_original_tasks_per_direction=18423,
                fully_prepared_original_native_windows=4 * 18423,
                results=results,
                every_original_train_target_prepared=True,
                old_adapter_pins_reused_as_new_pins=False,
                model_objectives_executed=0, optimizer_updates=0,
                actual_proxy_trials=0, actual_target_runs=0,
                original_source_admission_issued=False,
                target_model_training_admission_issued=False,
                heldout_targets_opened=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python -m mora3.research.rfam_native_full_preparation ACTIVE_ROOT WORKTREE OUTPUT")
    output = Path(sys.argv[3])
    result = prepare_all_original_rfam(sys.argv[1], worktree=sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"all": result["fully_prepared_original_native_windows"],
                      "position_counts": {
                          k: v["common"]["actual_native_target_positions"]
                          for k, v in result["results"].items()}}, sort_keys=True))
