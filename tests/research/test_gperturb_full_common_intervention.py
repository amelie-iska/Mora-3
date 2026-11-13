"""Native ORIGINAL GPerturb inverse task roles and full-source evidence, no updates."""
import pytest

from mora3.research.gperturb_full_common_intervention import audited_intervention_record


def _record(i, *, intervention="Dex"):
    return dict(
        direction="intervention", source_index=i, cell_id=f"source-cell-{i}",
        source_shard_sha256="a" * 64, native_payload_sha256="d" * 64,
        original_group="study:GSE139944:GSM4150377",
        source_context=dict(study="GSM4150377", cell_line="A549",
                            exposure_hours=24, baseline=None, desired_response=None),
        condition_keys=("context", "expression"), target_keys=("intervention",),
        expression_sha256="b" * 64, axis_sha256="c" * 64,
        intervention=dict(identity=intervention, source_coordinate=.001,
                          units="source_scaled_coordinate",
                          physical_concentration=None,
                          is_vehicle=intervention == "Vehicle"),
        target_positions=2,
    )


def test_reverse_native_roles_keep_expression_condition_and_two_targets():
    row = audited_intervention_record(_record(11))
    assert row["source_index"] == 11
    assert row["native_intervention_positions"] == 2
    assert row["expression_is_condition"] is True
    assert row["physical_concentration_known"] is False
    assert len(row["cell_id_sha256"]) == 64


@pytest.mark.parametrize("mutation", [
    {"direction": "response"},
    {"condition_keys": ("context", "intervention", "expression")},
    {"target_keys": ("expression",)},
    {"target_positions": 5000},
    {"source_index": -1},
    {"axis_sha256": "x" * 64},
    {"intervention": dict(identity="Dex", source_coordinate=.001,
                          units="source_scaled_coordinate",
                          physical_concentration=.001, is_vehicle=False)},
    {"intervention": dict(identity="Vehicle", source_coordinate=.001,
                          units="source_scaled_coordinate",
                          physical_concentration=None, is_vehicle=False)},
])
def test_inverse_native_refuses_target_leakage_or_fabricated_dose(mutation):
    with pytest.raises(ValueError):
        audited_intervention_record(_record(0) | mutation)


def test_real_original_inverse_terminal_readback_is_a_source_audit_not_a_run():
    from hashlib import sha256
    from pathlib import Path
    import json
    worktree = Path("/home/iska/Documents/amelie/bio/Mora-3-campaign26-20261110")
    name = worktree / "outputs/common-source-gperturb-full-intervention-20261111/actual-original-common-intervention.json"
    raw = name.read_bytes()
    assert sha256(raw).hexdigest() == "fa57f3e1e550a62fd5f6d5e1aba44e719621b4277badb76a0aad463ca3ad8f9e"
    observed = json.loads(raw)
    assert observed["source_cells"] == observed["native_target_preparation_executed"] == 20643
    assert observed["source_target_positions"] == 41286
    assert observed["terminal_cursor_exhausted"] is True
    assert observed["response_direction_aggregate_concordance"] is True
    assert observed["full_pairwise_source_identity_comparison"] is False
    assert observed["independent_biological_split_approved"] is False
    assert observed["training_admission"] is False
    assert observed["optimizer_updates"] == observed["counted_runs_started"] == 0
    assert observed["learned_objective_calls"] == 0
