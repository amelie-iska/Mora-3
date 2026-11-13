"""Full ORIGINAL GPerturb common-native response source contract, zero training."""
from pathlib import Path
import json
import pytest

from mora3.research.gperturb_full_common_response import audited_response_record

ROOT = Path("/home/iska/Documents/amelie/bio/Mora-3")
WORKTREE = Path("/home/iska/Documents/amelie/bio/Mora-3-campaign26-20261110")


def _row(i, *, intervention="Dex"):
    return dict(
        direction="response", source_index=i, cell_id=f"cell-{i}",
        source_shard_sha256="a" * 64, native_payload_sha256="d" * 64,
        original_group="study:GSE139944:GSM4150377",
        source_context=dict(study="GSM4150377", cell_line="A549",
                            exposure_hours=24, baseline=None, desired_response=None),
        condition_keys=("context", "intervention"), target_keys=("expression",),
        expression_sha256="b" * 64, axis_sha256="c" * 64,
        intervention=dict(identity=intervention, source_coordinate=.001,
                          units="source_scaled_coordinate",
                          physical_concentration=None, is_vehicle=False),
        target_positions=5000,
    )


def test_checked_response_signature_follows_source_roles_only():
    record = audited_response_record(_row(7))
    assert record["source_index"] == 7
    assert len(record["cell_id_sha256"]) == 64
    assert record["native_response_positions"] == 5000
    assert record["physical_concentration_known"] is False


@pytest.mark.parametrize("mutant", [
    {"direction": "intervention"}, {"target_keys": ("intervention",)},
    {"condition_keys": ("context", "intervention", "expression")},
    {"target_positions": 4999}, {"original_group": "study:unknown"},
    {"intervention": dict(identity="Dex", source_coordinate=.001,
                          units="source_scaled_coordinate",
                          physical_concentration=.001, is_vehicle=False)},
])
def test_response_signature_rejects_unsupported_source_renaming(mutant):
    with pytest.raises(ValueError):
        audited_response_record(_row(0) | mutant)


def test_original_source_integration_receipt_readback_without_duplicate_full_scan():
    """The separate executable original-source replay must have terminated first."""
    path = WORKTREE / "outputs/common-source-gperturb-full-response-20261111/actual-original-common-response.json"
    doc = json.loads(path.read_text())
    assert doc["source_cells"] == 20643
    assert doc["source_target_positions"] == 20643 * 5000
    assert doc["terminal_cursor_exhausted"] is True
    assert doc["first_two_source_rows_preserved"] is True
    assert doc["native_target_preparation_executed"] == 20643
    assert sum(doc["intervention_categories_original"].values()) == 20643
    assert doc["heldout_targets_opened"] is False
    assert doc["learned_objective_calls"] == 0
    assert doc["optimizer_updates"] == doc["counted_runs_started"] == 0
    assert doc["training_admission"] is False
