"""Source-only BioGRID target isolation and frozen original-route reference tests."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from mora3.data.biogrid_chemical_current import BioGRIDChemicalCondition, BioGRIDChemicalPacket
from mora3.research.campaign26_biogrid_complete_common import (
    verify_biogrid_native_target, frozen_biogrid_prefix,
)
from mora3.research.native_annotation_registry import load_original_annotation_assets
from mora3.research.native_annotation_campaign_routes import bind_biogrid_chemical

ACTIVE = Path("/home/iska/Documents/amelie/bio/Mora-3")
CANDIDATE = "1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"
COMMON = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"


def test_literal_joint_targets_not_supplied_to_condition():
    condition = BioGRIDChemicalCondition("CHEBI:1", "molecule", "34", "9606")
    packet = BioGRIDChemicalPacket(condition, ("inhibits", "chemical genetic"), ("a", 0, "0" * 64))
    doc = {"conditions": condition.inputs(), "targets": {
        "action": "inhibits", "interaction_type": "chemical genetic"}}
    prepared = SimpleNamespace(observed_positions=1, unavailable_reason=None, evaluate=lambda owner: None)
    assert verify_biogrid_native_target(doc, packet, prepared) == 1
    with pytest.raises(ValueError, match="target"):
        verify_biogrid_native_target({**doc, "conditions": {
            **doc["conditions"], "action": "inhibits"}}, packet, prepared)
    with pytest.raises(ValueError, match="target"):
        verify_biogrid_native_target(doc, packet, SimpleNamespace(
            observed_positions=2, unavailable_reason=None, evaluate=lambda owner: None))


def test_exact_frozen_candidate_common_biogrid_source_cursor_prefix():
    if not ACTIVE.is_dir():
        pytest.skip("Original source checkout not mounted")
    view, sources, availability, _, _ = load_original_annotation_assets(ACTIVE)
    key = "biogrid_chemical_annotation"
    assert availability[key]["train_rows"] == 2706
    assert availability[key]["validation_rows"] == availability[key]["test_rows"] == 0
    source = sources["biogrid_chemical"]
    a = bind_biogrid_chemical(source, view, snapshot_sha256=CANDIDATE, batch_size=1)
    b = bind_biogrid_chemical(source, view, snapshot_sha256=COMMON, batch_size=1)
    first, old = frozen_biogrid_prefix(a, source)
    second, new = frozen_biogrid_prefix(b, source)
    assert old == new
    assert all(x.batch.native_rows.equals(y.batch.native_rows, check_metadata=True)
               for x, y in zip(first, second, strict=True))
    assert len(old) == 2 and all(item[4] == 1 for item in old)
