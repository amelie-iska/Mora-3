"""Original native source integration: real pinned train grouping, no heldout/training."""
from pathlib import Path

from mora3.research.cross_source_protein_three_family import audit_three_source_protein_closure


def test_exact_original_three_protein_family_component_closure():
    result = audit_three_source_protein_closure(Path.cwd())
    assert result["schema"] == "Mora3.Campaign26OriginalThreeProteinFamilyClosure/v1"
    assert result["source_train_shards"] == 571
    assert result["closure"]["rows"] == {
        "ReactZyme": 40464, "UniProt": 103171, "UniProt-kinetics": 618,
    }
    assert result["closure"]["triple_family_components"] >= 0
    assert result["closure"]["cross_family_components"] >= result["closure"]["triple_family_components"]
    assert result["closure"]["source_identity_only"] is True
    assert result["closure"]["study_not_identity"] is True
    assert result["original_heldout_shards_opened"] is False
    assert result["source_native_values_opened"] is False
    assert result["model_training_admitted"] is False
    assert result["optimizer_updates"] == result["actual_muP_proxy_trials"] == result["actual_target_runs"] == 0
