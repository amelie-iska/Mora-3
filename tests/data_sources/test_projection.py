from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT / "scripts"))
from data_sources_native import reaction_condition, known_atom_axis_condition


def test_energy_and_source_id_changes_do_not_change_reaction_conditions():
    a = {"source_native": {"reactant": "[C:1]=[O:2]", "product": "[C:1][O:2]", "DE_F": 3, "reaction": "MR_1_0"}, "measurements": {"DE_F": 3}}
    b = {"source_native": {**a["source_native"], "DE_F": 999, "reaction": "MR_99_99"}, "measurements": {"DE_F": 999}}
    assert reaction_condition(a, audit_only=True) == reaction_condition(b, audit_only=True)
    assert set(reaction_condition(a, audit_only=True)) == {"reactant", "product"}


def test_coordinate_target_mutation_does_not_change_known_atom_axis():
    a = [{"id": "1", "type_symbol": "C", "label_atom_id": "CA", "label_comp_id": "ALA", "label_asym_id": "A", "label_seq_id": "1", "label_entity_id": "1", "Cartn_x": "1", "x_angstrom": 1.0, "occupancy": "1", "B_iso_or_equiv": "10"}]
    b = [{**a[0], "Cartn_x": "999", "x_angstrom": 999.0, "occupancy": "0.1", "B_iso_or_equiv": "500"}]
    assert known_atom_axis_condition(a) == known_atom_axis_condition(b)
    assert "Cartn_x" not in known_atom_axis_condition(a)[0]
