from pathlib import Path
import sys
import pytest
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT / "scripts"))
from data_sources_pdb_qc import validation_record, component_context


def test_official_validation_identity_and_native_outliers_preserved():
    x = validation_record(b'<wwPDB-validation-information><Entry pdbid="1ABC" clashscore="24.02"/><ModelledSubgroup chain="A" resnum="3" rsrz="2.7"/></wwPDB-validation-information>', "1ABC")
    assert x["entry_attributes"]["clashscore"] == "24.02"
    assert x["modelled_subgroups"][0]["attributes"]["rsrz"] == "2.7"
    assert x["task_admitted"] is False
    with pytest.raises(ValueError): validation_record(b'<r><Entry pdbid="2ABC"/></r>', "1ABC")


def test_hetatm_modified_polymer_is_not_automatically_a_binding_ligand():
    c = {"_entity.id": ["1", "2"], "_entity.type": ["polymer", "non-polymer"],
        "_atom_site.group_PDB": ["HETATM", "HETATM"], "_atom_site.label_comp_id": ["7MG", "GOL"],
        "_atom_site.label_entity_id": ["1", "2"], "_atom_site.label_asym_id": ["A", "B"]}
    rows = component_context(c)
    assert rows[0]["ccd_id"] == "7MG" and rows[0]["source_entity_types"] == ["polymer"]
    assert rows[0]["binding_ligand_role"] is None and rows[1]["binding_ligand_role"] is None
