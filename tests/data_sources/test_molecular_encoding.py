import sys
from pathlib import Path
import pytest
import selfies as sf
sys.path.insert(0,str(Path.cwd()/"scripts"))
from data_sources_molecular_encoding import encode_molecule

@pytest.mark.parametrize("smiles",["C[NH3+]","[13CH3][C@H](O)[NH3+]",r"C/C=C\C","[Na+].CC(=O)[O-]"])
def test_charge_isotope_stereo_and_components_roundtrip_under_declared_parser(smiles):
    r=encode_molecule(smiles)
    assert r["native_smiles"]==smiles and r["graph_roundtrip_pass"]
    assert r["source_canonical_isomeric_smiles"]==r["decoded_canonical_isomeric_smiles"]
    assert r["standard_selfies_target_present"] and r["standard_selfies"]
    assert not r["task_admitted"] and not r["model_adapter_supported"]

def test_native_invalid_source_is_retained_without_target_or_invented_geometry():
    r=encode_molecule("this-is-not-smiles")
    assert r["native_smiles"]=="this-is-not-smiles"
    assert not r["native_current_RDKit_parse_pass"] and not r["standard_selfies_target_present"]
    assert r["standard_selfies"] is None and r["coordinate_geometry_present"] is False

def test_full_molecular_axis_not_cropped_to_model_token_limit():
    r=encode_molecule("C"*600)
    assert r["native_smiles"]=="C"*600 and r["graph_roundtrip_pass"]
    assert r["derived_atom_count"]==r["standard_selfies_token_count"]==600

def test_decoder_graph_change_is_quarantined_not_accepted(monkeypatch):
    monkeypatch.setattr(sf,"decoder",lambda _:"CCN")
    r=encode_molecule("CCO")
    assert r["native_current_RDKit_parse_pass"] and not r["graph_roundtrip_pass"]
    assert r["status"]=="graph_roundtrip_mismatch"
    assert r["generated_candidate_selfies"] and r["standard_selfies"] is None
    assert not r["standard_selfies_target_present"]
