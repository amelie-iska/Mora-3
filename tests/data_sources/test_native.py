import importlib.util
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve()
ROOT = next(p for p in ROOT.parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT / "scripts"))
from data_sources_native import parse_cif, atom_rows, mapped_reaction, physical_link


def test_cif_preserves_quoted_reserved_values_and_multiline():
    text = "data_x\n_title.text\n;first\nsecond\n;\nloop_\n_test.a\n_test.b\n'_literal' 'loop_'\nA \"B C\" # comment\n"
    d = parse_cif(text)
    assert d["_title.text"] == ["first\nsecond"]
    assert d["_test.a"] == ["_literal", "A"]
    assert d["_test.b"] == ["loop_", "B C"]


def test_incomplete_cif_loop_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        parse_cif("data_x\nloop_\n_t.a\n_t.b\n1\n")


def test_native_atoms_keep_altloc_model_author_ids_and_missing_values():
    d = {"_atom_site." + k: v for k, v in {
        "id": ["1", "2", "3"], "Cartn_x": ["1.001", "1.002", "3"],
        "Cartn_y": ["2", "2", "4"], "Cartn_z": ["3", "3", "5"],
        "label_alt_id": ["A", "B", "."], "pdbx_PDB_model_num": ["1", "1", "2"],
        "occupancy": ["0.4", "0.6", "?"], "auth_asym_id": ["AA", "AA", "B"]}.items()}
    rows = atom_rows(d)
    assert len(rows) == 3 and rows[1]["label_alt_id"] == "B"
    assert rows[2]["pdbx_PDB_model_num"] == "2"
    assert rows[2]["occupancy_numeric"] is None
    assert rows[0]["Cartn_x"] == "1.001" and rows[0]["x_angstrom"] == 1.001


def test_nonfinite_native_geometry_rejected():
    with pytest.raises(ValueError, match="finite"):
        atom_rows({"_atom_site.id": ["1"], "_atom_site.Cartn_x": ["nan"],
                   "_atom_site.Cartn_y": ["0"], "_atom_site.Cartn_z": ["0"]})


def rxn(r="[C:1]=[O:2]", p="[C:1][O:2]"):
    return dict(reaction="MR_12_1", reactant=r, product=p, DE_F="3", DE_B="4", DG_F="5", DG_B="6", DH="-1")


def test_reaction_preserves_source_units_mapping_and_conformer_group():
    r = mapped_reaction(rxn())
    assert r["reaction_family_key"] == "RGD1:MR_12"
    assert r["evidence_class"] == "computational_quantum_chemistry"
    assert r["measurements"]["DE_F"] == {"value": 3.0, "unit": "kcal/mol", "meaning": "forward_electronic_activation_energy"}
    assert r["reactant_atom_axis"] == [1, 2]
    assert r["geometry_available"] is False


@pytest.mark.parametrize("r,p", [("[C:1]=[O:1]", "[C:1]#[O:2]"),
    ("[C:1]=[O:2]", "[C:1]#[N:2]"), ("CO", "CO"), ("[C:1]=[O:2]", "[C:1]#[O:3]")])
def test_bad_atom_maps_cannot_be_admitted(r, p):
    with pytest.raises(ValueError):
        mapped_reaction(rxn(r, p))


def test_explicit_hydrogen_axis_is_not_removed():
    r = mapped_reaction(rxn("[C:1]([H:2])=[O:3]", "[C:1]([H:2])[O:3]"))
    assert r["atom_count"] == 3
    assert r["reactant_atom_axis"] == [1, 2, 3]


def test_nonfinite_energy_rejected():
    d = rxn(); d["DH"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        mapped_reaction(d)


def test_physical_link_is_not_called_direct_experiment_or_negative():
    r = physical_link(dict(protein1="9606.X", protein2="9606.Y", experimental="0", database="800", textmining="100", combined_score="801"))
    assert r["evidence_class"] == "source_integrated_physical_association"
    assert r["experimental_channel_supported"] is False
    assert r["direct_binding_demonstrated"] is None
    assert r["score_scale"] == "integer_0_to_1000"


def test_bad_score_rejected():
    with pytest.raises(ValueError):
        physical_link(dict(protein1="9606.X", protein2="9606.Y", experimental="1001", database="0", textmining="0", combined_score="800"))
