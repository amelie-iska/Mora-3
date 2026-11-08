from pathlib import Path
import sys
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_chemical_projection_scope import project_source_graph
CONFLICT=r'[C:1]([C:3](=[N:5]/[N:2]([H:8])[H:9])/[H:12])(/[C:4](=[N:7]\[H:11])[H:13])=[N:6]\[H:10]'

def test_real_RGD1_conflicting_stereo_is_not_exact_projection_evidence():
 result=project_source_graph(CONFLICT);assert result['source_smiles']==CONFLICT and result['lossy_stereo_warning'];assert not result['exact_projection_eligible'];assert any('STEREONONE' in x for x in result['parser_messages']);assert result['graphs']

def test_benign_retained_hydrogen_warning_does_not_become_lost_stereo():
 result=project_source_graph('C.[H+]');assert result['exact_projection_eligible'];assert not result['lossy_stereo_warning'];assert len(result['graphs'])==3;assert '[H+]' in [x[1]['canonical_isomeric_smiles'] for x in result['graphs']]

def test_exact_projection_preserves_isotope_charge_and_stereo_but_coarse_guard_does_not_assert_them():
 a=project_source_graph('[13CH3][C@H](O)C(=O)[O-]');b=project_source_graph('[13CH3][C@@H](O)C(=O)[O-]');assert a['exact_projection_eligible'] and b['exact_projection_eligible'];assert a['graphs'][0][1]['canonical_isomeric_smiles']!=b['graphs'][0][1]['canonical_isomeric_smiles'];assert a['graphs'][0][1]['molecule_group']==b['graphs'][0][1]['molecule_group'];assert '[13CH3]' in a['graphs'][0][1]['canonical_isomeric_smiles'] and '[O-]' in a['graphs'][0][1]['canonical_isomeric_smiles']

def test_unknown_NCI_stereo_never_becomes_exact_material_identity():
 a=project_source_graph('CCO',source_unknown_stereo=True);assert not a['exact_projection_eligible'] and not a['lossy_stereo_warning'];assert a['match_scope']=='source_unknown_stereo_connectivity_guard_only'

def test_source_atom_maps_remain_original_and_derived_copy_normalization_is_explicit():
 s='[CH3:1][OH:2]';a=project_source_graph(s);assert a['source_smiles']==s;assert a['source_atom_maps_present'];assert a['graphs'][0][1]['canonical_isomeric_smiles']=='CO'
