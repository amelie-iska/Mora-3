import importlib.util,copy
from pathlib import Path
p=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())/'scripts/data_intake_chebi.py'
s=importlib.util.spec_from_file_location('chebi',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def record():return dict(chebi_accession='CHEBI:1',secondary_ids=['CHEBI:2'],stars=3,is_released=True,modified_on='source-time',default_structure={'smiles':'CCO','is_r_group':False,'standard_inchi_key':'LFQSCWFLJHTTHZ-UHFFFAOYSA-N'})
def test_explicit_secondary_alias_and_original_structure_preserved():
 r=record();before=copy.deepcopy(r);out=m.chebi_reference(r,'CHEBI:2');assert out['status']=='curated_single_compound_structure' and out['source_smiles']=='CCO' and out['molecule_group'];assert r==before
 assert m.chebi_reference(r,'CHEBI:999')['status']=='requested_id_not_primary_or_explicit_secondary'
def test_generic_or_missing_structure_never_becomes_exact_molecule():
 r=record();r['default_structure']={'smiles':'*O','is_r_group':True};out=m.chebi_reference(r,'CHEBI:1');assert out['status']=='generic_R_group_not_single_compound' and out['molecule_group'] is None
 r['default_structure']=None;out=m.chebi_reference(r,'CHEBI:1');assert out['status']=='no_default_structure_specificity_unresolved' and out['molecule_group'] is None
def test_curation_and_wildcard_gates_fail_closed():
 r=record();r['stars']=1;assert m.chebi_reference(r,'CHEBI:1')['molecule_group'] is None
 r=record();r['default_structure']={'smiles':'*O','is_r_group':False};assert m.chebi_reference(r,'CHEBI:1')['status']=='wildcard_structure_not_single_compound'

def cytosine():return dict(chebi_accession='CHEBI:16040',secondary_ids=[],stars=3,is_released=True,chemical_data={'formula':'C4H5N3O','charge':0},default_structure={'smiles':'Nc1ccnc(=O)n1','standard_inchi':'InChI=1S/C4H5N3O/c5-3-1-2-6-4(8)7-3/h1-2H,(H3,5,6,7,8)','standard_inchi_key':'OPTASPLRGRRNAP-UHFFFAOYSA-N','is_r_group':False})
def test_source_InChI_fallback_is_explicit_and_crosschecked_not_SMILES_repair():
 r=cytosine();out=m.chebi_reference(r,'CHEBI:16040');assert out['status']=='curated_single_compound_InChI_reference_after_SMILES_failure';assert out['source_smiles']=='Nc1ccnc(=O)n1';assert out['identity_representation_used']=='source_standard_inchi';assert out['formula_charge_key_consistency'] is True
 assert out['molecule_group']
def test_inchi_formula_or_key_mismatch_never_silently_recovers_identity():
 r=cytosine();r['default_structure']['standard_inchi_key']='WRONG'
 assert m.chebi_reference(r,'CHEBI:16040')['molecule_group'] is None
 r=cytosine();r['chemical_data']['formula']='C4H3N3O'
 assert m.chebi_reference(r,'CHEBI:16040')['molecule_group'] is None

def test_valid_smiles_with_conflicting_source_structure_fields_is_gated():
 r=dict(chebi_accession='CHEBI:1',secondary_ids=[],stars=3,is_released=True,chemical_data={'formula':'C2H6O','charge':0},default_structure={'smiles':'CCO','is_r_group':False,'standard_inchi_key':'WRONG'})
 out=m.chebi_reference(r,'CHEBI:1');assert out['molecule_group'] is None and out['status']=='source_SMILES_formula_charge_or_key_inconsistent'
 r['default_structure']['standard_inchi_key']='LFQSCWFLJHTTHZ-UHFFFAOYSA-N';assert m.chebi_reference(r,'CHEBI:1')['molecule_group']

from rdkit import Chem

def test_charge_and_hydrate_formula_notation_does_not_create_false_mismatch():
 assert m.formula_inventory('C12H12N+')==m.formula_inventory('C12H12N')
 assert m.formula_inventory('C7H15Cl2N2O2P.H2O')==m.formula_inventory('C7H17Cl2N2O3P')
 assert m.formula_inventory('CuSO4.5H2O')=={'Cu':1,'S':1,'O':9,'H':10}
 smiles='C[n+]1ccc(-c2ccccc2)cc1';r=dict(chebi_accession='CHEBI:1',secondary_ids=[],stars=3,is_released=True,chemical_data={'formula':'C12H12N','charge':1},default_structure={'smiles':smiles,'is_r_group':False,'standard_inchi_key':Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))})
 assert m.chebi_reference(r,'CHEBI:1')['status']=='curated_single_compound_structure'

def test_symbolic_polymer_and_unsupported_formula_are_local_identity_holds():
 r=dict(chebi_accession='CHEBI:1',secondary_ids=[],stars=3,is_released=True,chemical_data={'formula':'(C5H8)n.C20H36O','charge':0},default_structure={'smiles':'CCO','is_r_group':False})
 out=m.chebi_reference(r,'CHEBI:1');assert out['status']=='generic_polymeric_formula_not_single_compound' and out['molecule_group'] is None
 r['chemical_data']['formula']='C2(H6)O';out=m.chebi_reference(r,'CHEBI:1');assert out['status']=='source_formula_notation_unresolved' and out['molecule_group'] is None
