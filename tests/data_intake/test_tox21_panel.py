import importlib.util,json,copy
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());s=importlib.util.spec_from_file_location('tox',root/'scripts/data_intake_tox21_panel.py');m=importlib.util.module_from_spec(s)
try:s.loader.exec_module(m)
except FileNotFoundError:m=None

def module():assert m is not None,'source Tox21 panel projection missing';return m

def row():return dict(source_split='source_training',source_name='AUDIT_ID',derived_canonical_isomeric_smiles='C[C@H](O)Cl',source_properties={'NR-AR':'0','NR-AhR':'1','FW':'999','Formula':'hidden'},original_sdf_record='hidden depiction and labels')

def test_full_twelve_assay_axis_distinguishes_inactive_zero_missing_x_and_absence():
 p=module().panel({'NR-AR':'0','NR-AhR':'1','SR-p53':'x'});assert len(p['axis'])==12
 assert p['values'][p['axis'].index('NR-AR')]==0 and p['observed_mask'][p['axis'].index('NR-AR')] is True
 assert p['values'][p['axis'].index('SR-p53')] is None and p['observed_mask'][p['axis'].index('SR-p53')] is False
 with pytest.raises(ValueError):module().panel({'NR-AR':'-1'})

def test_final_labels_join_sample_id_not_compound_id():
 r=row();r.update(source_split='source_final_evaluation_structures',source_name='COMPOUND');r['source_properties']={'Compound ID':'COMPOUND','Sample ID':'SAMPLE-01'}
 p=module().row_panel(r,{'SAMPLE-01':{'NR-AR':'1'}});assert p['values'][p['axis'].index('NR-AR')]==1
 with pytest.raises(ValueError):module().row_panel(r,{'COMPOUND':{'NR-AR':'1'}})

def test_observed_call_targets_names_formula_and_SDF_cannot_enter_molecule_predictors():
 r=row();forward,inverse=module().tasks(r,module().panel(r['source_properties']),source_ref={'id':'a'})
 q=copy.deepcopy(r);q['source_properties'].update({'NR-AR':'1','NR-AhR':'0','Formula':'mutated'});q['source_name']='MUTATED';q['original_sdf_record']='MUTATED'
 f2,_=module().tasks(q,module().panel(q['source_properties']),source_ref={'id':'b'});assert forward.input_bytes()==f2.input_bytes()
 d=json.loads(forward.canonical_bytes());assert d['conditions']['canonical_isomeric_smiles']=='C[C@H](O)Cl' and len(d['conditions']['known_assay_axis'])==12
 assert 'Formula' not in str(d['conditions']) and 'observed_mask' not in d['conditions']

def test_inverse_pattern_never_includes_molecule_target_or_source_mass_coordinates():
 r=row();p=module().panel(r['source_properties']);_,t=module().tasks(r,p,source_ref={'id':'a'})
 r['derived_canonical_isomeric_smiles']='CCCC';r['source_name']='different';r['source_properties']['FW']='12';_,other=module().tasks(r,p,source_ref={'id':'b'})
 assert t.input_bytes()==other.input_bytes();c=json.loads(t.canonical_bytes())['conditions'];assert set(c)=={'known_assay_axis','desired_source_call_values','desired_source_call_mask'}
