"""Task-specific original-assay masks and no target/audit shortcut predictors."""
import importlib.util,json
from pathlib import Path
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('nci_tasks',ROOT/'scripts/data_intake_NCI_AIDS_tasks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def endpoint(kind='EC50',value=-5.5,unit='M',high=-4):
 return {'source_NSC':'48','endpoint':kind,'source_value':value,'source_ConcUnit':unit,'source_log10_highest_concentration':high,'point_target_available':True,'compound_highestdose_context_key':['48',unit,str(high)],'source_stddev':0,'source_NumExp':2}

def test_class_prediction_hides_class_and_identifiers():
 a=m.conclusion_task('CCO','CA','prediction',{'NSC':'48','CAS':'A'})
 b=m.conclusion_task('CCO','CI','prediction',{'NSC':'MUTATED','CAS':'B'})
 assert a.input_bytes()==b.input_bytes()
 assert json.loads(a.canonical_bytes())['targets']['source_conclusion']=='CA'

def test_class_generation_only_uses_explicit_desired_class():
 a=m.conclusion_task('CCO','CM','generation',{'target_length':3})
 b=m.conclusion_task('CCCCCCCC','CM','generation',{'target_length':8})
 assert a.input_bytes()==b.input_bytes()
 assert json.loads(a.canonical_bytes())['conditions']['desired_source_conclusion']=='CM'

def test_censored_and_unknown_unit_means_not_admitted_as_point_targets():
 p=endpoint();p['point_target_available']=False
 with pytest.raises(ValueError,match='point'):m.endpoint_task('CCO',p,'prediction',{})
 with pytest.raises(ValueError,match='unit'):m.endpoint_task('CCO',endpoint(unit='V'),'prediction',{})

def test_endpoint_response_values_and_postassay_counts_not_predictors():
 a=endpoint(value=-5);b=endpoint(value=-6);b['source_NumExp']=999;b['source_stddev']=0.9
 x=m.endpoint_task('CCO',a,'prediction',{'NSC':'48'});y=m.endpoint_task('CCO',b,'prediction',{'NSC':'MUTATED'})
 assert x.input_bytes()==y.input_bytes()
 assert set(json.loads(x.canonical_bytes())['conditions'])=={'molecule_source_connectivity_smiles','assay_panel','assay_readout','concentration_unit','source_log10_highest_tested_concentration','source_stereochemistry_known'}

def test_joint_profile_requires_literal_matching_range_context():
 a=endpoint();b=endpoint('IC50',-4.5,high=-3.7)
 with pytest.raises(ValueError,match='context'):m.joint_task('CCO',a,b,'prediction',{})

def test_joint_values_hidden_and_samecell_not_inferred():
 a=m.joint_task('CCO',endpoint(),endpoint('IC50',-4.5),'prediction',{})
 b=m.joint_task('CCO',endpoint(value=-6),endpoint('IC50',-3.9),'prediction',{'NSC':'MUTATED'})
 assert a.input_bytes()==b.input_bytes()
 assert json.loads(a.canonical_bytes())['audit']['same_cell_or_replicate_pair'] is False

def test_bad_source_class_not_silently_relabelled():
 with pytest.raises(ValueError,match='class'):m.conclusion_task('CCO','unknown','prediction',{})
