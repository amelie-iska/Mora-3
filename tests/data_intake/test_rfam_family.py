import importlib.util,json
from pathlib import Path
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('rfam_family',ROOT/'scripts/data_intake_rfam_family.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_prediction_only_exposes_sequence_while_unknown_classes_are_not_negatives():
 a=m.family_task('aCGTuR',['RF00001'],['CL00001'],{'score':'1e-400','citation':'source'},'prediction')
 b=m.family_task('aCGTuR',['RF00002'],[],{'score':'999','citation':'changed'},'prediction')
 assert a.input_bytes()==b.input_bytes()
 d=json.loads(a.canonical_bytes());assert d['conditions']=={'source_native_sequence':'aCGTuR'}
 assert d['targets']['family_label_policy']=='positive_source_assignments_only;all_unlisted_families_unknown'

def test_generation_has_no_sequence_or_score_proxy_and_preserves_native_axes():
 a=m.family_task('aCGTuR',['RF00001'],['CL00001'],{'length':6},'generation')
 b=m.family_task('ACGT'*3000,['RF00001'],['CL00001'],{'length':12000},'generation')
 assert a.input_bytes()==b.input_bytes()
 assert json.loads(a.canonical_bytes())['targets']=={'source_native_sequence':'aCGTuR'}
 assert set(json.loads(a.input_bytes())['conditions'])=={'desired_source_Rfam_family_ids','desired_source_clan_ids'}

def test_only_exact_release_family_and_explicit_clan_labels_are_admitted():
 for family,clan in [(['5S_rRNA'],[]),(['RF1'],[]),([],[]),(['RF00001'],['not-a-clan'])]:
  with pytest.raises(ValueError):m.family_task('ACGT',family,clan,{},'prediction')
 assert m.family_task('ACGT',['RF00001','RF00001'],[],{},'prediction')
