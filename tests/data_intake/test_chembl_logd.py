import importlib.util,json
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('chembl_logd',root/'scripts/data_intake_chembl_logd.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row():
    return dict(activity_id=1,assay_chembl_id='CHEMBL3301363',assay_type='P',standard_type='LogD7.4',type='LogD7.4',standard_value='1.2',value='1.2',standard_relation=None,relation=None,standard_units=None,units=None,standard_flag=1,data_validity_comment=None,canonical_smiles='CCO',molecule_chembl_id='CHEMBL1',parent_molecule_chembl_id='CHEMBL1',target_pref_name='No relevant target')
def test_source_logd_target_and_audit_do_not_enter_predictor_inputs():
    a=row();t=m.source_logd_task(a);a['standard_value']='2.5';a['value']='2.5';a['activity_id']=9;a['molecule_chembl_id']='CHEMBL9';u=m.source_logd_task(a)
    assert t.input_bytes()==u.input_bytes() and t.canonical_bytes()!=u.canonical_bytes()
    d=json.loads(t.canonical_bytes());assert d['conditions']['pH']==7.4
    assert d['targets']['source_reported_logD7_4']==1.2
    assert d['audit']['source_relation'] is None and d['audit']['source_units'] is None
    assert 'source_relation_unspecified' in d['missing']
def test_source_assay_boundary_is_not_assumed_uncensored():
    for value in ['-1.5','4.5']:
        a=row();a['standard_value']=a['value']=value
        with pytest.raises(ValueError,match='boundary'):m.source_logd_task(a)
def test_computed_or_other_source_properties_cannot_enter_observed_objective():
    a=row();a['standard_type']='CX_LogD'
    with pytest.raises(ValueError):m.source_logd_task(a)
    a=row();a['standard_relation']='<'
    with pytest.raises(ValueError,match='relation'):m.source_logd_task(a)
    a=row();a['data_validity_comment']='flagged'
    with pytest.raises(ValueError):m.source_logd_task(a)
