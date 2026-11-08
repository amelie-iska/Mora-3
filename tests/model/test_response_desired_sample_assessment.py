import copy
import pytest
import torch
from test_response_desired_sampling import setup_case, sample

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bound_assessment_reference(device):
    from mora3.model.response_desired_sampling import assess_desired_sample
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    h,c,d=setup_case(device);s=sample(h,c,d)
    r=assess_desired_sample(s,[d])
    v=float(s['log_expression'][0,0]);row=r['per_record'][0]
    assert row['squared_violation_mean']==max(-v,0.,v-1.)**2
    assert row['assessed_count']==1 and row['unavailable_requested_count']==1
    assert row['axis_ordinals']==[0] and row['unrequested_gene_count']==0
    assert r['sample_binding']==s['binding']
    assert r['prediction_sha256']==s['prediction_sha256']
    assert not r['training_admission']

@pytest.mark.parametrize('field',['prediction','request','binding','weight','schema','encoding'])
def test_tampered_sample_refuses(field):
    from mora3.model.response_desired_sampling import assess_desired_sample, _digest
    h,c,d=setup_case();s=copy.deepcopy(sample(h,c,d))
    if field=='prediction':s['log_expression'][0,0]+=1
    if field=='request':s['configuration']['requests'][0]['intervals']=((0,'A',2.,3.),)
    if field=='binding':s['binding']='f'*64
    if field=='weight':s['model_state_sha256']='f'*64
    if field=='schema':s['configuration']['model']['architecture']='legacy'
    if field=='encoding':s['prediction_hash_encoding']='other'
    if field in ('request','schema'):s['binding']=_digest(s['configuration'])
    with pytest.raises(ValueError):assess_desired_sample(s,[d])


def test_unknown_and_unrequested_are_not_successes():
    from mora3.model.response_desired_sampling import assess_desired_sample
    from mora3.model.response_desired import DesiredExpressionOrdinal
    h,c,_=setup_case();d=DesiredExpressionOrdinal(c,((1,'B',None,None),))
    r=assess_desired_sample(sample(h,c,d),[d])['per_record'][0]
    assert r['squared_violation_mean'] is None
    assert r['satisfied_count']==r['assessed_count']==0
    assert r['unavailable_requested_count']==r['unrequested_gene_count']==1

@pytest.mark.parametrize('location,key,value',[('config','steps',0),('config','seed',True),('config','sigma_min',float('inf')),('flow','state','other'),('flow','parameterization','other')])
def test_invalid_rebound_solver_schema_refuses(location,key,value):
    from mora3.model.response_desired_sampling import assess_desired_sample, _digest
    h,c,d=setup_case();s=sample(h,c,d)
    target=s['configuration'] if location=='config' else s['configuration']['model']['flow']
    target[key]=value
    # Nonfinite configuration cannot be represented by the canonical encoder.
    if value!=float('inf'):s['binding']=_digest(s['configuration'])
    with pytest.raises(ValueError):assess_desired_sample(s,[d])
