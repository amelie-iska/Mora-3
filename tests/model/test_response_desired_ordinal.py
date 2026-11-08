"""Repeated native names require explicit zero-based ordinal AND exact name."""
from dataclasses import replace
import pytest
import torch
from test_response_desired_assessment import case


def request(context, entries):
    from mora3.model.response_desired import DesiredExpressionOrdinal
    return DesiredExpressionOrdinal(context, entries)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_repeated_axis_coordinates_are_independently_assessed(device):
    from mora3.model.response_desired import assess_desired_expression
    from mora3.model.response_sampling import sample_response
    from mora3.model.response_flow import ResponseFlowHead
    from mora3.model.sciplex import SciPlexSharedModel
    c, _ = case(device)
    c = replace(c, axis_ids=('A','B','B'))
    d = request(c, ((2,'B',1.,2.),(1,'B',0.,1.),(0,'A',None,None)))
    model = ResponseFlowHead(SciPlexSharedModel(c.axis_ids,width=4)).to(device).double()
    s = sample_response(model,[c],steps=2,seed=25801,sigma_min=.01,negative_policy='preserve')
    s['log_expression'] = torch.tensor([[99.,-.5,3.]],device=device,dtype=torch.float64)
    r = assess_desired_expression(s,(d,))['per_record'][0]
    assert r['axis_ordinals'] == [2,1]
    assert r['gene_ids'] == ['B','B']
    assert r['squared_violation_mean'] == .625
    assert r['satisfied_count'] == 0
    assert r['unavailable_requested_count'] == 1
    assert r['unrequested_gene_count'] == 0
    with pytest.raises(ValueError):
        assess_desired_expression(s,(request(replace(c,dose_coordinate=987),d.intervals),))


@pytest.mark.parametrize('entries', [(),((True,'A',0,1),),((-1,'B',0,1),),
    ((3,'B',0,1),),((1,'A',0,1),),((1.,'B',0,1),),
    ((1,'B',0,1),(1,'B',1,2)),((1,'B',None,1),),((1,'B',2,1),),
    ((1,'B',-1,1),),((1,'B',0,float('inf')),),((1,'B',False,1),),
    ([1,'B',0,1],),((1,'B',0),)])
def test_invalid_ordinal_name_bounds_refuse(entries):
    c, _ = case()
    with pytest.raises(ValueError): request(replace(c,axis_ids=('A','B','B')),entries)
