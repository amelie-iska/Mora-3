"""Independent native-coordinate likelihood and cooperative owner checks."""
import copy
import math
from dataclasses import replace

import pytest
import torch

from mora3.data.source_scalar_native_rows import ScalarTarget
from mora3.model.source_scalar_objectives import (SharedNativeScalarModel,SourceScalarAxis,
    native_scalar_likelihood,scalar_condition_graph,source_scalar_mup_parameter_roles)
from test_full_owner import actual_core_assets,cases,_construct


FREE=('FreeSolv','experimental_hydration_free_energy','experimental_hydration_free_energy',
      'kcal/mol',None,None,'source_experimental_reported_point')
LOGD=('ChEMBL37-AZ-experimental-logD7_4','source_reported_experimental_logD7_4',
      'source_reported_nominal_logD7_4',None,'shake_flask',None,
      'declared_native_nominal_logD7_4_coordinate_relation_unspecified')


def target(kind,*,value=None,lower=None,upper=None,relation=None,lower_inclusive=None,upper_inclusive=None,uncertainty=None):
    return ScalarTarget(kind,value,lower,upper,lower_inclusive,upper_inclusive,relation,None,
        'fixture-explicit-native-coordinate',b'{}',uncertainty,
        'experimental_uncertainty_may_be_default' if uncertainty is not None else None,b'{}')


def condition(key):
    return dict(task='source_scalar_prediction',native_entity=dict(
        molecule_class='small_molecule',representation_type='SMILES',representation='CCO'),
        endpoint=dict(semantic_name=key[2],description='Reported native source coordinate; physical and assay missingness remains explicit',readout=key[2],
            unit=key[3],unit_basis='unresolved' if key[3] is None else 'source_literal',scale=key[6],
            measurement_type=key[2],binary_readout=None,unit_applicability='unknown' if key[3] is None else 'declared_native_unit'),
        known_context={} if key==FREE else dict(pH=7.4,solvent_pair=['octan-1-ol','water'],method='shake_flask'))


def test_full_point_censor_interval_likelihood_and_gradients():
    rows=(target('reported_point',value=0.7,uncertainty=9.),
          target('censored_left',upper=0.5,relation='<=',lower_inclusive=False,upper_inclusive=True),
          target('censored_right',lower=-0.6,relation='>',lower_inclusive=False,upper_inclusive=False),
          target('interval',lower=-0.5,upper=1.6,relation='interval',lower_inclusive=True,upper_inclusive=False),
          target('missing'))
    mu=torch.tensor([0.1,0.2,0.3,0.4,70.],dtype=torch.float64,requires_grad=True)
    sigma=torch.tensor([0.8,1.2,0.7,0.9,3.],dtype=torch.float64,requires_grad=True)
    got=native_scalar_likelihood(mu,sigma,rows)
    d=torch.distributions.Normal(mu,sigma)
    exact=-d.log_prob(mu.new_tensor([0.7,0,0,0,0]))[0]
    left=-d.cdf(mu.new_tensor([0,0.5,0,0,0]))[1].log()
    right=-torch.log1p(-d.cdf(mu.new_tensor([0,0,-0.6,0,0]))[2])
    interval=-(d.cdf(mu.new_tensor([0,0,0,1.6,0]))[3]-d.cdf(mu.new_tensor([0,0,0,-0.5,0]))[3]).log()
    reference=(exact+left+right+interval)/4
    torch.testing.assert_close(got,reference,rtol=1e-14,atol=1e-14)
    a=torch.autograd.grad(got,(mu,sigma),retain_graph=True);b=torch.autograd.grad(reference,(mu,sigma))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=1e-13,atol=1e-13);assert x[-1]==0


def test_nominal_coordinate_and_errorbar_stay_distinct():
    m=torch.tensor([0.],dtype=torch.float64);s=m+1
    row=target('reported_nominal_coordinate',value=1.2)
    assert row.raw_relation is None
    got=native_scalar_likelihood(m,s,(row,),nominal=True)
    torch.testing.assert_close(got,m.new_tensor(0.5*1.2**2+0.5*math.log(2*math.pi)))
    with pytest.raises(ValueError,match='partition'):
        native_scalar_likelihood(m,s,(row,))
    a=target('reported_point',value=-2.0,uncertainty=0.6)
    b=replace(a,uncertainty_value=300.0)
    assert torch.equal(native_scalar_likelihood(m,s,(a,)),native_scalar_likelihood(m,s,(b,)))


def test_no_label_or_assay_injection_into_known_conditions():
    for key in (FREE,LOGD):scalar_condition_graph(key,condition(key))
    for change in (dict(target=7.),dict(native_entity={'molecule_class':'small_molecule','representation_type':'SMILES','representation':'CCO','source_id':'x'}),
                   dict(known_context={'pH':7.4}),dict(endpoint={**condition(FREE)['endpoint'],'unit':'kJ/mol'})):
        with pytest.raises(ValueError):scalar_condition_graph(FREE,{**condition(FREE),**change})
    with pytest.raises(ValueError):scalar_condition_graph(LOGD,{**condition(LOGD),'known_context':{**condition(LOGD)['known_context'],'pH':8.}})


def test_real_shared_owner_readout_roles_strict_state_and_gradient(tmp_path,actual_core_assets):
    owner=_construct(tmp_path,actual_core_assets)
    shared=owner.core.parent.parent.annotation
    axis=SourceScalarAxis('a'*64,'b'*64,((FREE,2),(LOGD,3)))
    model=SharedNativeScalarModel(shared,axis,max_batch_size=2)
    assert model.shared is shared
    mean,scale=model(FREE,(condition(FREE),))
    assert mean.dtype==scale.dtype==torch.float64 and bool((scale>0).all())
    nll=native_scalar_likelihood(mean,scale,(target('reported_point',value=0.7),))
    head=model._partition(FREE);other=model._partition(LOGD)
    grads=torch.autograd.grad(nll,(head.weight,head.bias,shared.payload.embedding.weight,other.weight),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in grads[:3])
    assert grads[3] is None
    roles=source_scalar_mup_parameter_roles(model,identifier_axis=('context','observation'))
    assert set(roles)=={n for n,_ in model.named_parameters(remove_duplicate=False)}
    for key in model.heads:
        assert roles['heads.'+key+'.weight'].role=='output'
        assert roles['heads.'+key+'.weight'].axes[0].kind=='finite'
    state=copy.deepcopy(model.state_dict());copy.deepcopy(model).load_state_dict(state)
    state['_extra_state']['proof_sha256']='c'*64
    with pytest.raises(ValueError,match='checkpoint'):model.load_state_dict(state)
    with pytest.raises(ValueError,match='filter'):model.source_batch_objectives(object(),object(),split='train')
