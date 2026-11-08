"""Synthetic mathematical ADMET loss oracles; no real ADMET model supervision."""
import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import pytest
import torch


def api():
    path=os.environ.get("MORA3_ADMET_OBJECTIVES_MODULE")
    if not path:
        return importlib.import_module("mora3.model.admet_objectives"),importlib.import_module("mora3.data.admet")
    if not Path(path).is_file():pytest.fail("ADMET objective kernels not implemented")
    data_path=os.environ["MORA3_ADMET_DATA_MODULE"]
    if not Path(data_path).is_file():pytest.fail("typed ADMET contracts not implemented")
    spec=importlib.util.spec_from_file_location("mora3.data.admet",data_path);d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
    spec=importlib.util.spec_from_file_location("mora3.model._admet_staged_objectives",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m,d


def masks(n):
    return dict(observed=torch.ones(n,dtype=torch.bool),applicable=torch.ones(n,dtype=torch.bool),
        applicability_known=torch.ones(n,dtype=torch.bool))


def test_BCE_oracle_and_gradient_ignore_missing_unknown_inapplicable_NaNs():
    m,d=api();x=torch.tensor([.2,-.7,float("nan"),float("nan"),float("nan")],dtype=torch.double,requires_grad=True)
    y=torch.tensor([1.,0.,float("nan"),float("nan"),float("nan")],dtype=torch.double)
    mask=dict(observed=torch.tensor([True,True,False,True,True]),applicable=torch.tensor([True,True,True,False,True]),
        applicability_known=torch.tensor([True,True,True,True,False]))
    loss=m.masked_binary_bce(x,y,**mask)
    oracle=(math.log1p(math.exp(-.2))+math.log1p(math.exp(-.7)))/2
    assert loss.item()==pytest.approx(oracle,rel=1e-14)
    loss.backward();expected=torch.tensor([(torch.sigmoid(x[0]).item()-1)/2,torch.sigmoid(x[1]).item()/2,0,0,0],dtype=torch.double)
    torch.testing.assert_close(x.grad,expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize("fn",["masked_binary_bce","masked_regression_mse"])
def test_all_absent_inapplicable_unknown_gives_nonNaN_differentiable_zero(fn):
    m,d=api();x=torch.tensor([float("nan"),float("nan")],dtype=torch.double,requires_grad=True);y=x.detach().clone()
    for fields in [dict(observed=torch.zeros(2,dtype=torch.bool)),dict(applicable=torch.zeros(2,dtype=torch.bool)),
                   dict(applicability_known=torch.zeros(2,dtype=torch.bool))]:
        kw=masks(2);kw.update(fields)
        loss=getattr(m,fn)(x,y,**kw);assert loss.item()==0 and torch.isfinite(loss)
        grad,=torch.autograd.grad(loss,x);assert torch.equal(grad,torch.zeros(2,dtype=torch.double))


def test_MSE_value_and_gradient_reference_with_inactive_nonfinite_targets():
    m,d=api();p=torch.tensor([3.,-2.,float("nan")],dtype=torch.double,requires_grad=True)
    t=torch.tensor([1.,1.,float("nan")],dtype=torch.double);kw=masks(3);kw["observed"][-1]=False
    loss=m.masked_regression_mse(p,t,**kw);assert loss.item()==6.5
    loss.backward();torch.testing.assert_close(p.grad,torch.tensor([2.,-3.,0.],dtype=torch.double))


@pytest.mark.parametrize("relation",["equal","left","right","interval"])
def test_gaussian_censor_oracle_and_autograd_gradcheck(relation):
    m,d=api();mu=torch.tensor([.4],dtype=torch.double,requires_grad=True);s=torch.tensor([1.2],dtype=torch.double,requires_grad=True)
    v=torch.tensor([.8],dtype=torch.double);lo=torch.tensor([-.3],dtype=torch.double);hi=torch.tensor([1.1],dtype=torch.double)
    kw=dict(relation=(relation,),value=v,lower=lo,upper=hi,**masks(1))
    loss=m.gaussian_censored_nll(mu,s,**kw)
    Phi=lambda z:.5*math.erfc(-z/math.sqrt(2))
    if relation=="equal":expected=.5*((.8-.4)/1.2)**2+math.log(1.2)+.5*math.log(2*math.pi)
    elif relation=="left":expected=-math.log(Phi((1.1-.4)/1.2))
    elif relation=="right":expected=-math.log(Phi((.4+.3)/1.2))
    else:expected=-math.log(Phi((1.1-.4)/1.2)-Phi((-.3-.4)/1.2))
    assert loss.item()==pytest.approx(expected,rel=2e-14)
    assert torch.autograd.gradcheck(lambda a,b:m.gaussian_censored_nll(a,b,**kw),(mu,s),eps=1e-6,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize("lo,hi",[(20.,20.1),(-20.1,-20.),(-1e-30,1e-30),(1.,1.+1e-10)])
def test_interval_tails_and_narrow_probability_stable(lo,hi):
    m,d=api();mu=torch.tensor([0.],dtype=torch.double,requires_grad=True);s=torch.tensor([1.],dtype=torch.double,requires_grad=True)
    loss=m.gaussian_censored_nll(mu,s,relation=("interval",),value=torch.tensor([float("nan")],dtype=torch.double),
        lower=torch.tensor([lo],dtype=torch.double),upper=torch.tensor([hi],dtype=torch.double),**masks(1))
    if lo>=8:prob=.5*(math.erfc(lo/math.sqrt(2))-math.erfc(hi/math.sqrt(2)));expected=-math.log(prob)
    elif hi<=-8:prob=.5*(math.erfc(-hi/math.sqrt(2))-math.erfc(-lo/math.sqrt(2)));expected=-math.log(prob)
    else:expected=.5*((lo+hi)/2)**2+.5*math.log(2*math.pi)-math.log(hi-lo)
    assert torch.isfinite(loss) and loss.item()==pytest.approx(expected,rel=1e-12,abs=2e-11)
    grads=torch.autograd.grad(loss,(mu,s));assert all(torch.isfinite(g).all() for g in grads)


def test_gaussian_all_unknown_never_evaluates_NaN_bounds_or_scale():
    m,d=api();mu=torch.tensor([float("nan")],dtype=torch.double,requires_grad=True);scale=mu.detach().clone().requires_grad_()
    z=torch.tensor([float("nan")],dtype=torch.double);kw=masks(1);kw["observed"][:]=False
    loss=m.gaussian_censored_nll(mu,scale,relation=("unreported",),value=z,lower=z,upper=z,**kw)
    assert loss.item()==0;g=torch.autograd.grad(loss,(mu,scale));assert all(torch.equal(x,torch.zeros_like(x)) for x in g)


@pytest.mark.parametrize("bad",["mask_number","negative_scale","bad_relation","interval_reverse","binary_continuous","active_nan"])
def test_explicit_mask_value_and_censor_contract_rejections(bad):
    m,d=api();x=torch.zeros(1,dtype=torch.double);kw=masks(1)
    if bad=="mask_number":
        kw["observed"]=torch.ones(1,dtype=torch.long)
        with pytest.raises(ValueError):m.masked_regression_mse(x,x,**kw)
    elif bad=="binary_continuous":
        with pytest.raises(ValueError):m.masked_binary_bce(x,torch.tensor([.2],dtype=torch.double),**kw)
    elif bad=="active_nan":
        with pytest.raises(ValueError):m.masked_regression_mse(x,torch.tensor([float("nan")],dtype=torch.double),**kw)
    else:
        scale=torch.tensor([-1. if bad=="negative_scale" else 1.],dtype=torch.double)
        lo=torch.tensor([2. if bad=="interval_reverse" else 0.],dtype=torch.double);hi=torch.ones(1,dtype=torch.double)
        rel=("other" if bad=="bad_relation" else "interval",)
        with pytest.raises(ValueError):m.gaussian_censored_nll(x,scale,relation=rel,value=x,lower=lo,upper=hi,**kw)


def test_typed_endpoint_contract_preserves_units_partitions_and_no_universal_score():
    m,d=api();ep=dict(endpoint_key="synthetic:half-life",source_endpoint="half-life",unit="min",unit_basis="source_literal",
        readout="synthetic decay endpoint",measurement_type="synthetic method")
    app={c:"unknown" for c in d.MOLECULE_CLASSES};app["protein"]="applicable"
    definition=d.EndpointDefinition.from_native(ep,objective_kind="regression",applicability=app)
    a=m.AdmetObjectiveContract(definition,d.AssayContext.from_mapping({"matrix":"serum","species":"human"}))
    b=m.AdmetObjectiveContract(definition,d.AssayContext.from_mapping({"matrix":"buffer","species":"human"}))
    assert a.partition_key!=b.partition_key and a.unit=="min"
    x=torch.tensor([1.],dtype=torch.double)
    assert a.point_loss(x,x,native_unit="min",**masks(1)).item()==0
    with pytest.raises(ValueError):a.point_loss(x,x,native_unit="hours",**masks(1))
    with pytest.raises(ValueError):m.require_single_admet_partition((a,b))
    assert not hasattr(m,"universal_ADMET_score")


def test_desired_penalty_is_not_observation_likelihood_and_masks_unknown():
    m,d=api();x=torch.tensor([-2.,.5,3.,float("nan")],dtype=torch.double,requires_grad=True)
    lo=torch.zeros(4,dtype=torch.double);hi=torch.ones(4,dtype=torch.double);active=torch.tensor([True,True,True,False])
    penalty=m.desired_interval_penalty(x,lo,hi,active=active)
    assert penalty.item()==pytest.approx(8/3)
    penalty.backward();torch.testing.assert_close(x.grad,torch.tensor([-4/3,0,4/3,0],dtype=torch.double))


def test_actual_supervision_gate_cannot_be_bypassed_by_flag_dictionary():
    m,d=api()
    for proof in [None,True,{"verified":True},{"source_rights":True,"split_group_review":True}]:
        with pytest.raises(ValueError,match="admission|reader|verified"):m.require_observational_admet_admission(proof)


def test_explicit_binary_unit_inapplicable_is_distinct_from_unknown():
    m,d=api();native=dict(endpoint_key="synthetic:binary",source_endpoint="native_binary_call",unit=None,
        unit_basis="unresolved",readout="source-recorded binary call")
    unknown=d.EndpointDefinition.from_native(native,objective_kind="binary",binary_readout="source positive class")
    classified=d.EndpointDefinition.from_native(native,objective_kind="binary",binary_readout="source positive class",
        unit_applicability="not_applicable")
    assert unknown.unit_applicability=="unknown" and classified.unit_applicability=="not_applicable"
    c=m.AdmetObjectiveContract(classified,d.AssayContext.from_mapping({}))
    x=torch.zeros(1,dtype=torch.double);y=torch.ones(1,dtype=torch.double)
    assert c.binary_loss(x,y,native_unit=None,**masks(1)).item()==pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        m.AdmetObjectiveContract(unknown,d.AssayContext.from_mapping({})).binary_loss(x,y,native_unit=None,**masks(1))


def test_molecule_classes_are_distinct_loss_partitions_even_with_same_native_units():
    m,d=api();native=dict(endpoint_key="synthetic:stability",source_endpoint="native_stability",
        unit="min",unit_basis="source_literal",readout="source matrix stability")
    definition=d.EndpointDefinition.from_native(native,objective_kind="regression")
    context=d.AssayContext.from_mapping({"matrix":"serum"})
    a=m.AdmetObjectiveContract(definition,context,molecule_class="small_molecule")
    b=m.AdmetObjectiveContract(definition,context,molecule_class="RNA")
    assert a.partition_key!=b.partition_key
    with pytest.raises(ValueError):m.require_single_admet_partition((a,b))


def test_observation_level_default_entry_gates_actual_source_before_numerical_work():
    m,d=api();native=dict(endpoint_key="synthetic:source",source_endpoint="half_life",unit="min",unit_basis="source_literal")
    definition=d.EndpointDefinition.from_native(native,objective_kind="regression")
    ctx=d.AssayContext.from_mapping({})
    c=m.AdmetObjectiveContract(definition,ctx,molecule_class="small_molecule")
    def observation(literal):
        wire=dict(schema="mora3.ADMETNativeObservation/v1",
            entity=dict(molecule_class="small_molecule",representation_type="native_SMILES",representation="CCO"),
            endpoint=native,cell=d.NativeCell.parse(literal).as_mapping(),known_context={},
            source_ref={"source_id":"synthetic-unadmitted"},native_source_record={"half_life":literal},
            evidence_role="source_curated_measurement",source_split={"native_export":"test"},
            admission={k:True for k in d.GATES},absence_is_not_negative=True,
            native_fields_not_projected_remain_in_source_record=True,inferred_unit_conversions=[],
            source_native_axis_cell_multiplicity=1,independent_experimental_measurement_count=None)
        return d.AdmetObservation.from_native(d.canonical(wire),applicability="applicable")
    p=torch.tensor([float("nan")],dtype=torch.double,requires_grad=True)
    with pytest.raises(ValueError,match="admission|reader|verified"):
        c.observed_loss(p,observation("1.00"),source_admission=True)
    assert p.grad is None
    zero=c.observed_loss(p,observation(""),source_admission=None)
    assert zero.item()==0
    zero.backward();assert torch.equal(p.grad,torch.zeros_like(p))
