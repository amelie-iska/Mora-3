"""Learned quality axes with explicit labels, not biological-validity fixtures."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_flow import make,problem,skeleton,state,SOURCE
from mora3.model.forest_quality import (ForestQualitySpec,NativeForestQualityHead,
    ForestQualityLabels,forest_quality_loss,forest_quality_parameter_roles)
from mora3.model.parameterization import parameter_ledger,width_family


def build(width=8):
    return NativeForestQualityHead(make(width),ForestQualitySpec(('supported-order','constraint-compatible'),'c'*64,(1.,2.)))


def labels():
    whole=torch.tensor([[1.,0.],[.5,1.]],dtype=torch.float64)
    nodes=torch.zeros(2,5,2,dtype=torch.float64);wm=torch.ones_like(whole,dtype=torch.bool)
    nm=torch.ones_like(nodes,dtype=torch.bool);nm[:,0]=False
    return ForestQualityLabels(whole,nodes,wm,nm,SOURCE,'c'*64,'d'*64,'train')


def test_native_quality_returns_all_axes_and_real_shared_gradients():
    h=build();r,t=problem();f=skeleton();x=state().requires_grad_()
    out=h(r,t,x,f)
    assert out['criterion_logits'].shape==(2,2)and out['node_logits'].shape==(2,5,2)
    assert out['criteria']==h.spec.criteria and out['calibrated']is False
    assert torch.allclose(out['ranking_score'],(out['criterion_logits'].sigmoid()*x.new_tensor([1.,2.])/3).sum(-1))
    out['criterion_logits'].square().sum().backward()
    for p in(h.forest_readout.weight,h.field.state_projection.weight,h.field.geometry.graph.feature.weight):
        assert p.grad is not None and p.grad.abs().sum()>0
    assert x.grad[:,1:].abs().sum()>0


def test_masked_quality_equation_unknown_nan_and_hidden_targets():
    h=build();r,t=problem();f=skeleton();x=state();gold=labels()
    gold.node_available[0,2,1]=False;gold.node_values[0,2,1]=float('nan')
    out=h(r,t,x,f);result=forest_quality_loss(h,r,t,x,f,gold)
    a=torch.nn.functional.binary_cross_entropy_with_logits(out['criterion_logits'][gold.whole_available],gold.whole_values[gold.whole_available],reduction='sum')
    b=torch.nn.functional.binary_cross_entropy_with_logits(out['node_logits'][gold.node_available],gold.node_values[gold.node_available],reduction='sum')
    expected=(a+b)/(gold.whole_available.sum()+gold.node_available.sum())
    assert torch.equal(result['loss'],expected)
    result['loss'].backward();assert h.node_readout.weight.grad.abs().sum()>0
    rr,tt=problem(target=1e9)
    assert torch.equal(h(r,t,x,f)['criterion_logits'],h(rr,tt,x,f)['criterion_logits'])


@pytest.mark.parametrize('fault',('test','source','policy','clamp','range','nan'))
def test_bad_or_heldout_quality_labels_refuse(fault):
    h=build();gold=labels();r,t=problem();f=skeleton();x=state()
    if fault=='test':gold=replace(gold,split='test')
    if fault=='source':gold=replace(gold,source_snapshot_sha256='f'*64)
    if fault=='policy':gold=replace(gold,label_policy_sha256='f'*64)
    if fault=='clamp':gold.node_available[:,0]=True
    if fault=='range':gold.whole_values[0,0]=1.1
    if fault=='nan':gold.whole_values[0,0]=float('nan')
    with pytest.raises(ValueError):forest_quality_loss(h,r,t,x,f,gold)


def test_no_quality_labels_are_inactive_and_not_false_negatives(monkeypatch):
    h=build();gold=labels();gold.whole_available[:]=False;gold.node_available[:]=False
    monkeypatch.setattr(h,'forward',lambda *a:pytest.fail('no labels'))
    r,t=problem();assert forest_quality_loss(h,r,t,state(),skeleton(),gold)['loss']is None


def test_quality_roles_width_family_and_strict_checkpoint():
    hs=[build(w)for w in(8,12,16)]
    family=width_family(*[parameter_ledger(h,forest_quality_parameter_roles(h,identifier_axis=tuple('g:'+str(i)for i in range(8))))for h in hs])
    assert family.total_numel==tuple(sum(p.numel()for p in h.parameters())for h in hs)
    h=hs[0];clone=copy.deepcopy(h);clone.load_state_dict(copy.deepcopy(h.state_dict()))
    r,t=problem();assert torch.equal(h(r,t,state(),skeleton())['criterion_logits'],clone(r,t,state(),skeleton())['criterion_logits'])
    bad=copy.deepcopy(h.state_dict());bad['_extra_state']['spec']['label_policy_sha256']='0'*64
    before={n:p.clone()for n,p in h.named_parameters()}
    with pytest.raises(ValueError):h.load_state_dict(bad)
    assert all(torch.equal(before[n],p)for n,p in h.named_parameters())


@pytest.mark.parametrize('weights',((1e300,1e300),(1e-300,1e-300)))
def test_extreme_ranking_weights_normalize_before_float32_conversion(weights):
    h=NativeForestQualityHead(make().float(),ForestQualitySpec(('a','b'),'c'*64,weights));r,t=problem()
    f=skeleton();f=replace(f,states=f.states.float());out=h(r,t,state().float(),f)
    assert torch.isfinite(out['ranking_score']).all()
    assert torch.allclose(out['ranking_score'],out['criterion_logits'].sigmoid().mean(-1))


def test_native_quality_binding_cannot_override_refused_or_unknown_constraints():
    from mora3.model.forest_quality import ForestConstraintDecision
    from mora3.reasoning.flow_forest_refinement import forest_fingerprint
    h=build();r,t=problem();f=skeleton()
    with torch.no_grad():h.forest_readout.bias.fill_(100.)
    for status in('refused','unknown'):
        def certify(candidate):return ForestConstraintDecision(forest_fingerprint(candidate),candidate.context_sha256,'e'*64,status,'f'*64,True)
        bound=h.bind_context(r,t,certify,constraint_policy_sha256='e'*64)
        result=bound(f)
        assert result.status==status and result.score>.99 and result.repairable
        assert bound.evidence(result.evidence_sha256)['criterion_scores']
        assert bound.evidence(result.evidence_sha256)['calibrated']is False


def test_quality_binding_rejects_stale_constraint_and_model():
    from mora3.model.forest_quality import ForestConstraintDecision
    from mora3.reasoning.flow_forest_refinement import forest_fingerprint
    h=build();r,t=problem();f=skeleton()
    def bad(candidate):return ForestConstraintDecision('0'*64,candidate.context_sha256,'e'*64,'pass','f'*64)
    bound=h.bind_context(r,t,bad,constraint_policy_sha256='e'*64)
    with pytest.raises(ValueError):bound(f)
    def good(candidate):return ForestConstraintDecision(forest_fingerprint(candidate),candidate.context_sha256,'e'*64,'pass','f'*64)
    bound=h.bind_context(r,t,good,constraint_policy_sha256='e'*64)
    with torch.no_grad():h.forest_readout.bias.add_(1)
    with pytest.raises(ValueError):bound(f)
