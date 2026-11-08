"""Applied annotation/physical native μP owners, synthetic vocabulary/views only."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.physical_trainer import SharedPhysicalModel
from mora3.model.physical_head import StructuredPhysicalHead
from mora3.model.annotation_trainer import AnnotationTrainer
from mora3.model.annotation_mup import annotation_mup_parameter_roles,annotation_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from test_annotation_trainer import batch
from test_annotation_vocabulary import admitted
from test_sciplex_mup import make as native_response
from test_flow_mup import trees_equal


def vocab():
    return AnnotationVocabulary(json.dumps(dict(learned_hex=[],controls=['pad','node','edge'],
        certificate={'scope':'synthetic-unit-fixture-not-fitted-data'}),sort_keys=True))


def make(width=8,physical=False,**kw):
    args=dict(width=width,heads=2,layers=2,max_tokens=512,parameterization_policy='mup-annotation-v1',
        base_width=8,readout_mult=1.5,attn_mult=.5,encoder_var=.7)
    args.update(kw)
    cls=SharedPhysicalModel if physical else SharedAnnotationModel
    if physical:args.update(parameter_axis=('mass','pH'),method_axis=('MALDI','ESI'))
    return cls(vocab(),('GO:0000001','GO:0000002'),('1.1.1.1','2.2.2.2'),**args)


@pytest.mark.parametrize('physical',[False,True])
def test_complete_owner_family_parameter_counts_roles_and_muadamw_no_updates(physical):
    ms=[make(w,physical).double() for w in (8,12,16)]
    family=annotation_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    assert family.total_numel==((5370,8722,12522) if physical else (5300,8620,12388))
    for member,m in zip(('base','delta','target'),ms):
        roles=annotation_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))
        assert not any(r.role=='custom' for r in roles.values())
        assert roles['payload.projection.hidden_weight'].role=='hidden'
        assert roles['payload.projection.length_weight'].role=='input'
        opt=MuAdamW(m,family,roles,base_lr=.002,weight_decay=.01,
            decay_aliases=tuple(n for n,r in roles.items() if len(r.axes)==2),member=member)
        assert not opt.state
    assert ms[-1].get_extra_state()['version']==2


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_go_ec_actual_readouts_target_exclusion_and_all_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model.uniprot_go import annotation_graph_task
    from dataclasses import replace
    torch.manual_seed(2026110651);m=make(16).to(device=device,dtype=dtype)
    v=admitted();record,query=annotation_graph_task(v)
    graph=m.payload.forward_task(record,task=query,identifiers=m.sequence_identifiers())
    latent=m.backbone([graph])[0][:,0]
    expected={name:F.linear(latent*(1.5/(16/8)),getattr(m,name).output.weight,
                            getattr(m,name).output.bias) for name in ('go','ec')}
    actual=m([v]);tol=2e-5 if dtype==torch.float32 else 1e-11
    for name,key in [('go','GO'),('ec','EC')]:
        torch.testing.assert_close(actual[key],expected[name],rtol=tol,atol=tol)
    owners=tuple(m.parameters())
    ga=torch.autograd.grad(sum(t.sum() for t in actual.values()),owners,allow_unused=True)
    gb=torch.autograd.grad(sum(t.sum() for t in expected.values()),owners,allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    changed=replace(v,target_bytes=b'{"annotation_packet":{"proteinDescription":{"changed":"held-out-target"}}}')
    after=m([changed])
    for key in actual:assert torch.equal(actual[key],after[key])


def test_physical_readout_scales_complete_latent_plus_query_and_preserves_query_exclusion():
    h=StructuredPhysicalHead(16,('mass','pH'),('MALDI','ESI'),
        parameterization_policy='mup-physical-v1',base_width=8,output_mult=1.5).double()
    latent=torch.linspace(-1,1,32,dtype=torch.float64).reshape(2,16).requires_grad_()
    target=[[dict(parameter_type='mass',unit_status='unknown',molecule_scope=None,method='MALDI')],[]]
    features=(latent[:1]+h.query.weight[:1])*(1.5/(16/8))
    actual=h(latent,target)
    expected={name:F.linear(features,getattr(h,name).weight,getattr(h,name).bias)
              for name in ('unit','scope','method')}
    for name in actual:torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
    changed=copy.deepcopy(target);changed[0][0].update(method='ESI',unit_status='explicit_source_unit',molecule_scope='other')
    for name,value in h(latent,changed).items():assert torch.equal(value,actual[name])


def test_shared_native_backbone_not_rescaled_and_alias_roles_remain_compatible():
    source=native_response(12)
    before={n:p.detach().clone() for n,p in source.backbone.named_parameters()}
    annotation=make(12,backbone=source.backbone)
    roles=annotation_mup_parameter_roles(annotation,identifier_axis=('context','observation'))
    assert annotation.backbone is source.backbone
    for name,p in source.backbone.named_parameters():assert torch.equal(before[name],p)
    assert roles['backbone.incidence.weight'].role=='input'


def test_legacy_annotation_owner_cannot_silently_adopt_new_native_initializer():
    source=native_response(12);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='explicit|policy|schema'):
        SharedAnnotationModel(vocab(),('GO:0000001',),('1.1.1.1',),width=12,heads=2,layers=2,
                              backbone=source.backbone)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('failure',['base','readout','axis','payload','native','extra','physical-policy'])
def test_source_owner_policy_axis_module_drift_refuses_family(physical,failure):
    ms=[make(w,physical).double() for w in (8,12,16)]
    if failure=='base':ms[1]=make(12,physical,base_width=6).double()
    if failure=='readout':ms[1]=make(12,physical,readout_mult=2.).double()
    if failure=='axis':ms[1].go.axis_ids=('GO:0000002','GO:0000001')
    if failure=='payload':ms[1].payload.pooling='bag'
    if failure=='native':ms[1].backbone._native_mup=('legacy',None,None)
    if failure=='extra':ms[1].extra=torch.nn.Linear(12,12).double()
    if failure=='physical-policy':
        if physical:ms[1].physical.configuration['mup']['base_width']=4
        else:ms[1]._configuration['mup']['base_width']=4
    with pytest.raises(ValueError):annotation_mup_width_family(*ms,identifier_axis=('protein-coordinate',))


def test_annotation_muadamw_staged_resume_with_new_split_state_and_cursor():
    torch.manual_seed(2026110652)
    ms=[make(w).double() for w in (8,12,16)]
    family=annotation_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    from dataclasses import replace
    view=replace(admitted(),target_bytes=b'{"annotation_packet":{"uniProtKBCrossReferences":[{"database":"GO","id":"GO:0000001"}]}}')
    b=batch(view);roles=annotation_mup_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    decay=tuple(n for n,r in roles.items() if len(r.axes)==2)
    trainer=AnnotationTrainer(ms[-1],b,lr=.002,mu_family=family,
        mu_identifier_axis=('protein-coordinate',),weight_decay=.01,decay_aliases=decay)
    trainer.step();state=trainer.state_dict();trainer.step();expected=trainer.state_dict()
    fresh=AnnotationTrainer(make(16).double(),b,lr=.002,mu_family=family,
        mu_identifier_axis=('protein-coordinate',),weight_decay=.01,decay_aliases=decay)
    fresh.load_state_dict(state);fresh.step()
    assert trees_equal(expected,fresh.state_dict())


def test_opted_annotation_trainer_requires_qualified_family():
    with pytest.raises(ValueError,match='family'):AnnotationTrainer(make(),batch())


@pytest.mark.parametrize('failure',['owner','payload','head-axis','native-base','legacy-target'])
def test_new_owner_checkpoint_policy_preflight_before_any_child_copy(failure):
    m=make();state=copy.deepcopy(m.state_dict());target=make()
    if failure=='owner':state['_extra_state']['mup']['base_width']=4
    if failure=='payload':state['payload._payload_parameterization']['token_axis_sha256']='f'*64
    if failure=='head-axis':state['go._extra_state']['axis_ids']=('GO:0000002','GO:0000001')
    if failure=='native-base':state['backbone._native_initialization']['parameterization']['base_width']=4
    if failure=='legacy-target':
        target=SharedAnnotationModel(vocab(),('GO:0000001','GO:0000002'),('1.1.1.1','2.2.2.2'),width=8,heads=2,layers=2,max_tokens=512)
    first=next(k for k,v in state.items() if isinstance(v,torch.Tensor));state[first]=torch.full_like(state[first],99.)
    before=[p.clone() for p in target.parameters()];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):target.load_state_dict(state)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,target.parameters()))
