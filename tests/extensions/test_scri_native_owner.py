"""New bounded CPU qualification; synthetic inputs are not source admission."""
import copy
from dataclasses import replace
import json
import pytest
import torch
from mora3.data.annotation_axes import AnnotationAxes
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.physical_trainer import SharedPhysicalModel
from mora3.model.tokengt import TokenGT
from mora3.extensions.scri.contracts import NativeValue, SourceEvidence, StateRecord
from mora3.extensions.scri.model import SCRITargetAxis
from mora3.extensions.scri.spatial import build_neighborhood_query
from mora3.extensions.scri.native_owner import NativeSCRIModel, SpatialBudget, native_scri_roles
from mora3.model.parameterization import parameter_ledger, width_family

AXIS=SCRITargetAxis('a'*64,(('rna','G','count'),),'poisson')
ID_AXIS=('context','observation')

def shared(width=8):
    vocab=AnnotationVocabulary(json.dumps({'learned_hex':[],'controls':['pad','node','edge'],
        'certificate':{'scope':'synthetic-test-only'}}))
    registry=AnnotationAxes(json.dumps({'rows':{'GO':[{'axis_id':'GO:0000001'}],'EC':[{'axis_id':'1.1.1.1'}]},
        'binding':{'manifest_sha256':'b'*64,'registry_is_admission':False}}))
    backbone=TokenGT(width,2,width=width,heads=2,layers=2,scaling_policy='mup-v1',
        initialization_policy='mup-transformer-v1',parameterization_policy='mup-tokengt-v1',
        base_width=8,base_ffn_width=8)
    return SharedPhysicalModel.from_registry(vocab,registry,parameter_axis=('mass',),method_axis=('assay',),
        width=width,heads=2,layers=2,backbone=backbone,max_tokens=8192,
        parameterization_policy='mup-annotation-v1',base_width=8)

def query(n=8):
    source=SourceEvidence('fixture','r1','c'*64,'fixture','CC0','synthetic_fixture','study','sample','donor','experiment',None)
    rows=tuple(StateRecord(str(i),replace(source,source_locator=str(i)),'fixture-system',None,0,'s','baseline',
        (NativeValue('rna','G',i,'count',True),),(i/n,0.),'um','fixture-frame',(),None) for i in range(n))
    return build_neighborhood_query(rows,query_record_id='0',target_axis=AXIS.features,radius=1.)

def model(width=8):
    return NativeSCRIModel(shared(width),AXIS,SpatialBudget(64,4097,68000000,2000000,2),'d'*64)

def test_large_graph_preserves_shared_identity_and_orthogonal_rank():
    m=model();q=query(16);z=m.forward_spatial((q,))
    assert z.shape==(1,1) and torch.isfinite(z).all()
    assert m.graph.incidence.in_features==128
    assert m.core.shared.backbone.incidence.in_features==4
    for name in ('feature','role','graph_token','layers','final_norm'):
        assert getattr(m.graph,name) is getattr(m.core.shared.backbone,name)
    assert m.graph.incidence is not m.core.shared.backbone.incidence
    loss=m.objective(z,[[2]],[[True]])['loss'];loss.backward()
    assert m.graph.incidence.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.shared.backbone.layers.parameters())
    assert not m.training_admitted

def test_roles_cover_every_shared_alias_across_three_widths():
    models=tuple(model(w) for w in (8,12,16))
    ledgers=[parameter_ledger(m,native_scri_roles(m,identifier_axis=ID_AXIS)) for m in models]
    family=width_family(*ledgers)
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in models)
    assert any(len(e.aliases)>1 for e in ledgers[0].parameters)

def test_targets_and_audit_are_excluded_and_malformed_topology_rejected():
    m=model();q=query();z=m.forward_spatial((q,))
    torch.testing.assert_close(z,m.forward_spatial((replace(q,audit={'secret_target':999}),)),rtol=0,atol=0)
    node=q.record.nodes[0]
    node=replace(node,measurements=tuple(replace(x,value=999,status='observed',available=True) if x.name=='target:0' else x for x in node.measurements))
    with pytest.raises(ValueError):m.forward_spatial((replace(q,record=replace(q.record,nodes=(node,*q.record.nodes[1:]))),))
    with pytest.raises(ValueError):m.forward_spatial((replace(q,record=replace(q.record,edges=q.record.edges[:-1])),))

def test_bounds_reject_before_encoder_and_boolean_limits():
    for field in ('max_cells','max_entities','max_attention_elements','max_condition_bytes','max_batch_size'):
        args=dict(max_cells=64,max_entities=4097,max_attention_elements=68000000,max_condition_bytes=2000000,max_batch_size=2)
        args[field]=True
        with pytest.raises(ValueError):SpatialBudget(**args)
    m=NativeSCRIModel(shared(),AXIS,SpatialBudget(4,17,2000,100000,1),'d'*64)
    with pytest.raises(ValueError,match='budget'):m.forward_spatial((query(8),))

def test_checkpoint_preflight_checks_aliases_before_mutation_and_restores_exactly():
    m=model();q=query();expected=m.forward_spatial((q,));state=copy.deepcopy(m.state_dict())
    original={n:p.detach().clone() for n,p in m.named_parameters()}
    bad=copy.deepcopy(state);bad['graph.layers.0.attention.output.weight']=bad['graph.layers.0.attention.output.weight'].clone()+1
    with pytest.raises(ValueError,match='alias'):m.load_state_dict(bad)
    assert all(torch.equal(original[n],p) for n,p in m.named_parameters())
    with pytest.raises(ValueError):m.load_state_dict(state,assign=True)
    clone=copy.deepcopy(m);clone.load_state_dict(state)
    torch.testing.assert_close(expected,clone.forward_spatial((q,)),rtol=0,atol=0)

def test_state_drift_and_invalid_orthogonality_fail():
    m=model();m.graph.incidence=torch.nn.Linear(128,8,bias=False)
    with pytest.raises(ValueError,match='identity'):m.get_extra_state()

def test_batched_padding_agrees_with_individual_predictions():
    m=model();a=query(4);b=query(8)
    batch=m.forward_spatial((a,b))
    torch.testing.assert_close(batch,torch.cat((m.forward_spatial((a,)),m.forward_spatial((b,)))),rtol=2e-5,atol=2e-6)
