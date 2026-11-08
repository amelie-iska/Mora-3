import copy
from dataclasses import replace
import pytest
import torch
from test_bound_instance_inference import setup_case
from mora3.data import segmentation_shared as shared
from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from mora3.model.color_instance_inference import infer_nuclear_instances, instance_model_digest
from mora3.model.color_semantic_inference import image_condition_graph


def case():
    m,image,pixels,kw=setup_case()
    with torch.no_grad():
        m.color.head.projection.weight.zero_();m.color.head.projection.bias.zero_()
        m.offset.weight.zero_();m.offset.bias.zero_();m.color.head.projection.bias[2]=-1
    kw['expected_model_sha256']=instance_model_digest(m)
    result=infer_nuclear_instances(m,image,pixels,**kw)
    return image,result,image_condition_graph(image,pixel_sha256=kw['expected_pixel_sha256'])[0].provenance


def project(image,result,prov,targets=('response',)):
    return shared.project_bound_slot_geometry(image,result,provenance=prov,
        requested_targets=targets,max_mask_bytes=189)


def test_overlaps_local_identity_missingness_and_roundtrip():
    image,r,p=case();out=project(image,r,p);g=out['input_graph'];t=out['task']
    assert len(g.nodes)==3
    assert [dict((m.name,m.value) for m in n.measurements)['area_px'] for n in g.nodes[1:]]==[63,63]
    overlap=[e for e in g.edges if e.kind=='pixel_membership_overlap']
    assert [(e.source,e.target,e.measurements[0].value) for e in overlap]==[(1,2,63),(2,1,63)]
    assert out['slot_node_indices']=={1:1,2:2} and out['empty_slot_ids']==[3]
    assert not any(m.name in ('object_id','slot_id','parent_id','source_id') for n in g.nodes for m in n.measurements)
    assert t.available_targets(g)==() and out['modality_masks']['response'] is False
    assert g.leakage_keys[0].value==image.parent_id and g.provenance==p
    assert out['artifacts'][1]['evidence']=='model_predicted'
    assert out['artifacts'][1]['uncertainty']=='uncalibrated'
    assert not out['training_admission'] and not out['biological_validation']
    assert GraphRecord.from_bytes(g.canonical_bytes())==g
    assert ConditionalTask.from_bytes(t.canonical_bytes())==t
    t.input_fields(g)


@pytest.mark.parametrize('change',['mask','geometry','parent','checkpoint','seed','binding','threshold','quality','role','targets'])
def test_tampered_or_incompatible_predictions_refuse(change):
    image,r,p=case();r=copy.deepcopy(r);targets=('response',)
    if change=='mask':r['masks']=r['masks'].copy();r['masks'][0,0,0]=False
    if change=='geometry':r['measurements']['objects'][0]['area_px']=999
    if change=='parent':image=replace(image,parent_id='other')
    if change=='seed':object.__setattr__(r['measurements']['artifact'],'seed',True)
    if change in ('checkpoint','role'):
        r['measurements']['artifact']=replace(r['measurements']['artifact'],**{change:{'checkpoint':'wrong','seed':True,'role':'target'}[change]})
    if change=='binding':r['binding']['graph_sha256']='0'*64
    if change=='threshold':r['threshold']='argmax'
    if change=='quality':r['quality_accepted']=True
    if change=='targets':targets=('predicted_geometry',)
    with pytest.raises(ValueError):project(image,r,p,targets)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_payload_target_exclusion_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    from mora3.model.payload import PayloadEncoder
    from mora3.model.tokengt import TokenGT
    from mora3.tokenization.shortest_path import ByteTokenizer
    image,r,p=case();a=project(image,r,p);b=project(image,r,p,('expression',))
    torch.manual_seed(23901)
    enc=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),8).to(device)
    net=TokenGT(8,3,width=8,heads=2,layers=1).to(device)
    outputs=[]
    for x in (a,b):
        v=enc.forward_task(x['input_graph'],task=x['task'],identifiers=torch.eye(3,device=device))
        outputs.append(net([v])[0])
    assert torch.equal(*outputs)
    outputs[0].square().sum().backward()
    assert enc.embedding.weight.grad.abs().sum()>0
