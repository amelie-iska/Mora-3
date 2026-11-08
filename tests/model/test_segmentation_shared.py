from dataclasses import replace
import pytest
import torch
from test_segmentation_inference import setup
from mora3.data import segmentation_shared as shared
from mora3.data.provenance import Provenance
from mora3.data.segmentation_measurements import measure_predicted_instances
from mora3.model.payload import PayloadEncoder
from mora3.model.tokengt import TokenGT
from mora3.tokenization.shortest_path import ByteTokenizer


def fixture(tmp_path):
    t,n,r=setup(tmp_path)
    labels=torch.tensor([[0,1,1,0],[0,1,2,0],[0,0,2,0]])
    m=measure_predicted_instances(labels,r.artifacts[0],checkpoint=n.identity,mask_asset='pred.pt')
    p=Provenance(t.source,t.revision,t.source_id,t.image_sha256,t.license)
    return r,{'image':dict(status='predicted',labels=labels,measurement=m)},(p,)


def test_geometry_equations_and_explicit_missing_modalities(tmp_path):
    r,p,prov=fixture(tmp_path)
    x=shared.project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    g=x['input_graph']; task=x['task']
    assert len(g.nodes)==3
    fields={m.name:m for m in g.nodes[1].measurements}
    assert fields['area_px'].value==3 and fields['area_px'].status=='predicted'
    assert fields['centroid_y'].value==pytest.approx(1/3)
    assert fields['centroid_x'].value==pytest.approx(4/3)
    assert fields['perimeter_px'].value==8
    assert len([e for e in g.edges if e.kind=='pixel_boundary_adjacency'])==2
    assert task.available_targets(g)==()
    assert x['modality_masks']=={'image_context':True,'predicted_geometry':True,'response':False}
    assert x['missing_context']['image']==('perturbation','expression')
    assert g.provenance==prov and g.leakage_keys[0].value==r.artifacts[0].parent_id
    assert not x['biological_validation']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_encoder_target_invariance_and_geometry_gradients(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    r,p,prov=fixture(tmp_path)
    a=shared.project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    hidden=replace(r.artifacts[0],artifact_id='secret',role='target',spatial_shape=(200,300))
    b=shared.project_native_geometry(replace(r,artifacts=r.artifacts+(hidden,)),dict(p,secret=object()),provenance=prov,requested_targets=('response',))
    assert a['input_graph']==b['input_graph']
    torch.manual_seed(61)
    enc=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),8).to(device)
    net=TokenGT(8,3,width=8,heads=2,layers=1).to(device)
    outputs=[]
    for x in (a,b):
        graph=enc.forward_task(x['input_graph'],task=x['task'],identifiers=torch.eye(3,device=device))
        outputs.append(net([graph])[0])
    assert torch.equal(*outputs)
    outputs[0].square().sum().backward()
    assert enc.embedding.weight.grad.abs().sum()>0
    # Native prediction is a discrete condition, never differentiated supervision.
    assert not p['image']['labels'].requires_grad


@pytest.mark.parametrize('change',['parent','checkpoint','geometry','role'])
def test_mismatched_prediction_binding_rejected(tmp_path,change):
    r,p,prov=fixture(tmp_path)
    m=p['image']['measurement']
    if change=='parent': m['artifact']=replace(m['artifact'],parent_id='wrong')
    if change=='checkpoint': m['artifact']=replace(m['artifact'],checkpoint='wrong')
    if change=='geometry': m['objects'][0]['area_px']=999
    if change=='role': m['artifact']=replace(m['artifact'],role='target')
    with pytest.raises(ValueError): shared.project_native_geometry(r,p,provenance=prov,requested_targets=('response',))


def test_empty_predictions_preserve_missingness_and_no_fake_object(tmp_path):
    r,p,prov=fixture(tmp_path)
    p['image']['labels'].zero_()
    p['image']['measurement']=measure_predicted_instances(p['image']['labels'],r.artifacts[0],checkpoint=r.head_checkpoint,mask_asset='empty.pt')
    x=shared.project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    assert len(x['input_graph'].nodes)==1
    assert x['modality_masks']['predicted_geometry']
    assert x['task'].available_targets(x['input_graph'])==()


def test_distinct_visible_sources_have_distinct_audit_record_ids(tmp_path):
    r,p,prov=fixture(tmp_path)
    a=shared.project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    other=replace(r,artifacts=(replace(r.artifacts[0],asset='other-source#image'),))
    b=shared.project_native_geometry(other,p,provenance=prov,requested_targets=('response',))
    assert a['input_graph'].record_id!=b['input_graph'].record_id


def test_unavailable_head_does_not_invent_geometry(tmp_path):
    r,p,prov=fixture(tmp_path)
    x=shared.project_native_geometry(replace(r,head_checkpoint=None),{},provenance=prov,requested_targets=('response',))
    assert not x['modality_masks']['predicted_geometry']
    assert len(x['input_graph'].nodes)==1
    assert x['task'].available_targets(x['input_graph'])==()
