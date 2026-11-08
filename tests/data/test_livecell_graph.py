from dataclasses import replace
import numpy as np
import pytest
import torch
from mora3.data.livecell import LiveCellExample
from mora3.data.livecell_graph import LiveCellGraphTask, audit_morphology_splits
from mora3.data.schema import GraphRecord
from mora3.tokenization.graph_payload import GraphPayloadCodec
from mora3.tokenization.shortest_path import ByteTokenizer


def example(**changes):
    x = LiveCellExample.from_coco(image=np.arange(30,dtype=np.uint8).reshape(5,6),
        image_info=dict(id=7,file_name='A172_Phase_C7_1_02d16h00m_2.tif',height=5,width=6),
        annotations=[dict(id=9,image_id=7,segmentation=[[1,1,4,1,4,3,1,3]])],
        split='train',source_sha256='a'*64)
    return replace(x,**changes)


def fields(node):
    return {m.name:m.value for m in node.measurements}


def test_mask_graph_has_directed_observation_edges_and_exact_bbox():
    t=LiveCellGraphTask(example(),'mask_to_image')
    g=t.input_graph()
    assert isinstance(g,GraphRecord)
    assert len(g.nodes)==2 and len(g.edges)==1
    assert (g.edges[0].source,g.edges[0].target,g.edges[0].kind,g.edges[0].directed)==(0,1,'contains_observed_cell',True)
    assert fields(g.nodes[1])==dict(x_min=1.,y_min=1.,x_max=4.,y_max=3.,vertex_count=4,component_count=1)
    assert fields(g.nodes[0])['pixel_size_um'] is None
    assert all(m.units=='pixel' for m in g.nodes[1].measurements[:4])
    assert GraphRecord.from_bytes(g.canonical_bytes())==g


def test_target_geometry_counts_and_provenance_never_enter_input_entities():
    a=LiveCellGraphTask(example(),'image_to_mask')
    b=LiveCellGraphTask(example(polygons=(((0,0,2,0,2,2),),((1,1,2,1,2,2),)),source_sha256='b'*64),'image_to_mask')
    assert a.input_graph().nodes==b.input_graph().nodes
    assert a.input_graph().edges==b.input_graph().edges==()
    assert len(a.input_graph().nodes)==1
    assert len(a.target_graph().nodes)==2 and len(b.target_graph().nodes)==3
    assert a.input_metadata()==b.input_metadata()
    codec=GraphPayloadCodec(ByteTokenizer(controls=('pad','node','edge')))
    assert codec.encode(a.input_graph())['entities']==codec.encode(b.input_graph())['entities']
    assert a.canonical_bytes()!=b.canonical_bytes()


def test_hidden_image_does_not_change_inverse_inputs():
    a=LiveCellGraphTask(example(),'mask_to_image')
    b=LiveCellGraphTask(example(pixels_hex='ff'*30),'mask_to_image')
    assert a.input_graph()==b.input_graph()
    assert a.input_metadata()==b.input_metadata()
    pa,pb=a.project(),b.project()
    assert torch.equal(pa['inputs']['polygons'][0][0],pb['inputs']['polygons'][0][0])
    assert not torch.equal(pa['targets']['image'],pb['targets']['image'])


def test_canonical_resume_checks_entire_source_task():
    t=LiveCellGraphTask(example(),'image_to_mask')
    u=LiveCellGraphTask.from_bytes(t.canonical_bytes())
    assert u.canonical_bytes()==t.canonical_bytes()
    t.assert_resume_matches(u)
    for other in (LiveCellGraphTask(example(),'mask_to_image'),LiveCellGraphTask(example(split='val'),'image_to_mask')):
        with pytest.raises(ValueError,match='resume'): t.assert_resume_matches(other)
    with pytest.raises(ValueError): LiveCellGraphTask.from_bytes(t.canonical_bytes().replace(b'"version":1',b'"version":2',1))


def test_source_parent_and_well_split_overlap_is_reported_without_reassignment():
    a=LiveCellGraphTask(example(),'image_to_mask')
    b=LiveCellGraphTask(example(image_id=8,file_name='A172_Phase_C7_2_03d00h00m_1.tif',split='test'),'image_to_mask')
    c=LiveCellGraphTask(example(image_id=9,file_name='A172_Phase_A7_2_03d00h00m_1.tif',split='val'),'image_to_mask')
    assert audit_morphology_splits([a,b,c])==[{'group':a.input_graph().record_id,'splits':['test','train']}]
    assert audit_morphology_splits([a,c])==[]
    assert b.example.split=='test'


def test_conflicting_same_image_snapshot_rejected():
    a=LiveCellGraphTask(example(),'image_to_mask')
    b=LiveCellGraphTask(example(pixels_hex='ff'*30),'image_to_mask')
    with pytest.raises(ValueError,match='conflicting'): audit_morphology_splits([a,b])


def test_holdouts_refused_for_training_and_unknown_direction_refused():
    for s in ('val','test'):
        with pytest.raises(ValueError,match='split'): LiveCellGraphTask(example(split=s),'image_to_mask').project(for_training=True)
    LiveCellGraphTask(example(),'image_to_mask').project(for_training=True)
    with pytest.raises(ValueError): LiveCellGraphTask(example(),'invented')


def test_empty_masks_preserve_image_input_schema_and_no_cells():
    a=LiveCellGraphTask(example(polygons=()),'image_to_mask')
    b=LiveCellGraphTask(example(),'image_to_mask')
    assert a.input_graph().nodes==b.input_graph().nodes
    assert len(a.target_graph().nodes)==1 and not a.target_graph().edges


def test_cpu_cuda_full_polygon_projection():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    for direction in ('image_to_mask','mask_to_image'):
        t=LiveCellGraphTask(example(),direction)
        a,b=t.project(),t.project(device='cuda')
        assert a['input_graph']==b['input_graph']
        where='targets' if direction=='image_to_mask' else 'inputs'
        assert torch.equal(a[where]['polygons'][0][0],b[where]['polygons'][0][0].cpu())


def test_multicomponent_geometry_keeps_all_vertices_and_does_not_infer_contacts():
    x=example(polygons=(((0,0,1,0,1,1),(3,2,5,2,5,4)),))
    t=LiveCellGraphTask(x,'mask_to_image')
    g=t.input_graph()
    assert fields(g.nodes[1])==dict(x_min=0.,y_min=0.,x_max=5.,y_max=4.,vertex_count=6,component_count=2)
    assert len(g.edges)==1
    assert sum(p.numel() for p in t.project()['inputs']['polygons'][0])==12


def test_existing_payload_encoder_features_and_gradients_exclude_hidden_topology():
    from mora3.model.payload import PayloadEncoder
    torch.manual_seed(26)
    encoder=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),8).double()
    a=LiveCellGraphTask(example(),'image_to_mask')
    b=LiveCellGraphTask(example(polygons=()),'image_to_mask')
    results=[]; gradients=[]
    for t in (a,b):
        g=t.input_graph()
        allowed={(role,i,m.name) for role,entities in (('node',g.nodes),('edge',g.edges))
                 for i,e in enumerate(entities) for m in e.measurements if m.available}
        result=encoder(g,input_fields=allowed,identifiers=torch.eye(len(g.nodes),dtype=torch.float64))
        results.append(result)
        gradients.append(torch.autograd.grad(result['features'].sum(),tuple(encoder.parameters())))
    for key in results[0]: torch.testing.assert_close(results[0][key],results[1][key],atol=0,rtol=0)
    for p,q in zip(*gradients): torch.testing.assert_close(p,q,atol=0,rtol=0)
