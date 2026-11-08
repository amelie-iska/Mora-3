"""Full native geometry must share learned transforms without truncating ID rank."""
import copy
from dataclasses import replace
import pytest
import torch
from test_segmentation_shared import fixture
from mora3.data.segmentation_shared import project_native_geometry
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_conditioning import GeometryConditioner

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_full_geometry_reference_and_gradient_boundary(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(71)
    model=CooperativeModel(('A','B'),width=8).double().to(device)
    bridge=GeometryConditioner(model.response.backbone,id_width=4).double().to(device)
    r,p,prov=fixture(tmp_path)
    x=project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    g,t=x['input_graph'],x['task']
    actual=bridge(g,t)
    tensor=bridge.payload.forward_task(g,task=t,identifiers=torch.eye(3,4,dtype=torch.double,device=device))
    expected=bridge.graph([tensor])[0][:,0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert bridge.graph.layers is model.response.backbone.layers
    assert bridge.graph.feature is model.response.backbone.feature
    assert bridge.graph.incidence is not model.response.backbone.incidence
    actual.square().sum().backward()
    assert model.response.backbone.layers[0].attention.qkv.weight.grad.abs().sum()>0
    assert bridge.payload.embedding.weight.grad.abs().sum()>0
    assert bridge.graph.incidence.weight.grad.abs().sum()>0
    assert model.image.head.weight.grad is None
    assert model.response.response_head.weight.grad is None
    assert not p['image']['labels'].requires_grad
    # Changing unavailable target values/status must not affect conditions.
    n=g.nodes[0];ms=tuple(replace(m,value=999.,units='dimensionless',available=True,status='observed') if m.name.startswith('requested:') else m for m in n.measurements)
    changed=replace(g,nodes=(replace(n,measurements=ms),)+g.nodes[1:])
    assert torch.equal(bridge(changed,t),actual)
    with pytest.raises(ValueError,match='identifier'):
        GeometryConditioner(model.response.backbone,id_width=2).double().to(device)(g,t)


def test_configuration_and_checkpoint_binding():
    m=CooperativeModel(('A','B'),width=8)
    a=GeometryConditioner(m.response.backbone,id_width=4)
    b=GeometryConditioner(copy.deepcopy(m.response.backbone),id_width=4)
    b.load_state_dict(a.state_dict())
    with pytest.raises(ValueError):
        GeometryConditioner(copy.deepcopy(m.response.backbone),id_width=5).set_extra_state(a.get_extra_state())
    for invalid in [0,-1,True,1.5]:
        with pytest.raises(ValueError):GeometryConditioner(m.response.backbone,id_width=invalid)


def test_geometry_and_source_context_are_effective_conditions(tmp_path):
    torch.manual_seed(71)
    model=CooperativeModel(('A','B'),width=8)
    bridge=GeometryConditioner(model.response.backbone,id_width=4)
    r,p,prov=fixture(tmp_path)
    x=project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    g,t=x['input_graph'],x['task']; y=bridge(g,t)
    for index,name in [(1,'area_px'),(0,'height')]:
        node=g.nodes[index]
        ms=tuple(replace(m,value=m.value+100) if m.name==name else m for m in node.measurements)
        nodes=list(g.nodes);nodes[index]=replace(node,measurements=ms)
        assert not torch.equal(y,bridge(replace(g,nodes=tuple(nodes)),t))
    with pytest.raises(ValueError,match='unknown topology'):
        bridge(g,replace(t,topology_known=False))
