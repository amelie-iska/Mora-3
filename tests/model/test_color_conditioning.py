"""Task-selected context must affect pixels without exposing hidden targets."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_segmentation_shared import fixture
from mora3.data.segmentation_shared import project_native_geometry
from mora3.model.color_semantic import ColorNuclearSemantic

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_context_reference_gradients_and_target_exclusion(tmp_path,device):
    from mora3.model.color_conditioning import ConditionedColorSemantic
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(17001)
    model=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=4).double().to(device)
    ref=copy.deepcopy(model)
    r,p,prov=fixture(tmp_path)
    x=project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    g,t=x['input_graph'],x['task']
    pixels=np.random.default_rng(17001).integers(0,256,(7,9,3),dtype=np.uint8)
    y=model(pixels,g,t)
    graph=ref.context.payload.forward_task(g,task=t,identifiers=torch.eye(3,4,dtype=torch.float64,device=device))
    latent=ref.context.graph([graph])[0][:,0]
    expected=ref.color(pixels)+(latent @ ref.offset.weight.T+ref.offset.bias).reshape(())
    torch.testing.assert_close(y,expected,atol=0,rtol=0)
    y.square().mean().backward();expected.square().mean().backward()
    for (name,a),(other,b) in zip(model.named_parameters(),ref.named_parameters()):
        assert name==other and a.grad is not None,name
        torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-10)
    assert model.context.graph.layers is model.color.backbone.layers
    assert model.context.graph.incidence is not model.color.backbone.incidence
    node=g.nodes[0]
    hidden=tuple(replace(m,value=999.,units='dimensionless',available=True,status='observed') if m.name.startswith('requested:') else m for m in node.measurements)
    assert torch.equal(y,model(pixels,replace(g,nodes=(replace(node,measurements=hidden),)+g.nodes[1:]),t))
    changed=tuple(replace(m,value=m.value+100) if m.name=='height' else m for m in node.measurements)
    assert not torch.equal(y,model(pixels,replace(g,nodes=(replace(node,measurements=changed),)+g.nodes[1:]),t))
    with pytest.raises(ValueError,match='unknown topology'):model(pixels,g,replace(t,topology_known=False))
    with pytest.raises(ValueError,match='identifier'):ConditionedColorSemantic(copy.deepcopy(model.color),id_width=2)(pixels,g,t)

def test_explicit_configuration():
    from mora3.model.color_conditioning import ConditionedColorSemantic
    with pytest.raises(ValueError):ConditionedColorSemantic(torch.nn.Linear(4,4),id_width=4)
    a=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=4)
    b=copy.deepcopy(a);b.load_state_dict(a.state_dict())
    state=a.get_extra_state();state['version']=2
    with pytest.raises(ValueError):a.set_extra_state(state)
