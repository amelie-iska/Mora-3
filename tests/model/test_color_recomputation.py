"""Exact opt-in activation recomputation; synthetic pixels are test fixtures."""
import copy
import numpy as np
import pytest
import torch
from mora3.model.color_semantic import ColorNuclearSemantic,nuclear_semantic_objective
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic_inference import image_condition_graph

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('shape',[(17,19),(1,3)])
def test_recomputation_preserves_outputs_all_gradients_state_and_rng(device,shape):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(21501)
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    pixels=np.random.default_rng(21501).integers(0,256,(*shape,3),dtype=np.uint8)
    target=torch.arange(shape[0]*shape[1],device=device).reshape(shape)%2==0
    state=copy.deepcopy(model.state_dict());outputs=[];gradients=[]
    for flag in (False,True):
        model.zero_grad(set_to_none=True);rng=torch.get_rng_state().clone()
        logits=model(pixels,recompute_pixels=flag)
        loss=nuclear_semantic_objective(logits,target)
        loss.backward()
        assert torch.equal(torch.get_rng_state(),rng)
        outputs.append(logits.detach());gradients.append([p.grad.clone() for p in model.parameters()])
        for k,v in state.items():
            if torch.is_tensor(v):assert torch.equal(v,model.state_dict()[k])
            else:assert v==model.state_dict()[k]
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_recomputation_reduces_saved_bytes_and_inference_falls_back(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device)
    pixels=np.zeros((32,33,3),dtype=np.uint8);totals=[]
    for flag in (False,True):
        sizes=[]
        def pack(t):sizes.append(t.numel()*t.element_size());return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            model(pixels,recompute_pixels=flag).sum().backward()
        totals.append(sum(sizes));model.zero_grad(set_to_none=True)
    assert totals[1]<totals[0]/2
    calls=[]
    handle=model.pixels.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode():
        a=model(pixels);b=model(pixels,recompute_pixels=True)
    handle.remove()
    assert len(calls)==2
    torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('flag',[None,1,'yes'])
def test_recompute_requires_boolean(flag):
    with pytest.raises(ValueError,match='boolean'):
        ColorNuclearSemantic(4,alpha_policy='require-opaque')(np.zeros((3,3,3),dtype=np.uint8),recompute_pixels=flag)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_conditioned_recomputation_preserves_context_gradients_and_target_exclusion(tmp_path,device):
    from dataclasses import replace
    from test_segmentation_shared import fixture
    from mora3.data.segmentation_shared import project_native_geometry
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(21501)
    model=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=4).double().to(device)
    r,p,prov=fixture(tmp_path)
    x=project_native_geometry(r,p,provenance=prov,requested_targets=('response',))
    g,t=x['input_graph'],x['task'];pixels=np.zeros((9,11,3),dtype=np.uint8)
    results=[];gradients=[]
    for flag in (False,True):
        model.zero_grad(set_to_none=True)
        y=model(pixels,g,t,recompute_pixels=flag);y.square().mean().backward()
        results.append(y.detach());gradients.append([p.grad.clone() for p in model.parameters()])
    torch.testing.assert_close(*results,rtol=0,atol=0)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)
    node=g.nodes[0]
    hidden=tuple(replace(m,value=999.,units='dimensionless',available=True,status='observed') if m.name.startswith('requested:') else m for m in node.measurements)
    poisoned=replace(g,nodes=(replace(node,measurements=hidden),)+g.nodes[1:])
    torch.testing.assert_close(results[1],model(pixels,poisoned,t,recompute_pixels=True),rtol=0,atol=0)
