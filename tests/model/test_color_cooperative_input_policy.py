"""Explicit native extent is inherited, versioned, and never silently migrated."""
import copy
import numpy as np
import pytest
import torch
from test_lincs_cooperative import build, cases
from mora3.model.color_cooperative import ColorCooperativeModel
from mora3.model.instance_cooperative import InstanceCooperativeModel
from mora3.model.color_semantic import ColorNuclearSemantic
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic_inference import image_condition_graph
from test_color_supervised_task import source

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_opt_in_policy_direct_reference_all_gradients_and_inheritance(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(24701)
    m=ColorCooperativeModel(build(tmp_path,cases,device).model,color_id_width=2,input_policy='native-max1000-v1')
    assert m.get_extra_state()['version']==2
    assert m.get_extra_state()['native_color']==m.color.color.get_extra_state()
    s=source();g,t=image_condition_graph(s.image,pixel_sha256='a'*64)
    pixels=np.zeros((513,3,3),dtype=np.uint8)
    ref=copy.deepcopy(m.color)
    z=m.color(pixels,g,t)
    oracle=ref.color(pixels)+ref.offset(ref.context(g,t)).reshape(())
    torch.testing.assert_close(z,oracle,rtol=0,atol=0)
    for a,b in zip(torch.autograd.grad(z.square().mean(),tuple(m.color.parameters())),torch.autograd.grad(oracle.square().mean(),tuple(ref.parameters()))):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    inst=InstanceCooperativeModel(m,slots=3,max_output_bytes=1000000)
    assert inst.instance.color.input_policy=='native-max1000-v1'
    assert inst.instance.color.pixels is m.color.color.pixels
    assert inst.instance(pixels,g,t).shape==(3,513,3)
    clone=copy.deepcopy(inst);clone.load_state_dict(inst.state_dict())
    torch.testing.assert_close(clone.instance(pixels,g,t),inst.instance(pixels,g,t),rtol=0,atol=0)


def test_default_and_cross_policy_configuration_refusal(tmp_path,cases):
    p=build(tmp_path,cases).model
    old=ColorCooperativeModel(p,color_id_width=2)
    explicit=ColorCooperativeModel(copy.deepcopy(p),color_id_width=2,input_policy='native-max512-v1')
    assert old.get_extra_state()==explicit.get_extra_state()
    assert old.get_extra_state()['version']==1 and 'native_color' not in old.get_extra_state()
    new=ColorCooperativeModel(copy.deepcopy(p),color_id_width=2,input_policy='native-max1000-v1')
    for a,b in [(old,new),(new,old)]:
        with pytest.raises(ValueError,match='configuration'):a.set_extra_state(b.get_extra_state())
    with pytest.raises(ValueError,match='policy'):ColorCooperativeModel(p,color_id_width=2,input_policy='resize')
    calls=[];hook=old.color.color.pixels.register_forward_pre_hook(lambda *a:calls.append(True))
    with pytest.raises(ValueError):old.color.color(np.zeros((513,3,3),dtype=np.uint8))
    hook.remove();assert not calls
    calls=[];hook=new.color.color.pixels.register_forward_pre_hook(lambda *a:calls.append(True))
    with pytest.raises(ValueError):new.color.color(np.zeros((1001,3,3),dtype=np.uint8))
    hook.remove();assert not calls
