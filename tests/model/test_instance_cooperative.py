"""Native complete overlapping targets share the cooperative image/context encoder."""
import copy
import hashlib
import itertools
from dataclasses import replace
import pytest
import torch
from test_lincs_cooperative import build, cases
from test_color_supervised_task import source
from mora3.model.color_cooperative import ColorCooperativeModel
from mora3.model.color_instance_task import ColorInstanceTask
from mora3.model.color_semantic_inference import image_condition_graph


def attach(parent, **kwargs):
    from mora3.model.instance_cooperative import InstanceCooperativeModel
    return InstanceCooperativeModel(parent, slots=kwargs.pop('slots', 3),
        max_output_bytes=kwargs.pop('max_output_bytes', 1000000), **kwargs)


def base(tmp_path, cases, device='cpu'):
    torch.manual_seed(24101)
    return ColorCooperativeModel(build(tmp_path,cases,device).model,color_id_width=2)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('block',[None,3])
def test_complete_bce_all_active_gradients_and_shared_ownership(tmp_path,cases,device,block):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    parent=base(tmp_path,cases,device);before=copy.deepcopy(parent.state_dict())
    m=attach(parent)
    assert m.instance.color.pixels is parent.color.color.pixels
    for name in ('feature','role','graph_token','layers','final_norm'):
        assert getattr(m.instance.color.backbone,name) is getattr(parent.color.color.backbone,name)
        assert getattr(m.instance.context.graph,name) is getattr(parent.color.context.graph,name)
    assert m.instance.context.graph.incidence is parent.color.context.graph.incidence
    assert len(tuple(m.parameters()))==len(set(map(id,m.parameters())))
    for k,v in parent.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,before[k])
        else:assert v==before[k]
    s=source('ordered_binary_instances');s=replace(s,masks_encoded=s.masks_encoded*2,mask_sha256=s.mask_sha256*2)
    task=ColorInstanceTask(s,s.binding_sha256)
    actual=m.instance_objective(task,pixel_block_size=block)
    ref=copy.deepcopy(m.instance);pixels=s.image_array()
    g,t=image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    z=ref(pixels,g,t);targets=torch.zeros_like(z)
    targets[:2]=torch.tensor(s.target_arrays()[0],device=device)
    expected=torch.stack([(torch.nn.functional.softplus(z)-targets[list(p)]*z).mean()
        for p in itertools.permutations(range(3))]).min()
    torch.testing.assert_close(actual['loss'],expected,rtol=1e-12,atol=1e-12)
    for a,b in zip(torch.autograd.grad(actual['loss'],tuple(m.instance.parameters())),
                   torch.autograd.grad(expected,tuple(ref.parameters()))):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    assert not actual['training_admission'] and not t.available_targets(g)
    clone=copy.deepcopy(m);clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(clone.instance_objective(task)['loss'],actual['loss'])
    with pytest.raises(ValueError,match='admission|held-out'):task.require_training()


@pytest.mark.parametrize('case',['pixels','transform','incidence','parent_config','instance_config','task'])
def test_live_mutation_refuses_before_decode(tmp_path,cases,monkeypatch,case):
    m=attach(base(tmp_path,cases));s=source('ordered_binary_instances');t=ColorInstanceTask(s,s.binding_sha256)
    if case=='pixels':m.instance.color.pixels=copy.deepcopy(m.instance.color.pixels)
    if case=='transform':m.instance.color.backbone.layers=copy.deepcopy(m.instance.color.backbone.layers)
    if case=='incidence':m.instance.context.graph.incidence=copy.deepcopy(m.instance.context.graph.incidence)
    if case=='parent_config':m.parent.configuration['version']=99
    if case=='instance_config':m.instance.config['id_width']=99
    def forbidden(*args):raise AssertionError('decode before validation')
    monkeypatch.setattr(ColorInstanceTask,'input_pixels',forbidden)
    with pytest.raises(ValueError):m.instance_objective(None if case=='task' else t)


@pytest.mark.parametrize('case',['capacity','budget'])
def test_complete_set_preflight_before_decode(tmp_path,cases,monkeypatch,case):
    m=attach(base(tmp_path,cases),max_output_bytes=1 if case=='budget' else 1000000)
    s=source('ordered_binary_instances')
    if case=='capacity':s=replace(s,masks_encoded=s.masks_encoded*4,mask_sha256=s.mask_sha256*4)
    t=ColorInstanceTask(s,s.binding_sha256)
    monkeypatch.setattr(ColorInstanceTask,'input_pixels',lambda *a:pytest.fail('premature decode'))
    with pytest.raises(ValueError,match='capacity|output-byte'):m.instance_objective(t)


def test_explicit_extended_policy_and_strict_state(tmp_path,cases):
    parent=base(tmp_path,cases);m=attach(parent,slots=65,slot_policy='extended-max512-v2')
    assert m.instance.color.head.slot_policy=='extended-max512-v2'
    assert m.get_extra_state()['instance_head']['slots']==65
    with pytest.raises(ValueError,match='configuration'):m.set_extra_state({})
    other=attach(copy.deepcopy(parent))
    with pytest.raises((ValueError,RuntimeError)):other.load_state_dict(m.state_dict())
    with pytest.raises(ValueError):attach(None)
    with pytest.raises(ValueError):attach(copy.deepcopy(parent),slots=65)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_hidden_targets_do_not_enter_predictor(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    m=attach(base(tmp_path,cases,device));s=source('ordered_binary_instances')
    changed=replace(s,masks_encoded=s.masks_encoded*2,mask_sha256=s.mask_sha256*2)
    seen=[]
    hook=m.instance.register_forward_hook(lambda _m,_a,z:seen.append(z.detach().clone()))
    for v in (s,changed):m.instance_objective(ColorInstanceTask(v,v.binding_sha256))
    hook.remove();torch.testing.assert_close(*seen,rtol=0,atol=0)
