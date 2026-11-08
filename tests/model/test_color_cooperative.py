import copy
import pytest
import torch
from test_lincs_cooperative import build, cases
from test_color_supervised_task import source
from mora3.model.color_supervised_task import ColorSemanticTask

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_native_attachment_reference_gradients_and_unique_ownership(tmp_path,cases,device):
    from mora3.model.color_cooperative import ColorCooperativeModel
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    parent=build(tmp_path,cases,device).model
    before=copy.deepcopy(parent.configuration)
    model=ColorCooperativeModel(parent,color_id_width=2)
    assert parent.configuration==before
    for name in ('feature','role','graph_token','layers','final_norm'):
        assert getattr(model.color.color.backbone,name) is getattr(parent.annotation.backbone,name)
        assert getattr(model.color.context.graph,name) is getattr(parent.annotation.backbone,name)
    ps=tuple(model.parameters());assert len(ps)==len(set(map(id,ps)))
    s=source();t=ColorSemanticTask(s,s.binding_sha256)
    result=model.color_objective(t)
    oracle=t.evaluate_conditioned(model.color)
    torch.testing.assert_close(result['loss'],oracle['loss'])
    active=tuple(model.color.parameters())
    a=torch.autograd.grad(result['loss'],active)
    b=torch.autograd.grad(oracle['loss'],active)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    assert all(torch.isfinite(x).all() for x in a)
    assert model.color.color.head.weight.dtype==parent.annotation.backbone.feature.weight.dtype
    assert model.color.color.head.weight.device==parent.annotation.backbone.feature.weight.device
    with pytest.raises(ValueError,match='admission|held-out'):t.require_training()
    clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.color_objective(t)['loss'],result['loss'])
    with pytest.raises(ValueError,match='configuration'):model.set_extra_state({})
    changed=copy.deepcopy(model.state_dict())
    changed['_extra_state']['color']['id_width']=3
    with pytest.raises(ValueError,match='configuration'):clone.load_state_dict(changed)
    with pytest.raises(ValueError,match='configuration'):clone.load_state_dict(parent.state_dict())

@pytest.mark.parametrize('bad',['parent','rank','alias','task'])
def test_refuse_invalid_attachment_or_live_alias(tmp_path,cases,bad):
    from mora3.model.color_cooperative import ColorCooperativeModel
    parent=build(tmp_path,cases).model
    if bad in ('parent','rank'):
        with pytest.raises(ValueError):ColorCooperativeModel(None if bad=='parent' else parent,color_id_width=0 if bad=='rank' else 2)
        return
    m=ColorCooperativeModel(parent,color_id_width=2)
    s=source();task=ColorSemanticTask(s,s.binding_sha256)
    if bad=='alias':m.color.color.backbone.layers=copy.deepcopy(m.color.color.backbone.layers)
    with pytest.raises(ValueError):m.color_objective(None if bad=='task' else task)
