import hashlib
import io
from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
import torch
from test_color_semantic_inference import fixture
from mora3.data.public_image_masks import PublicImageMasks
from mora3.model.color_semantic import ColorNuclearSemantic


def source(encoding='semantic_binary', split='test'):
    image,pixels=fixture()
    def png(a):
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');return b.getvalue()
    raw=png(pixels);mask=png((np.arange(63).reshape(7,9)%3==0).astype('uint8')*255)
    return PublicImageMasks(image,raw,(mask,),encoding,split,hashlib.sha256(raw).hexdigest(),(hashlib.sha256(mask).hexdigest(),))

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_source_bound_loss_reference_gradients_and_target_exclusion(device):
    from mora3.model.color_supervised_task import ColorSemanticTask
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(18801)
    s=source();task=ColorSemanticTask(s,expected_source_sha256=s.binding_sha256)
    m=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    result=task.evaluate(m)
    logits=m(s.image_array());target=torch.tensor(s.target_arrays()[0],device=device)
    oracle=(torch.nn.functional.softplus(logits)-target*logits).mean()
    torch.testing.assert_close(result['loss'],oracle)
    actual=torch.autograd.grad(result['loss'],tuple(m.parameters()))
    expected=torch.autograd.grad(oracle,tuple(m.parameters()))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
    assert result['source_sha256']==s.binding_sha256 and result['training_admission'] is False
    assert np.array_equal(task.input_pixels(),s.image_array())
    b=io.BytesIO();Image.fromarray((~s.target_arrays()[0]).astype('uint8')*255).save(b,format='PNG')
    raw=b.getvalue()
    changed=replace(s,masks_encoded=(raw,),mask_sha256=(hashlib.sha256(raw).hexdigest(),))
    assert changed.binding_sha256 != s.binding_sha256
    assert np.array_equal(ColorSemanticTask(changed,expected_source_sha256=changed.binding_sha256).input_pixels(),task.input_pixels())

@pytest.mark.parametrize('split',['train','test','validation','evaluation'])
def test_no_training_without_native_corpus_admission(split):
    from mora3.model.color_supervised_task import ColorSemanticTask
    s=source(split=split);t=ColorSemanticTask(s,expected_source_sha256=s.binding_sha256)
    with pytest.raises(ValueError,match='admission|held-out'):t.require_training()

@pytest.mark.parametrize('case',['binding','encoding','object','channels'])
def test_reject_inapplicable_sources(case):
    from mora3.model.color_supervised_task import ColorSemanticTask
    s=source('instance_id_map' if case=='encoding' else 'semantic_binary')
    if case=='object':s=replace(s,image=replace(s.image,context=(('object_type','whole_cell'),)))
    if case=='channels':s=replace(s,image=replace(s.image,channels=('blue','green','red')))
    with pytest.raises(ValueError):ColorSemanticTask(s,expected_source_sha256='0'*64 if case=='binding' else s.binding_sha256)
