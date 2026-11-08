import copy
from dataclasses import replace
import pytest
import torch
from test_annotation_multihead import model
from test_annotation_vocabulary import admitted
from mora3.model.annotation_trainer import AnnotationBatch, AnnotationTrainer


def batch(v=None):
    return AnnotationBatch((v or admitted(),), (True,), (False,))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_resume_and_batch_binding(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(4501)
    b=batch();t=AnnotationTrainer(model().to(device),b,lr=.003)
    t.step();state=t.state_dict();t.step();expected=t.state_dict()
    r=AnnotationTrainer(model().to(device),b,lr=.003);r.load_state_dict(state);r.step()
    for k,v in expected['model'].items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,r.state_dict()['model'][k])
    assert r.steps==2
    for v in [replace(admitted(),sequence='MAM'),replace(admitted(),target_bytes=b'{} '),
              replace(admitted(),provenance_bytes=admitted().provenance_bytes+b' ')]:
        other=AnnotationTrainer(model().to(device),batch(v),lr=.003)
        with pytest.raises(ValueError,match='binding'): other.load_state_dict(state)
    with pytest.raises(ValueError,match='binding'):
        AnnotationTrainer(model().to(device),b,lr=.004).load_state_dict(state)


def test_masks_and_failed_restore_do_not_mutate():
    with pytest.raises(ValueError): AnnotationBatch((admitted(),),(1,),(False,))
    with pytest.raises(ValueError): AnnotationBatch((admitted(),),(),(False,))
    t=AnnotationTrainer(model(),batch());s=t.state_dict();bad=copy.deepcopy(s)
    bad['model']['_extra_state']['layers']=999
    with pytest.raises(ValueError): t.load_state_dict(bad)
    assert t.model.get_extra_state()==s['model']['_extra_state']
    assert t.steps==0


def test_observed_update_and_checkpoint_optimizer_tamper():
    v=replace(admitted(),target_bytes=b'{"annotation_packet":{"uniProtKBCrossReferences":[{"database":"GO","id":"GO:0000001"}]}}')
    t=AnnotationTrainer(model(),batch(v));before=t.model.go.output.weight.detach().clone()
    loss=t.step()
    assert loss['GO']>0 and not torch.equal(before,t.model.go.output.weight)
    assert loss['EC']==0
    state=t.state_dict();state['optimizer']['param_groups'][0]['lr']=99
    with pytest.raises(ValueError,match='optimizer'):t.load_state_dict(state)
    changed=AnnotationTrainer(model(),AnnotationBatch((v,),(False,),(True,)))
    with pytest.raises(ValueError,match='binding'):changed.load_state_dict(t.state_dict())
