import copy
from dataclasses import replace
import pytest
import torch
from test_annotation_vocabulary import admitted
from mora3.model.annotation_vocabulary import fit_annotation_vocabulary
from mora3.model.annotation_trainer import AnnotationTrainer
from mora3.model.physical_trainer import PhysicalTrainingBatch, SharedPhysicalModel


def view():
    return replace(admitted(),target_bytes=b'{"annotation_packet":{"comments":[{"commentType":"MASS SPECTROMETRY","molWeight":42,"method":"MALDI","molecule":"chain"}]}}')


def model(device='cpu',methods=('MALDI','ESI')):
    return SharedPhysicalModel(fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2),
        ('GO:0000001',),('1.1.1.1',),parameter_axis=('mass_spectrometry_source_mass',),
        method_axis=methods,width=4,heads=1,layers=1).to(device)


def batch(v=None,active=True):
    return PhysicalTrainingBatch((v or view(),),(False,),(False,),
        parameter_axis=('mass_spectrometry_source_mass',),physical_applicable=(active,))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reference_update_resume_and_exclusion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(4901)
    m=model(device);b=batch();t=AnnotationTrainer(m,b)
    z=m(b.views);a=b.applicability(device)
    loss=m.objective(z,b.views,applicable=a)
    expected=m.physical.objective(z['physical'],b.physical.targets())
    torch.testing.assert_close(loss['total'],expected)
    mutated=replace(view(),target_bytes=view().target_bytes.replace(b'42',b'99').replace(b'chain',b'other').replace(b'MALDI',b'ESI'))
    other=m((mutated,))
    for k in z['physical']:assert torch.equal(z['physical'][k],other['physical'][k])
    t.step();s=t.state_dict();t.step();end=t.state_dict()
    r=AnnotationTrainer(model(device),b);r.load_state_dict(s);r.step()
    for k,v in end['model'].items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,r.state_dict()['model'][k])
    assert r.steps==2
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.backbone.parameters())
    for bad in (batch(mutated),batch(active=False)):
        with pytest.raises(ValueError,match='binding'):AnnotationTrainer(model(device),bad).load_state_dict(s)
    with pytest.raises(ValueError,match='binding'):
        AnnotationTrainer(model(device,('ESI','MALDI')),b).load_state_dict(s)


def test_inactive_zero_and_forged_binding_rejected():
    m=model();b=batch(active=False);z=m(b.views)
    loss=m.objective(z,b.views,applicable=b.applicability('cpu'))
    loss['total'].backward()
    assert loss['physical'].item()==0
    assert m.physical.unit.weight.grad.abs().sum()==0
    with pytest.raises(ValueError,match='Boolean'):batch(active=1)
    a=batch();app=a.applicability('cpu');app['physical']=batch(replace(view(),sequence='MAM')).physical
    with pytest.raises(ValueError,match='source'):
        m.objective(m(a.views),a.views,applicable=app)
