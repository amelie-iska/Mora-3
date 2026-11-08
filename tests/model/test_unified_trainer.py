import copy
import pytest
import torch
from test_annotation_cooperative import make
from test_physical_trainer import batch
from test_morphology import task as image
from test_sciplex_shared import task as response
from mora3.model.unified_trainer import UnifiedModel, UnifiedTrainer


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_equations_gradients_schedule_and_all_state_resume(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    from mora3.model.cooperative_training import CooperativeModel
    from mora3.model.physical_trainer import SharedPhysicalModel
    from test_annotation_vocabulary import admitted
    from mora3.model.annotation_vocabulary import fit_annotation_vocabulary
    torch.manual_seed(5101)
    native=CooperativeModel(('A','B'),width=4).to(device)
    ann=SharedPhysicalModel(fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2),('GO:0000001',),('1.1.1.1',),parameter_axis=('mass_spectrometry_source_mass',),method_axis=('MALDI','ESI'),width=4,heads=2,layers=2,backbone=native.response.backbone).to(device)
    m=UnifiedModel(native,ann);b=batch()
    pools={'image':(image(),),'response':(response(),),'intervention':(response('intervention'),)}
    t=UnifiedTrainer(m,b,pools,batch_size=1)
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    for i,kind in enumerate(('annotation','image','response','intervention')):
        expected=(ann.objective(ann(b.views),b.views,applicable=b.applicability(device))['total'] if kind=='annotation' else native.objective(kind,pools[kind]))
        result=t.step()
        assert result['kind']==kind
        assert result['loss']==pytest.approx(float(expected.detach()),abs=1e-7)
        assert ann.backbone.feature.weight.grad.abs().sum()>0
        assert (ann.physical.unit.weight.grad is not None)==(kind=='annotation')
        assert (native.image.head.weight.grad is not None)==(kind=='image')
    s=t.state_dict();t.step();end=t.state_dict()
    t.load_state_dict(s);t.step()
    def eq(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:eq(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            for x,y in zip(a,b):eq(x,y)
        else:assert a==b
    eq(end,t.state_dict())
    other=UnifiedTrainer(m,b,pools,batch_size=2)
    before=copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError,match='binding'):other.load_state_dict(s)
    eq(before,m.state_dict())
    with pytest.raises(ValueError):UnifiedTrainer(m,b,{'image':(image('test'),)},batch_size=1)


def test_reject_unshared_backbone():
    a,_=make();_,b=make()
    with pytest.raises(ValueError,match='shared'):UnifiedModel(a,b)
