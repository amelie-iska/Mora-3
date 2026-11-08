import copy
import pytest
import torch
from test_annotation_cooperative import make
from test_physical_trainer import batch
from test_morphology import task as image
from test_sciplex_shared import task as response
from test_response_flow_trainer import equal
from mora3.model.flow_unified import FlowUnifiedModel, FlowUnifiedTrainer
from mora3.model.flow_matching import gaussian_path, velocity_loss


def build(device='cpu', **kwargs):
    from mora3.model.cooperative_training import CooperativeModel
    from mora3.model.physical_trainer import SharedPhysicalModel
    from mora3.model.annotation_vocabulary import fit_annotation_vocabulary
    from test_annotation_vocabulary import admitted
    native=CooperativeModel(('A','B'),width=4).to(device)
    ann=SharedPhysicalModel(fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2),
        ('GO:0000001',),('1.1.1.1',),parameter_axis=('mass_spectrometry_source_mass',),
        method_axis=('MALDI','ESI'),width=4,heads=2,layers=2,backbone=native.response.backbone).to(device)
    pools=dict(image=(image(),),response=(response(),response(counts=(3,7),cell='b')),
               intervention=(response('intervention'),))
    return FlowUnifiedTrainer(FlowUnifiedModel(native,ann).to(device),batch(),pools,
                              batch_size=1,flow_batch_size=1,**kwargs)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cycle_native_equations_flow_reference_and_resume(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(5701);tr=build(device);m=tr.model
    assert m.flow.native is m.native.response
    params=list(m.parameters());assert len(params)==len({id(p) for p in params})
    for kind in ('annotation','image','response','intervention'):
        expected=(m.annotation.objective(m.annotation(tr.batch.views),tr.batch.views,
            applicable=tr.batch.applicability(device))['total'] if kind=='annotation'
            else m.native.objective(kind,tr.pools[kind][:1]))
        r=tr.step();assert r['kind']==kind
        assert r['loss']==pytest.approx(float(expected.detach()),abs=1e-7)
        assert m.flow.output.weight.grad is None
    saved=tr.state_dict();reference=copy.deepcopy(m)
    y=torch.log1p(tr.pools['response'][0].project(device=device,dtype=next(m.parameters()).dtype)['targets']['expression'])[None]
    noise=torch.randn_like(y);time=torch.rand(1,device=device)
    state,velocity=gaussian_path(noise,y,time,sigma_min=.01)
    expected=velocity_loss(reference.flow(tr.pools['response'][:1],state,time),velocity,torch.ones_like(y,dtype=torch.bool))
    tr.load_state_dict(saved)
    r=tr.step();assert r['kind']=='flow';assert r['indices']==(0,)
    assert r['loss']==float(expected.detach())
    assert m.flow.output.weight.grad.abs().sum()>0
    assert m.native.response.backbone.feature.weight.grad.abs().sum()>0
    assert m.native.image.head.weight.grad is None
    assert m.native.response.response_head.weight.grad is None
    assert m.annotation.physical.unit.weight.grad is None
    end=tr.state_dict();fresh=build(device);fresh.load_state_dict(saved)
    assert fresh.step()==r;equal(fresh.state_dict(),end)
    for _ in range(5):r=tr.step()
    assert r['indices']==(1,)


def test_binding_rejects_path_batch_and_source_without_mutation():
    tr=build();state=tr.state_dict()
    for other in (build(sigma_min=.1),build(lr=.002)):
        before=copy.deepcopy(other.model.state_dict())
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(before,other.model.state_dict())
    for sigma in (float('nan'),1,-1):
        with pytest.raises(ValueError):build(sigma_min=sigma)


def test_flow_target_exclusion_and_pool_schedule_binding():
    tr=build();m=tr.model
    x=torch.ones(1,2);time=torch.tensor([.3])
    assert torch.equal(m.flow((response(),),x,time),
                       m.flow((response(counts=(99,88),cell='hidden'),),x,time))
    saved=tr.state_dict()
    for changes in (dict(flow_batch_size=2),dict(batch_size=2),
                    dict(response=(response(counts=(99,88)),))):
        pools=dict(tr.pools)
        options=dict(batch_size=1,flow_batch_size=1)
        if 'response' in changes:pools.update(changes)
        else:options.update(changes)
        other=FlowUnifiedTrainer(copy.deepcopy(m),tr.batch,pools,**options)
        before=copy.deepcopy(other.model.state_dict())
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(saved)
        equal(before,other.model.state_dict())
