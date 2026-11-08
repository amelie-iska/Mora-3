import copy
from dataclasses import replace
import pytest
import torch
from test_residue_cooperative_trainer import build as residue_build
from test_molecule_profile_model import view
from test_response_flow_trainer import equal
from mora3.model import molecule_profile_training as module


@pytest.fixture(autouse=True)
def deterministic():
    prior=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prior)


def build(device='cpu', *, size=1, changed=False):
    base=residue_build(device)
    views=(view(),view(modality='CellPainting'))
    if changed:
        views=(replace(views[0],target_bytes=views[0].target_bytes.replace(b'0.25',b'1.25')),views[1])
    m=module.MoleculeProfileCooperativeModel(base.model.native,base.model.annotation,
        base.residue_batch.feature_axis,{v.modality:v.axis for v in views}).to(device)
    return module.MoleculeProfileCooperativeTrainer(m,base.batch,base.pools,
        residue_batch=base.residue_batch,profile_batch=module.MoleculeProfileBatch(views),
        profile_batch_size=size,batch_size=1,flow_batch_size=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_seventh_objective_reference_adam_shared_gradients_and_resume(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(8201)
    t=build(device,size=2)
    assert t.kinds==('annotation','image','response','intervention','flow','residue','profile')
    m=t.model
    assert m.profile.shared is m.annotation
    assert m.profile.backbone is m.native.response.backbone is m.residue.shared.backbone
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len({id(p) for p in m.parameters()})==len({id(p) for p in ps})
    for kind in t.kinds[:-1]:assert t.step()['kind']==kind
    saved=t.state_dict()
    ref=copy.deepcopy(m);opt=torch.optim.Adam(ref.parameters(),lr=.001)
    opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    opt.zero_grad(set_to_none=True)
    z=ref.profile(t.profile_batch.views)
    expected=torch.stack([(p-p.new_full(p.shape,.25)).square().mean() for p in z]).mean()
    expected.backward();opt.step()
    result=t.step()
    assert result['loss']==float(expected.detach()) and result['examples']==2
    for (name,p),(other,q) in zip(m.named_parameters(),ref.named_parameters()):
        assert name==other and torch.equal(p,q),name
        if p.grad is None:assert q.grad is None
        else:assert torch.equal(p.grad,q.grad),name
    assert m.profile.output['L1000'].weight.grad.abs().sum()>0
    assert m.annotation.payload.embedding.weight.grad.abs().sum()>0
    assert m.native.response.backbone.feature.weight.grad.abs().sum()>0
    assert m.flow.output.weight.grad is None
    assert m.native.image.head.weight.grad is None
    end=t.state_dict();fresh=build(device,size=2);fresh.load_state_dict(saved)
    assert fresh.step()==result;equal(end,fresh.state_dict())
    tail=[t.step() for _ in range(14)]
    fresh.load_state_dict(end)
    assert [fresh.step() for _ in range(14)]==tail
    equal(t.state_dict(),fresh.state_dict())


def test_profile_schedule_wrap_and_binding_rejection_before_mutation():
    t=build();state=t.state_dict()
    for other in (build(size=2),build(changed=True)):
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(before,other.state_dict())
    seen=[]
    for _ in range(21):
        expected=None
        if t.steps % 7 == 6:
            selected=(t.profile_batch.views[(t.steps//7)%2],)
            expected=float(t.model.profile.objective(t.model.profile(selected),selected).detach())
        result=t.step()
        if result['kind']=='profile':
            assert result['loss']==expected
            seen.append(result['loss'])
    assert len(seen)==3 and all(torch.isfinite(torch.tensor(seen)))
    for bad in (0,-1,True,1.5):
        with pytest.raises(ValueError,match='profile batch size'):build(size=bad)
