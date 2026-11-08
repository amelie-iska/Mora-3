import copy
from dataclasses import replace
import pytest
import torch
from test_profile_decoder_cooperative import build as decoder_build
from test_native_tox21 import fixture, view
from test_tox21_training import equal


def build(device='cpu',size=1,changed=False):
    from mora3.model.tox21_cooperative import Tox21CooperativeModel, Tox21CooperativeTrainer
    from mora3.model.tox21_training import Tox21Batch
    t=decoder_build(device)
    m=Tox21CooperativeModel(t.model.native,t.model.annotation,t.residue_batch.feature_axis,
        t.model.profile.axes,codec=t.model.decoder.codec,max_tokens=64).to(device)
    v=view(fixture()[0])
    if changed:v=replace(v,provenance_bytes=v.provenance_bytes+b' ')
    return Tox21CooperativeTrainer(m,t.batch,t.pools,residue_batch=t.residue_batch,
        profile_batch=t.profile_batch,profile_batch_size=2,decoder_batch=t.decoder_batch,
        decoder_batch_size=2,tox21_batch=Tox21Batch((v,)),tox21_batch_size=size,
        batch_size=1,flow_batch_size=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_ninth_objective_reference_ownership_and_replay(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(8901);t=build(device)
    assert len(t.kinds)==9 and t.kinds[-1]=='tox21'
    assert t.model.tox21.shared is t.model.annotation
    assert t.model.tox21.backbone is t.model.native.response.backbone
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(list(t.model.parameters()))
    for kind in t.kinds[:-1]:assert t.step()['kind']==kind
    state=t.state_dict();ref=copy.deepcopy(t.model)
    opt=torch.optim.Adam(ref.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    opt.zero_grad();vs=t.tox21_batch.views
    loss=ref.tox21.objective(ref.tox21(vs),vs);loss.backward();opt.step()
    result=t.step();assert result['kind']=='tox21' and result['loss']==float(loss.detach())
    equal(t.model.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())
    assert t.model.tox21.output.weight.grad.abs().sum()>0
    assert t.model.annotation.payload.embedding.weight.grad.abs().sum()>0
    fresh=build(device);fresh.load_state_dict(state);assert fresh.step()==result
    equal(t.state_dict(),fresh.state_dict())


def test_binding_and_invalid_size_transactional():
    t=build();state=t.state_dict()
    for other in (build(size=2),build(changed=True)):
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(before,other.state_dict())
    for size in (0,-1,True):
        with pytest.raises(ValueError):build(size=size)
