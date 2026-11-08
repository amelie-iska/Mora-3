import copy
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_cooperative import build as profile_build
from test_molecule_profile_model import view
from test_response_flow_trainer import equal
from mora3.model import profile_decoder as decoder


def build(device='cpu', *, max_tokens=64, changed=False):
    from mora3.model.profile_decoder_cooperative import DecoderBatch, DecoderCooperativeModel, DecoderCooperativeTrainer
    base=profile_build(device,size=2)
    views=(view('profile_to_molecule'),view('profile_to_molecule',modality='CellPainting'))
    if changed:views=(replace(views[0],target_bytes=b'{"canonical_isomeric_smiles":"CCCO"}'),views[1])
    codec=decoder.SelfiesCodec(('CCO','CCCO'))
    model=DecoderCooperativeModel(base.model.native,base.model.annotation,base.residue_batch.feature_axis,
        base.model.profile.axes,codec=codec,max_tokens=max_tokens).to(device)
    return DecoderCooperativeTrainer(model,base.batch,base.pools,residue_batch=base.residue_batch,
        profile_batch=base.profile_batch,profile_batch_size=2,decoder_batch=DecoderBatch(views),
        decoder_batch_size=2,batch_size=1,flow_batch_size=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_eighth_objective_reference_shared_optimizer_and_recovery(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.use_deterministic_algorithms(True);torch.manual_seed(8401)
    t=build(device)
    assert t.kinds[-2:]==('profile','molecule_decoder')
    assert t.model.decoder.profiles is t.model.profile
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(list(t.model.parameters()))
    for kind in t.kinds[:-1]:assert t.step()['kind']==kind
    saved=t.state_dict();ref=copy.deepcopy(t.model)
    opt=torch.optim.Adam(ref.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    loss=ref.decoder.objective(t.decoder_batch.views);loss.backward();opt.step()
    result=t.step();assert result['kind']=='molecule_decoder' and result['loss']==float(loss.detach())
    for p,q in zip(t.model.parameters(),ref.parameters()):assert torch.equal(p,q)
    for p in (t.model.decoder.rnn.weight_ih_l0,t.model.profile.profile_input['L1000'].weight,t.model.annotation.payload.embedding.weight):
        assert p.grad is not None and p.grad.abs().sum()>0
    fresh=build(device);fresh.load_state_dict(saved);assert fresh.step()==result
    equal(t.state_dict(),fresh.state_dict())


def test_admission_binding_and_transactional_rejection():
    from mora3.model.profile_decoder_cooperative import DecoderBatch
    v=view('profile_to_molecule');b=DecoderBatch((v,))
    assert b.binding()!=DecoderBatch((replace(v,target_bytes=b'{"canonical_isomeric_smiles":"CCCO"}'),)).binding()
    with pytest.raises(ValueError,match='inverse'):DecoderBatch((view(),))
    with pytest.raises(ValueError,match='admission'):DecoderBatch((replace(v,provenance_bytes=b'{}'),))
    t=build();state=t.state_dict()
    for other in (build(max_tokens=65),build(changed=True)):
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(before,other.state_dict())
