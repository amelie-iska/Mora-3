import copy
import pytest
import torch
from test_tox21_cooperative import build as base_build
from test_native_nucleotide_prior import view, row
from test_tox21_training import equal


def build(device='cpu', size=1, sequence='aCTuRnx'):
    from mora3.model.nucleotide_cooperative import NucleotideBatch, NucleotideCooperativeModel, NucleotideCooperativeTrainer
    t=base_build(device)
    m=NucleotideCooperativeModel(t.model.native,t.model.annotation,t.residue_batch.feature_axis,
        t.model.profile.axes,codec=t.model.decoder.codec,max_tokens=64).to(device)
    return NucleotideCooperativeTrainer(m,t.batch,t.pools,
        residue_batch=t.residue_batch,profile_batch=t.profile_batch,profile_batch_size=2,
        decoder_batch=t.decoder_batch,decoder_batch_size=2,tox21_batch=t.tox21_batch,
        tox21_batch_size=1,batch_size=1,flow_batch_size=1,
        nucleotide_batch=NucleotideBatch((view(row(sequence)),)),nucleotide_batch_size=size)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_tenth_objective_reference_shared_ownership_and_recovery(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(9401);torch.backends.cudnn.allow_tf32=False
    t=build(device)
    assert len(t.kinds)==10 and t.kinds[-1]=='nucleotide'
    assert t.model.nucleotide.shared is t.model.annotation
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(list(t.model.parameters()))
    for kind in t.kinds[:-1]:assert t.step()['kind']==kind
    state=t.state_dict();ref=copy.deepcopy(t.model)
    opt=torch.optim.Adam(ref.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    opt.zero_grad();loss=ref.nucleotide.objective(t.nucleotide_batch.views);loss.backward();opt.step()
    result=t.step();assert result['kind']=='nucleotide' and result['loss']==float(loss.detach())
    equal(t.model.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())
    assert t.model.nucleotide.rnn.weight_ih_l0.grad.abs().sum()>0
    assert t.model.annotation.payload.embedding.weight.grad.abs().sum()>0
    fresh=build(device);fresh.load_state_dict(state);assert fresh.step()==result
    equal(t.state_dict(),fresh.state_dict())


def test_batch_binding_and_transactional_mismatch():
    from mora3.model.nucleotide_cooperative import NucleotideBatch
    from test_nucleotide_decoder import view as unadmitted
    with pytest.raises(ValueError):NucleotideBatch((unadmitted(),))
    with pytest.raises(ValueError):NucleotideBatch(())
    t=build();state=t.state_dict()
    for other in (build(size=2),build(sequence='U'*9001)):
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(before,other.state_dict())
    for size in (0,-1,True):
        with pytest.raises(ValueError):build(size=size)


def test_batch_rejects_changed_evidence_roles():
    import json
    from dataclasses import replace
    from mora3.model.nucleotide_cooperative import NucleotideBatch
    v=view();p=v.provenance();p['paired_cellular_observation']=True
    with pytest.raises(ValueError):
        NucleotideBatch((replace(v,provenance_bytes=json.dumps(p).encode()),))


def test_precision_policy_binds_checkpoint():
    old=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=False;t=build();state=t.state_dict()
        torch.backends.cudnn.allow_tf32=True;other=build()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
    finally:torch.backends.cudnn.allow_tf32=old
