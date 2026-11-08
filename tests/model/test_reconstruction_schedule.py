import copy
import pytest
import torch
from test_reconstruction_cooperative import build as model_build
from test_goa_cooperative_training import build as trainer_build
from test_tox21_training import equal
from mora3.model import reconstruction_cooperative as module
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.profile_reconstruction import FixedProfileReconstruction


def build(tmp_path, device='cpu', size=1):
    assert hasattr(module,'ReconstructionCooperativeTrainer'), 'native reconstruction scheduler missing'
    base=trainer_build(tmp_path,device)
    m,v=model_build(tmp_path,device)
    # Fixture models use different widths/precision; source pools stay unchanged.
    policy=ProfileGFNPolicy(m.profile,dimension=2,horizon=3,hidden=5,source_binding={'pool':'fixture'})
    fixed=FixedProfileReconstruction(m.reconstruction,policy,base.decoder_batch.views,seed=12501)
    t=module.ReconstructionCooperativeTrainer(m,base.batch,base.pools,
        reconstruction_batch=fixed,reconstruction_batch_size=size,
        residue_batch=base.residue_batch,profile_batch=base.profile_batch,profile_batch_size=2,
        decoder_batch=base.decoder_batch,decoder_batch_size=2,tox21_batch=base.tox21_batch,
        tox21_batch_size=1,batch_size=1,flow_batch_size=1,
        nucleotide_batch=base.nucleotide_batch,nucleotide_batch_size=1,
        rfam_batch=base.rfam_batch,rfam_prediction_batch_size=1,rfam_generation_batch_size=2,
        goa_packets=(v,),goa_batch_size=1)
    return t


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_scheduler_direct_adam_and_independent_rows(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=build(tmp_path,device)
    assert len(t.kinds)==14 and t.kinds[-2:]==('goa','profile_reconstruction')
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(list(t.model.parameters()))
    for cycle in range(3):
        t.steps=cycle*14+13
        indices=((cycle % len(t.decoder_batch.views)),)
        ref=copy.deepcopy(t.model);opt=torch.optim.Adam(ref.parameters(),lr=.001)
        opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
        opt.zero_grad();loss=ref.reconstruction_objective(t.reconstruction_batch,indices=indices)
        loss.backward();opt.step()
        got=t.step()
        assert got['kind']=='profile_reconstruction' and got['examples']==1
        assert got['loss']==loss.item()
        equal(t.model.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())


@pytest.mark.parametrize('size',[0,-1,True])
def test_invalid_size(tmp_path,size):
    with pytest.raises(ValueError):build(tmp_path,size=size)


def test_live_fixed_archive_corruption_rejected_before_update(tmp_path):
    t=build(tmp_path);before=copy.deepcopy(t.model.state_dict())
    t.reconstruction_batch._latents[0,0]+=1
    with pytest.raises(ValueError,match='latent'):t.step()
    equal(before,t.model.state_dict());assert t.steps==0


def test_live_reconstruction_schedule_size_rejected(tmp_path):
    t=build(tmp_path);before=copy.deepcopy(t.model.state_dict())
    t.reconstruction_batch_size=2
    with pytest.raises(ValueError,match='schedule'):t.step()
    equal(before,t.model.state_dict());assert t.steps==0
