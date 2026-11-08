"""Thirteenth objective binds canonical GOA pool and independent cursors."""
import copy
import pytest
import torch
from test_goa_cooperative import build as model_build
from test_rfam_cooperative_training import build as trainer_build
from test_tox21_training import equal
from mora3.model import goa_cooperative as module


def build(tmp_path, device='cpu', size=1, absent=False):
    base=trainer_build(device)
    model,v=model_build(tmp_path,device)
    assert hasattr(module,'GOACooperativeTrainer'), 'native GOA cooperative scheduler missing'
    return module.GOACooperativeTrainer(model,base.batch,base.pools,
        residue_batch=base.residue_batch,profile_batch=base.profile_batch,profile_batch_size=2,
        decoder_batch=base.decoder_batch,decoder_batch_size=2,tox21_batch=base.tox21_batch,
        tox21_batch_size=1,batch_size=1,flow_batch_size=1,
        nucleotide_batch=base.nucleotide_batch,nucleotide_batch_size=1,
        rfam_batch=base.rfam_batch,rfam_prediction_batch_size=1,rfam_generation_batch_size=2,
        goa_packets=() if absent else (v,),goa_batch_size=size)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_goa_schedule_direct_adam_and_recovery(tmp_path,device):
    t=build(tmp_path,device)
    assert len(t.kinds)==13 and t.kinds[-1]=='goa'
    t.steps=12
    ref=copy.deepcopy(t.model);opt=torch.optim.Adam(ref.parameters(),lr=.001)
    opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
    opt.zero_grad();loss=ref.goa_objective(t.goa_packets);loss.backward();opt.step()
    saved=t.state_dict();result=t.step()
    assert result['kind']=='goa' and result['loss']==loss.item()
    equal(t.model.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())
    after=t.state_dict();t.load_state_dict(saved);assert t.step()==result;equal(after,t.state_dict())
    bad=copy.deepcopy(after);slot=next(iter(bad['optimizer']['state'].values()))
    slot['exp_avg'].flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='Adam'):t.load_state_dict(bad)
    equal(after,t.state_dict())


def test_absent_goa_and_invalid_size(tmp_path):
    t=build(tmp_path,absent=True)
    assert len(t.kinds)==12 and 'goa' not in t.kinds
    with pytest.raises(ValueError):build(tmp_path,size=True)
    with pytest.raises(ValueError):build(tmp_path,size=0)


def test_goa_binding_rejects_changed_size_transactionally(tmp_path):
    t=build(tmp_path);other=build(tmp_path,size=2);before=other.state_dict()
    with pytest.raises(ValueError,match='binding'):other.load_state_dict(t.state_dict())
    equal(before,other.state_dict())


def test_live_goa_payload_policy_rejected_before_update(tmp_path):
    t=build(tmp_path);before=t.state_dict()
    t.model.annotation.payload.max_tokens+=1
    with pytest.raises(ValueError,match='GOA.*policy'):t.step()
    with pytest.raises(ValueError,match='GOA.*policy'):t.load_state_dict(before)
    t.model.annotation.payload.max_tokens-=1
    equal(before,t.state_dict())
