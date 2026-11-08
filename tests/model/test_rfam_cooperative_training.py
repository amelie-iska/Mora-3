import copy
import pytest
import torch
from mora3.model import nucleotide_cooperative as module
from mora3.model.rfam_family_training import RfamFamilyBatch
from test_rfam_cooperative import build as model_build
from test_nucleotide_cooperative import build as trainer_build
from test_rfam_family_objective import view
from test_tox21_training import equal


def build(device='cpu', prediction_size=1, generation_size=2, changed=False):
    assert hasattr(module,'RfamCooperativeTrainer'), 'native twelve-objective scheduler missing'
    base=trainer_build(device)
    p=(view(),view(seq='UU'),view(seq='NN'))
    if changed:p=(view(seq='AA'),)+p[1:]
    return module.RfamCooperativeTrainer(model_build(device),base.batch,base.pools,
        residue_batch=base.residue_batch,profile_batch=base.profile_batch,profile_batch_size=2,
        decoder_batch=base.decoder_batch,decoder_batch_size=2,tox21_batch=base.tox21_batch,
        tox21_batch_size=1,batch_size=1,flow_batch_size=1,
        nucleotide_batch=base.nucleotide_batch,nucleotide_batch_size=1,
        rfam_batch=RfamFamilyBatch(p,(view('generation'),view('generation',seq='UU'),view('generation',seq='NN'))),
        rfam_prediction_batch_size=prediction_size,rfam_generation_batch_size=generation_size)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_twelve_objectives_reference_adam_and_recovery(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(10201);torch.backends.cudnn.allow_tf32=False
    t=build(device)
    assert len(t.kinds)==12 and t.kinds[-2:]==('rfam_prediction','rfam_generation')
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(list(t.model.parameters()))
    for kind in t.kinds[:-2]:assert t.step()['kind']==kind
    for direction in ('prediction','generation'):
        ref=copy.deepcopy(t.model);opt=torch.optim.Adam(ref.parameters(),lr=.001)
        opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
        vs=getattr(t.rfam_batch,direction)[:getattr(t,'rfam_'+direction+'_batch_size')]
        opt.zero_grad();loss=getattr(ref.rfam,direction+'_objective')(vs);loss.backward();opt.step()
        saved=t.state_dict();got=t.step()
        assert got['kind']=='rfam_'+direction and got['loss']==loss.item()
        equal(t.model.state_dict(),ref.state_dict());equal(t.optimizer.state_dict(),opt.state_dict())
        expected=t.state_dict();t.load_state_dict(saved);assert t.step()==got
        equal(expected,t.state_dict())
    # Independent batch sizes/cursors wrap without dropping the final source row.
    t.steps=22
    loss,count=t.additional_objective('rfam_prediction');assert count==1
    assert loss.item()==t.model.rfam.prediction_objective((t.rfam_batch.prediction[1],)).item()
    t.steps=23
    loss,count=t.additional_objective('rfam_generation');assert count==2
    assert loss.item()==t.model.rfam.generation_objective((t.rfam_batch.generation[2],t.rfam_batch.generation[0])).item()


@pytest.mark.parametrize('kwargs',[{'prediction_size':0},{'generation_size':True},{'prediction_size':-1}])
def test_invalid_batch_size(kwargs):
    with pytest.raises(ValueError):build(**kwargs)


def test_changed_pool_or_size_rejected_before_restore_mutation():
    t=build();saved=t.state_dict()
    for other in (build(changed=True),build(prediction_size=2),build(generation_size=1)):
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(saved)
        equal(before,other.state_dict())


def test_runtime_precision_change_rejected_without_update():
    t=build();before=t.state_dict();old=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=not old
        with pytest.raises(ValueError,match='policy'):t.step()
        with pytest.raises(ValueError,match='policy'):t.load_state_dict(before)
        equal(before,t.state_dict())
    finally:torch.backends.cudnn.allow_tf32=old


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('damage',['nan','shape','negative','step','extra','missing'])
def test_malformed_adam_rejected_transactionally(damage, device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=build(device);t.step();before=t.state_dict();bad=copy.deepcopy(before)
    slot=next(iter(bad['optimizer']['state'].values()))
    if damage=='nan':slot['exp_avg'].flatten()[0]=float('nan')
    elif damage=='shape':slot['exp_avg']=torch.zeros(123)
    elif damage=='negative':slot['exp_avg_sq'].flatten()[0]=-1
    elif damage=='step':slot['step']=torch.tensor(999.)
    elif damage=='extra':slot['unexpected']=torch.tensor(0.)
    else:del slot['exp_avg']
    with pytest.raises(ValueError,match='Adam'):t.load_state_dict(bad)
    equal(before,t.state_dict())
