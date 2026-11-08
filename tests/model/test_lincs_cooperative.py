"""Native full-axis endpoints attached to the existing shared cooperative Adam."""
import copy
import pytest
import torch
from test_lincs_native import cases, project, api
from test_geometry_reconstruction_schedule import build as base_build
from mora3.model.profile_reconstruction import _digest


def build(tmp_path, cases, device='cpu', *, weights=None):
    from mora3.model.lincs_cooperative import LincsCooperativeModel, LincsCooperativeTrainer
    base = base_build(tmp_path, device)
    b = base.model
    views = tuple(project(api(), c) for c in cases.values())
    axes = {v.modality:c[7] for v,c in zip(views,cases.values())}
    torch.manual_seed(15001)
    m = LincsCooperativeModel(b.native,b.annotation,b.residue.head.feature_axis,b.profile.axes,
        codec=b.decoder.codec,max_tokens=64,
        annotation_registry=__import__('annotation_fixture').fixture_registry(),
        family_axis=b.rfam.family_axis,rfam_manifest_sha256='a'*64,goa_packets=base.goa_packets,
        reconstruction_dimension=2,reconstruction_horizon=3,reconstruction_hidden=5,
        reconstruction_source={'pool':'fixture'},geometry_id_width=16,lincs_axes=axes).to(device).double()
    return LincsCooperativeTrainer(m,base.batch,base.pools,
        reconstruction_batch=base.reconstruction_batch,reconstruction_batch_size=1,
        residue_batch=base.residue_batch,profile_batch=base.profile_batch,profile_batch_size=2,
        decoder_batch=base.decoder_batch,decoder_batch_size=2,tox21_batch=base.tox21_batch,
        tox21_batch_size=1,batch_size=1,flow_batch_size=1,
        nucleotide_batch=base.nucleotide_batch,nucleotide_batch_size=1,
        rfam_batch=base.rfam_batch,rfam_prediction_batch_size=1,rfam_generation_batch_size=2,
        goa_packets=base.goa_packets,goa_batch_size=1,
        geometry_tasks=base.geometry_tasks,image_requests=base.image_requests,
        admitted_task_sha256s=base.admitted_task_sha256s,prediction_assets=base.prediction_assets,
        native=base.geometry_native,geometry_batch_size=1,
        lincs_views=tuple(v for v in views if v.direction=='molecule_to_profile'),
        lincs_weights=weights or {'CellPainting':.25,'L1000':2.},lincs_batch_size=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_slots_direct_weighted_adam_and_exact_restore(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=build(tmp_path,cases,device)
    assert len(t.kinds)==17 and t.kinds[-2:]==('lincs_CellPainting','lincs_L1000')
    assert t.model.lincs.shared is t.model.annotation
    assert len(t.model.profile.axes['L1000'])==977
    assert len(t.model.lincs.axes['L1000'])==978
    ps=[p for g in t.optimizer.param_groups for p in g['params']]
    assert len(ps)==len(set(map(id,ps)))==len(tuple(t.model.parameters()))
    assert t.steps==0 and not t.optimizer.state
    archive=_digest(t.reconstruction_batch.state_dict())
    for cycle in range(2):
        for offset,modality in enumerate(('CellPainting','L1000')):
            t.steps=cycle*17+15+offset
            ref=copy.deepcopy(t.model)
            opt=torch.optim.Adam(ref.parameters(),lr=.001)
            opt.load_state_dict(copy.deepcopy(t.optimizer.state_dict()))
            selected=t.lincs_schedule.select(cycle*2+offset)[1]
            terms=[]
            for pred,v in zip(ref.lincs(selected),selected):
                a=v.native_profile();mask=torch.tensor(a.observed_mask,device=device)
                y=pred.new_tensor([x for x in a.values if x is not None])
                terms.append((pred[mask]-y).square().mean())
            loss=torch.stack(terms).mean()*t.lincs_schedule.weights[modality]
            loss.backward();opt.step()
            result=t.step()
            assert result['kind']=='lincs_'+modality and result['loss']==loss.item()
            assert _digest(t.model.state_dict())==_digest(ref.state_dict())
            assert _digest(t.optimizer.state_dict())==_digest(opt.state_dict())
            assert t.model.annotation.payload.embedding.weight.grad.abs().sum()>0
    assert _digest(t.reconstruction_batch.state_dict())==archive
    t.model.eval();t.model.decoder.train()
    state=t.state_dict();before=_digest(state)
    # Next native update, not an unrelated objective, after same-process restore.
    t.steps=49;expected=t.step();after=_digest(t.state_dict())
    t.load_state_dict(state);assert _digest(t.state_dict())==before
    t.steps=49;assert t.step()==expected
    assert _digest(t.state_dict())==after


@pytest.mark.parametrize('corruption',['weight','order','axis','alias','adam','gradient','rng'])
def test_checkpoint_rejection_is_transactional(tmp_path,cases,corruption):
    t=build(tmp_path,cases);t.steps=15;t.step()
    state=t.state_dict();before=_digest(state)
    if corruption=='weight':state['lincs_schedule']['weights']['L1000']=1.
    elif corruption=='order':state['lincs_schedule']['binding']='0'*64
    elif corruption=='axis':state['model']['lincs._extra_state']['axes']['L1000']=('bad',)
    elif corruption=='alias':
        state['model']['lincs.shared.payload.embedding.weight']=state['model']['lincs.shared.payload.embedding.weight'].clone()+1
    elif corruption=='adam':next(iter(state['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
    elif corruption=='gradient':
        k=next(k for k,v in state['gradients'].items() if v is not None)
        state['gradients'][k]=state['gradients'][k].to_sparse()
    elif corruption=='rng':state['cpu_rng']=torch.zeros(3,dtype=torch.uint8)
    with pytest.raises(ValueError):t.load_state_dict(state)
    assert _digest(t.state_dict())==before


@pytest.mark.parametrize('change',['weights','pool','size','schedule','model'])
def test_live_native_binding_rejected_before_mutation(tmp_path,cases,change):
    t=build(tmp_path,cases)
    if change=='weights':t.lincs_weights={'CellPainting':1.,'L1000':1.}
    elif change=='pool':t.lincs_views=t.lincs_views[::-1]
    elif change=='size':t.lincs_batch_size=2
    elif change=='schedule':t.kinds=t.kinds[::-1]
    elif change=='model':t.lincs_schedule.model=copy.deepcopy(t.model.lincs)
    before=_digest(dict(model=t.model.state_dict(),adam=t.optimizer.state_dict(),rng=torch.get_rng_state()))
    with pytest.raises(ValueError):t.step()
    assert _digest(dict(model=t.model.state_dict(),adam=t.optimizer.state_dict(),rng=torch.get_rng_state()))==before
    assert t.steps==0
