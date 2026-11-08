"""Schedule identity and all-head runtime continuation, without source joins."""
import copy
import pytest
import torch
from test_segmentation_inference import setup
from test_sciplex_shared import task as response_task
from test_geometry_pixel_trainer import equal, deterministic
from mora3.model import cooperative_training as module

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_joint_scheduler_exact_continuation_and_binding(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    task,native,request=setup(tmp_path,device)
    model=module.GeometryCooperativeModel(('A','B'),width=8,id_width=16).to(device)
    pools={'image':(task,), 'response':(response_task(),),
           'intervention':(response_task('intervention'),)}
    trainer=module.GeometryCooperativeTrainer(model,pools,image_requests=(request,),
        native=native,batch_size=1,prediction_asset='declared.pt')
    assert [trainer.step()['kind'] for _ in range(3)]==list(module.KINDS)
    state=trainer.snapshot(); original=copy.deepcopy(state)
    expected=[trainer.step() for _ in range(3)]; final=trainer.snapshot()
    trainer.restore(state)
    assert [trainer.step() for _ in range(3)]==expected
    equal(final,trainer.snapshot());equal(state,original)
    for patch in [dict(binding={}),dict(steps=-1),dict(steps=True)]:
        bad=copy.deepcopy(state);bad.update(patch);before=trainer.snapshot()
        with pytest.raises(ValueError):trainer.restore(bad)
        equal(before,trainer.snapshot())
    changed=module.GeometryCooperativeTrainer(copy.deepcopy(model),pools,
        image_requests=(request,),native=native,batch_size=2,prediction_asset='declared.pt')
    with pytest.raises(ValueError):changed.restore(state)
    # Mutation of caller-owned containers cannot silently change the schedule.
    pools['response']=()
    assert trainer.binding['pools']['response']
    assert trainer.step()['kind']=='image'


def test_modality_batches_cover_complete_pools_without_image_accumulation():
    pools={'image':tuple(range(7)), 'response':tuple(range(512)),
           'intervention':tuple(range(512))}
    sizes={'image':1,'response':64,'intervention':64}
    seen={k:set() for k in pools}
    for step in range(24):
        kind, batch=module.objective_schedule(pools,step=step,batch_size=sizes)
        assert len(batch)==sizes[kind]
        seen[kind].update(batch)
    assert all(seen[k]==set(pools[k]) for k in pools)
    for invalid in [dict(image=1,response=0,intervention=64),
                    dict(image=1,response=True,intervention=64),
                    dict(image=1,response=64),dict(sizes,unknown=1)]:
        with pytest.raises(ValueError):
            module.objective_schedule(pools,step=0,batch_size=invalid)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_modality_batches_are_bound_copied_and_resume_exactly(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    task,native,request=setup(tmp_path,device)
    pools={'image':(task,), 'response':tuple(response_task() for _ in range(3)),
           'intervention':tuple(response_task('intervention') for _ in range(3))}
    sizes={'image':1,'response':2,'intervention':3}
    model=module.GeometryCooperativeModel(('A','B'),width=8,id_width=16).to(device)
    trainer=module.GeometryCooperativeTrainer(model,pools,image_requests=(request,),
        native=native,batch_size=sizes,prediction_asset='declared.pt')
    sizes['response']=1
    state=trainer.snapshot()
    records=[trainer.step() for _ in range(6)]
    assert [r['examples'] for r in records]==[1,2,3,1,2,3]
    final=trainer.snapshot();trainer.restore(state)
    assert [trainer.step() for _ in range(6)]==records
    equal(final,trainer.snapshot())
    other=module.GeometryCooperativeTrainer(copy.deepcopy(model),pools,
        image_requests=(request,),native=native,batch_size=sizes,prediction_asset='declared.pt')
    with pytest.raises(ValueError): other.restore(state)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('corruption',['model','adam','rng','missing','groups'])
def test_malformed_restore_is_transactional(tmp_path,device,corruption):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(13001)
    task,native,request=setup(tmp_path,device)
    model=module.GeometryCooperativeModel(('A','B'),width=8,id_width=16).to(device)
    trainer=module.GeometryCooperativeTrainer(model,{'image':(task,),
        'response':(response_task(),)},image_requests=(request,),native=native,
        batch_size=1,prediction_asset='declared.pt')
    trainer.step(); saved=trainer.snapshot();trainer.step()
    before=trainer.snapshot();bad=copy.deepcopy(saved)
    if corruption=='model':
        key=next(k for k,v in bad['model'].items() if isinstance(v,torch.Tensor))
        bad['model'][key]=torch.zeros(1,device=device)
    elif corruption=='adam':
        slot=next(iter(bad['optimizer']['state'].values()))
        slot['exp_avg'].fill_(float('nan'))
    elif corruption=='rng':bad['cpu_rng']=torch.zeros(2,dtype=torch.uint8)
    elif corruption=='missing':del bad['cpu_rng']
    else:bad['optimizer']['param_groups'][0]['lr']=float('nan')
    with pytest.raises(ValueError):trainer.restore(bad)
    equal(before,trainer.snapshot())
