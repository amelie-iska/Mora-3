"""Synthetic admission tests; these receipts confer no real-source rights."""
import copy
import multiprocessing
import hashlib
import json
import pytest
import torch
from test_public_image_admission import train_image, receipt
from mora3.data.public_image_admission import PublicImageAdmission
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.public_image_training import PublicImageTrainer
from mora3.model.morphology import foreground_objective
from mora3.model.instance_affinity import affinity_objective


def admitted(tmp_path, **changes):
    source=train_image(); path,digest=receipt(tmp_path,source,**changes)
    return PublicImageAdmission.from_file(source,path,sha256=digest)


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def child_resume(directory,device):
    from pathlib import Path
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a=admitted(Path(directory))
    trainer=PublicImageTrainer(CooperativeModel(('gene',),width=8).to(device),[a])
    trainer.restore(torch.load(Path(directory)/'step1.pt',weights_only=False))
    loss=trainer.step()
    torch.save((loss,trainer.snapshot()),Path(directory)/'step2.pt')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_equation_gradients_and_fresh_process_resume(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)  # Match fresh-process CPU reduction order.
    torch.manual_seed(7201); torch.use_deterministic_algorithms(True)
    a=admitted(tmp_path)
    model=CooperativeModel(('gene',),width=8).to(device)
    trainer=PublicImageTrainer(model,[a])
    loss=model.objective('image',[a])
    z=model.image(a.native_input()); labels=a.instances(device=device)
    reference=foreground_objective(z[0],labels)+affinity_objective(z[1:],labels)
    torch.testing.assert_close(loss,reference,rtol=0,atol=0)
    parameters=tuple(model.image.parameters())
    g=torch.autograd.grad(loss,parameters,allow_unused=True)
    r=torch.autograd.grad(reference,parameters,allow_unused=True)
    for x,y in zip(g,r):
        if x is not None: torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert any(p is q for p in model.image.backbone.parameters() for q in model.response.backbone.parameters())
    trainer.step(); state=trainer.snapshot(); frozen=copy.deepcopy(state)
    torch.save(state,tmp_path/'step1.pt')
    expected_loss=trainer.step(); expected=trainer.snapshot(); equal(state,frozen)
    process=multiprocessing.get_context('spawn').Process(target=child_resume,args=(str(tmp_path),device))
    process.start(); process.join(45)
    assert process.exitcode==0
    observed_loss,observed=torch.load(tmp_path/'step2.pt',weights_only=False)
    assert observed_loss==expected_loss
    equal(expected,observed)


def test_reject_raw_and_changed_admission_config_and_order(tmp_path):
    a=admitted(tmp_path); b=admitted(tmp_path,rights_review='different-fixture-review')
    def trainer(tasks,**kw): return PublicImageTrainer(CooperativeModel(('gene',),width=8),tasks,**kw)
    for tasks in [[],[train_image()]]:
        with pytest.raises(ValueError): trainer(tasks)
    state=trainer([a,b]).snapshot()
    for other in [trainer([b,a]),trainer([a]),trainer([a,b],learning_rate=.002)]:
        before=copy.deepcopy(other.model.state_dict())
        with pytest.raises(ValueError,match='binding'): other.restore(state)
        equal(before,other.model.state_dict())
    for value in [float('nan'),float('inf'),0,-1]:
        with pytest.raises(ValueError): trainer([a],learning_rate=value)
    state['steps']=-1
    with pytest.raises(ValueError): trainer([a,b]).restore(state)
