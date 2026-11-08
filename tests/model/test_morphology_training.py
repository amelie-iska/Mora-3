"""Continuation must fail if data/order/config changes, including baseline config."""
import copy
from dataclasses import replace
import pytest
import torch
from test_morphology import task
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.model import morphology_training as training


@pytest.fixture(autouse=True)
def deterministic_runtime():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('variant',['foreground','instance'])
def test_exact_resume_and_fail_closed_task_configuration(variant,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(47)
    tasks=[task()]
    trainer=training.MorphologyTrainer(tasks,variant=variant,width=8,device=device)
    trainer.step()
    state=copy.deepcopy(trainer.snapshot())
    expected=trainer.step()
    resumed=training.MorphologyTrainer(tasks,variant=variant,width=8,device=device)
    resumed.restore(state)
    assert resumed.step()==expected
    for a,b in zip(trainer.model.parameters(),resumed.model.parameters()):
        assert torch.equal(a,b)
    changed=LiveCellGraphTask(replace(tasks[0].example,pixels_hex='ff'*30),'image_to_mask')
    with pytest.raises(ValueError,match='data|config'):
        training.MorphologyTrainer([changed],variant=variant,width=8,device=device).restore(state)
    with pytest.raises(ValueError,match='config'):
        training.MorphologyTrainer(tasks,variant=variant,width=8,device=device,learning_rate=.002).restore(state)
    with pytest.raises(ValueError,match='split'):
        training.MorphologyTrainer([task('test')],variant=variant,width=8,device=device)


def test_evaluation_is_read_only_and_reports_fixed_threshold_metrics():
    trainer=training.MorphologyTrainer([task()],variant='instance',width=8)
    before=copy.deepcopy(trainer.snapshot())
    metrics=trainer.evaluate(task('test'))
    assert 0<=metrics['foreground_iou']<=1
    assert metrics['objects']['target_objects']==1
    assert metrics['thresholds']==dict(foreground=.5,affinity=.5,iou=.5)
    assert trainer.steps==0
    for k,v in trainer.model.state_dict().items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,before['model'][k])
