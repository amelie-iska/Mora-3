"""Catch lost optimizer/RNG/order state and admission of incomplete checkpoints."""
import json
import pytest
import torch
from test_morphology import task
from mora3.model.morphology_training import MorphologyTrainer
from mora3.model import morphology_segments as segments


def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_segmented_maximin_matches_uninterrupted_full_state(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(43)
        trainer=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
        initial=trainer.snapshot()
        losses=[trainer.step() for _ in range(4)]
        expected=trainer.snapshot()
        trainer.restore(initial)
        report=segments.train_segment(trainer,2,tmp_path/'first')
        assert report['start_step']==0 and report['end_step']==2
        resumed=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
        first=segments.restore_segment(resumed,tmp_path/'first')
        segments.train_segment(resumed,2,tmp_path/'second')
        last=segments.restore_segment(resumed,tmp_path/'second')
        assert first['losses']+last['losses']==losses
        assert equal(expected,resumed.snapshot())
        before=resumed.snapshot()
        with pytest.raises(FileExistsError): segments.train_segment(resumed,2,tmp_path/'second')
        assert equal(before,resumed.snapshot())
    finally:
        torch.use_deterministic_algorithms(previous)


def test_incomplete_corrupt_and_mismatched_segments_fail_closed(tmp_path):
    trainer=MorphologyTrainer([task()],width=8)
    incomplete=tmp_path/'partial'; incomplete.mkdir()
    torch.save(trainer.snapshot(),incomplete/'state.pt')
    with pytest.raises(ValueError,match='receipt'): segments.restore_segment(trainer,incomplete)
    segments.train_segment(trainer,1,tmp_path/'ok')
    changed=MorphologyTrainer([task()],width=8,learning_rate=.002)
    with pytest.raises(ValueError,match='config'): segments.restore_segment(changed,tmp_path/'ok')
    p=tmp_path/'ok'/'state.pt'; p.write_bytes(p.read_bytes()+b'corruption')
    before=trainer.snapshot()
    with pytest.raises(ValueError,match='checksum'): segments.restore_segment(trainer,tmp_path/'ok')
    assert equal(before,trainer.snapshot())


@pytest.mark.parametrize('steps',[0,-1,True,1.5])
def test_invalid_segment_horizon_never_updates_or_creates_output(tmp_path,steps):
    trainer=MorphologyTrainer([task()],width=8)
    with pytest.raises(ValueError): segments.train_segment(trainer,steps,tmp_path/'bad')
    assert trainer.steps==0 and not (tmp_path/'bad').exists()


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_restore_never_mutates_caller_checkpoint_optimizer_or_model(device):
    if device=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    # load_state_dict can alias optimizer tensors: continuing must not change a
    # caller-owned snapshot which may later be persisted or reused for comparison.
    import copy
    trainer=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
    trainer.step()
    checkpoint=trainer.snapshot(); frozen=copy.deepcopy(checkpoint)
    resumed=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
    resumed.restore(checkpoint); resumed.step()
    assert equal(frozen,checkpoint)
