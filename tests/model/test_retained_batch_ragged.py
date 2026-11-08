"""Explicit per-call batch preparation preserves the single-reader equations."""
from functools import partial
from pathlib import Path
import pytest
import torch
import mora3.model.identity_noise_vector as mod
import test_retained_identity_noise_ragged as original


@pytest.fixture
def batch(monkeypatch):
    fn=partial(mod.retained_identity_noise_vector_loss,batch_identity=True)
    monkeypatch.setattr(mod,'retained_identity_noise_vector_loss',fn)
    return fn


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_batch_independent_equations_all_gradients(batch,device,dtype):
    original.test_equal_case_independent_equations_and_gradients(device,dtype)


@pytest.mark.parametrize('bad',['record','split','axis','context','payload','target','xyz','extra','selection_extra','owner','ready','length','container'])
def test_batch_late_refusal_before_learned_calls(batch,bad,tmp_path):
    original.test_late_refusal_precedes_all_learned_calls(bad,tmp_path)


def test_batch_mutation_between_calls_and_opaque_inactive(batch,tmp_path):
    original.test_each_call_reauthenticates_and_inactive_values_opaque(tmp_path)


def test_order_repetition_hash_once_per_call_and_default_preserved(monkeypatch):
    s,c,t=original.inputs();s=[s[1],s[0],s[1]];c=[c[1],c[0],c[1]];t=[t[1],t[0],t[1]]
    baseline=mod.retained_identity_noise_vector_loss(s,c,t,[True]*3)
    reads=[];read=Path.read_bytes
    def counted(path):
        reads.append(path.resolve());return read(path)
    monkeypatch.setattr(Path,'read_bytes',counted)
    result=mod.retained_identity_noise_vector_loss(s,c,t,[True]*3,batch_identity=True)
    torch.testing.assert_close(result.loss,baseline.loss,rtol=0,atol=0)
    assert result.active_cases==(0,1,2)
    assert len(reads)==2 and len(set(reads))==2
    mod.retained_identity_noise_vector_loss(s,c,t,[True]*3,batch_identity=True)
    assert len(reads)==4


@pytest.mark.parametrize('mode',[1,None,'batch'])
def test_exact_opt_in(mode):
    s,c,t=original.inputs()
    with pytest.raises(ValueError,match='batch_identity'):
        mod.retained_identity_noise_vector_loss(s,c,t,[True,True],batch_identity=mode)


def test_batch_active_limit_before_work():
    s,c,t=original.inputs()
    with pytest.raises(ValueError,match='64'):
        mod.retained_identity_noise_vector_loss([s[0]]*65,[c[0]]*65,[t[0]]*65,[True]*65,batch_identity=True)
