from dataclasses import replace

import pytest

from test_mup_trial_artifacts import execution
from mora3.research.metric_records import read_history
from mora3.research import mup_metric_journal as journal


def test_batches_are_canonical_and_prefix_is_forensic(tmp_path):
    _,run=execution()
    path=tmp_path/'metrics.jsonl'
    later=tuple(replace(row,step=row.step+1) for row in run.metrics)
    with journal.DurableMetricJournal(path) as writer:
        assert writer.append_batch(run.metrics)==2
        assert writer.append_batch(later)==4
    assert read_history(path)==(*run.metrics,*later)
    prefix=journal.read_forensic_prefix(path)
    assert prefix.metrics==(*run.metrics,*later) and prefix.torn_tail_bytes==0
    assert prefix.resume_authorized is False
    with pytest.raises(FileExistsError): journal.DurableMetricJournal(path)


@pytest.mark.parametrize('case',['duplicate','regression','meaning','binding'])
def test_cross_batch_checks_and_poisoned_writer(tmp_path,case):
    _,run=execution()
    writer=journal.DurableMetricJournal(tmp_path/'metrics.jsonl')
    writer.append_batch(run.metrics)
    row=run.metrics[0]
    if case=='regression': row=replace(row,step=row.step-1)
    elif case=='meaning': row=replace(row,step=row.step+1,unit='different')
    elif case=='binding': row=replace(row,step=row.step+1,seed=row.seed+1)
    with pytest.raises(ValueError): writer.append_batch((row,))
    with pytest.raises(ValueError): writer.append_batch((replace(row,step=row.step+2),))
    writer.close()
    assert read_history(tmp_path/'metrics.jsonl')==run.metrics


def test_torn_tail_is_reported_without_modification(tmp_path):
    _,run=execution()
    path=tmp_path/'metrics.jsonl'
    with journal.DurableMetricJournal(path) as writer: writer.append_batch(run.metrics)
    with path.open('ab') as stream: stream.write(b'{"schema":')
    before=path.read_bytes()
    result=journal.read_forensic_prefix(path)
    assert result.metrics==run.metrics and result.torn_tail_bytes==10
    assert path.read_bytes()==before


def test_malformed_complete_line_is_not_silently_a_tail(tmp_path):
    path=tmp_path/'metrics.jsonl'
    path.write_bytes(b'{broken}\n')
    with pytest.raises(ValueError): journal.read_forensic_prefix(path)


def test_one_fsync_per_batch_and_failure_poisoning(tmp_path,monkeypatch):
    _,run=execution()
    writer=journal.DurableMetricJournal(tmp_path/'metrics.jsonl')
    calls=[]
    original=journal.os.fsync
    def sync(fd): calls.append(fd); original(fd)
    monkeypatch.setattr(journal.os,'fsync',sync)
    writer.append_batch(run.metrics)
    assert len(calls)==1
    def fail(fd): raise OSError('simulated fsync failure')
    monkeypatch.setattr(journal.os,'fsync',fail)
    with pytest.raises(OSError): writer.append_batch(tuple(replace(r,step=r.step+1) for r in run.metrics))
    with pytest.raises(ValueError): writer.append_batch(())
    writer.close()
