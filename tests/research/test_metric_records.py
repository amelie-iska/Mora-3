import dataclasses
import json
from types import SimpleNamespace

import pytest

from mora3.research.metric_records import MetricRecord, write_history, read_history, emit_wandb_offline


def record(**kw):
    values = dict(run_id='run1', trial_id='trial1', segment_id='fresh', seed=71,
                  step=0, x_axis='optimizer_updates', name='validation/loss', value=1.25,
                  unit='nats/example', direction='minimize', task='rna', split='validation',
                  source_sha256='a'*64, evidence_sha256='b'*64, scope='implementation_diagnostic')
    return MetricRecord(**(values | kw))


def test_roundtrip_retains_resumed_step_and_missing_metric(tmp_path):
    rows = [record(), record(segment_id='resume1', value=.75),
            record(step=1, value=None, missing_reason='wall_time_limit')]
    path=tmp_path/'metrics.jsonl'
    assert write_history(path, rows)==3
    assert read_history(path)==tuple(rows)
    assert [json.loads(s)['event_index'] for s in path.read_text().splitlines()]==[0,1,2]
    with pytest.raises(FileExistsError):write_history(path,rows)


@pytest.mark.parametrize('kw', [dict(value=float('nan')), dict(value=True), dict(step=True),
    dict(seed=-1), dict(value=None), dict(value=1,missing_reason='missing'),
    dict(source_sha256='unknown'), dict(direction='best'), dict(scope='clinical_quality'),
    dict(numerator=1,denominator=0), dict(numerator=1),dict(denominator=-1),
    dict(segment_id=''),dict(unit='')])
def test_invalid_evidence_or_values_refuse(kw):
    with pytest.raises(ValueError):record(**kw)


@pytest.mark.parametrize('rows', [lambda:[record(),record(value=2)],
    lambda:[record(),record(step=1,unit='mse')],
    lambda:[record(step=2),record(step=1)],
    lambda:[record(),record(step=1,seed=72)],
    lambda:[record(),record(step=1,trial_id='trial2')]])
def test_conflicting_duplicate_unit_order_or_run_binding_refuses(tmp_path,rows):
    with pytest.raises(ValueError):write_history(tmp_path/'history.jsonl',rows())
    # Validate first, then create: a refused batch leaves no misleading partial history.
    assert not (tmp_path/'history.jsonl').exists()


def test_reader_refuses_tampered_order_extra_fields_or_duplicate_json_key(tmp_path):
    p=tmp_path/'history';write_history(p,[record()]);original=p.read_text()
    # New fixture files preserve the real history.
    for i,line in enumerate([original.replace('"event_index":0','"event_index":3'),
                              original.replace('"event_index":0','"extra":1,"event_index":0'),
                              original.replace('"event_index":0','"event_index":0,"event_index":0')]):
        q=tmp_path/str(i);q.write_text(line)
        with pytest.raises(ValueError):read_history(q)


class OfflineRun:
    def __init__(self,mode='offline'):
        self.settings=SimpleNamespace(mode=mode);self.logs=[];self.step=0
    def log(self,payload,step):self.logs.append((payload,step))


def test_offline_adapter_preserves_every_record_without_logging_into_service():
    rows=[record(),record(segment_id='resume1'),record(step=1,value=None,missing_reason='not_measured')]
    run=OfflineRun();assert emit_wandb_offline(rows,run)==3
    assert [step for _,step in run.logs]==[0,1,2]
    assert [p['mora3/record']['segment_id'] for p,_ in run.logs]==['fresh','resume1','fresh']
    assert run.logs[-1][0]['mora3/record']['missing_reason']=='not_measured'
    online=OfflineRun('online')
    with pytest.raises(ValueError):emit_wandb_offline(rows,online)
    assert not online.logs


def test_offline_adapter_refuses_preexisting_step_to_prevent_silent_wandb_drops():
    run=OfflineRun();run.step=4
    with pytest.raises(ValueError):emit_wandb_offline([record()],run)
    assert not run.logs


def test_ratio_keeps_native_counts_and_explicit_zero_denominator_missingness():
    assert record(value=.5,numerator=3,denominator=6).denominator==6
    assert record(value=None,numerator=0,denominator=0,missing_reason='no_eligible_records').value is None

