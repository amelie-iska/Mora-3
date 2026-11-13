from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from test_mup_trial_artifacts import execution
from mora3.research.mup_sweep_selection import planned_proxy_trials
from mora3.research.mup_campaign_recovery import recover_terminal_prefix
from mora3.research import mup_campaign_worker as worker


@pytest.fixture
def harness(monkeypatch,tmp_path):
    selection,result=execution()
    events=[]
    class Model:
        def require_training_admission(self,**scope):
            if not scope: events.append('admission')
    class Engine:
        model=Model()
    class Campaign:
        sha256=selection.campaign_manifest_sha256
        def selection_contract(self): return selection
        def require_next_trial(self,trial_id,terminal_trial_ids):
            events.append('next')
            grid=planned_proxy_trials()
            if terminal_trial_ids!=tuple(t.trial_id for t in grid[:len(terminal_trial_ids)]) or trial_id!=grid[len(terminal_trial_ids)].trial_id:
                raise ValueError('wrong next trial')
            return grid[len(terminal_trial_ids)]
    monkeypatch.setattr(worker,'NativeMuUpdateEngine',Engine)
    monkeypatch.setattr(worker,'SourceBoundOwner',Model)
    monkeypatch.setattr(worker,'FrozenPrimaryProxyCampaign',Campaign)
    monkeypatch.setattr(worker,'require_runtime_window_capacity',lambda *args,**kwargs: SimpleNamespace(sha256='a'*64))
    monkeypatch.setattr(worker,'_require_runtime',lambda *args: events.append('runtime'))
    def run(*args,**kwargs):
        events.append('run')
        kwargs['metric_capture'](result.metrics)
        return result
    monkeypatch.setattr(worker,'run_primary_proxy_trial',run)
    for name,label in [('reserve_next_attempt','reserve'),('seal_proxy_trial_execution','seal'),('publish_terminal_receipt','publish')]:
        original=getattr(worker,name)
        def wrapped(*args,_original=original,_label=label,**kwargs):
            events.append(_label)
            return _original(*args,**kwargs)
        monkeypatch.setattr(worker,name,wrapped)
    kwargs=dict(engine=Engine(),campaign=Campaign(),trial_id=result.trial_id,pins=(),bindings={},
        validation_view=None,validation_evaluators={},metric_identity=None,root='.',evidence_paths={},
        journal_path=tmp_path.parent/(tmp_path.name+'-journal.jsonl'))
    return selection,result,events,kwargs


def test_admission_refusal_precedes_reservation(tmp_path,harness,monkeypatch):
    selection,result,events,kwargs=harness
    def refuse(): raise ValueError('not admitted')
    monkeypatch.setattr(kwargs['engine'].model,'require_training_admission',refuse)
    with pytest.raises(ValueError,match='not admitted'): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert list(tmp_path.iterdir())==[] and 'reserve' not in events and 'run' not in events


def test_success_sequence_and_readback(tmp_path,harness):
    selection,result,events,kwargs=harness
    pin=worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert events==['admission','next','runtime','reserve','run','seal','publish']
    assert recover_terminal_prefix(tmp_path,selection=selection,pins=(pin,)).terminal_trial_ids==(result.trial_id,)
    with pytest.raises(ValueError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert events.count('run')==1


def test_wrong_requested_trial_never_reserves(tmp_path,harness):
    _,_,events,kwargs=harness
    kwargs['trial_id']=planned_proxy_trials()[1].trial_id
    with pytest.raises(ValueError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert list(tmp_path.iterdir())==[] and 'reserve' not in events


@pytest.mark.parametrize('failure',['overbudget','interrupt','wrong-result'])
def test_failed_attempt_preserved_and_never_reused(tmp_path,harness,monkeypatch,failure):
    selection,result,events,kwargs=harness
    def fail(*args,**kw):
        events.append('run')
        if failure=='interrupt': raise KeyboardInterrupt()
        if failure=='overbudget': return replace(result,measured_wall_seconds=Decimal('1801'))
        return replace(result,trial_id=planned_proxy_trials()[1].trial_id)
    monkeypatch.setattr(worker,'run_primary_proxy_trial',fail)
    with pytest.raises((ValueError,KeyboardInterrupt)): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    trial=tmp_path/result.trial_id
    assert (trial/'blocked.json').is_file() and not (trial/'receipt.json').exists()
    with pytest.raises(ValueError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert events.count('run')==1


def test_runtime_refusal_precedes_reservation(tmp_path,harness,monkeypatch):
    _,_,events,kwargs=harness
    def refuse(*args): raise ValueError('runtime refused')
    monkeypatch.setattr(worker,'_require_runtime',refuse)
    with pytest.raises(ValueError,match='runtime refused'): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert list(tmp_path.iterdir())==[] and 'run' not in events


def test_missing_capture_blocks_terminal_publication(tmp_path,harness,monkeypatch):
    _,result,events,kwargs=harness
    monkeypatch.setattr(worker,'run_primary_proxy_trial',lambda *a,**kw: result)
    with pytest.raises(ValueError,match='journal'): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert not (tmp_path/result.trial_id/'receipt.json').exists()


def test_journal_collision_preserves_file_and_never_runs(tmp_path,harness):
    _,result,events,kwargs=harness
    kwargs['journal_path'].write_bytes(b'protected prior output')
    with pytest.raises(FileExistsError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert kwargs['journal_path'].read_bytes()==b'protected prior output'
    assert 'run' not in events and (tmp_path/result.trial_id/'blocked.json').is_file()


def test_journal_inside_campaign_refused_before_reservation(tmp_path,harness):
    _,result,events,kwargs=harness
    kwargs['journal_path']=tmp_path/'metrics.jsonl'
    with pytest.raises(ValueError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert list(tmp_path.iterdir())==[] and 'reserve' not in events


def test_capture_failure_cannot_be_a_data_refusal(tmp_path,harness,monkeypatch):
    _,result,events,kwargs=harness
    def fail(self,records): raise ValueError('invalid captured metric')
    monkeypatch.setattr(worker.DurableMetricJournal,'append_batch',fail)
    with pytest.raises(worker.MetricCaptureError): worker.execute_one_primary_proxy_trial(tmp_path,**kwargs)
    assert not (tmp_path/result.trial_id/'receipt.json').exists()


def test_runner_capture_hook_is_optional_and_wraps_refusals():
    from mora3.research.mup_campaign import _capture_metrics, MetricCaptureError
    _capture_metrics(None,())
    def fail(rows): raise ValueError('journal refuses')
    with pytest.raises(MetricCaptureError): _capture_metrics(fail,())
