from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from test_mup_trial_artifacts import execution
from mora3.research import mup_campaign as campaign_module


@pytest.mark.parametrize('capture_fails',[False,True])
def test_actual_runner_dispatches_update_and_validation_capture(monkeypatch,capture_fails):
    _,fixture=execution()
    training=(replace(fixture.metrics[0],name='train-loss',scope='train',split='train',step=1),)
    validation=tuple(replace(row,step=1) for row in fixture.metrics)
    class Model:
        def require_training_admission(self): pass
    class Engine:
        model=Model()
        steps=0
        next_task='task'
    class Campaign:
        sha256=fixture.campaign_manifest_sha256
        train_routes=(SimpleNamespace(task_key='task',max_windows=1),)
        validation=SimpleNamespace(comparison_update=1,sha256=fixture.validation_exposure_sha256)
        max_one_update_seconds=Decimal('10')
        max_matched_validation_seconds=Decimal('10')
        def require_next_trial(self,*args): return SimpleNamespace(trial_id=fixture.trial_id)
    monkeypatch.setattr(campaign_module,'NativeMuUpdateEngine',Engine)
    monkeypatch.setattr(campaign_module,'SourceBoundOwner',Model)
    monkeypatch.setattr(campaign_module,'FrozenPrimaryProxyCampaign',Campaign)
    monkeypatch.setattr(campaign_module,'MAX_UPDATES',1)
    monkeypatch.setattr(campaign_module,'_require_runtime',lambda *args: None)
    def update(engine,*args,**kwargs):
        engine.steps+=1
        return SimpleNamespace(metrics=training,status='updated')
    monkeypatch.setattr(campaign_module,'run_native_primary_update',update)
    monkeypatch.setattr(campaign_module,'_evaluate',lambda *args:tuple(SimpleNamespace(coordinate=c) for c in fixture.validation))
    monkeypatch.setattr(campaign_module,'_validation_metrics',lambda *args:validation)
    seen=[]
    def capture(rows):
        seen.append(rows)
        if capture_fails: raise ValueError('simulated capture failure')
    kwargs=dict(trial_id=fixture.trial_id,terminal_trial_ids=(),bindings={},validation_view=None,
        validation_evaluators={},metric_identity=None,root='.',metric_capture=capture)
    if capture_fails:
        with pytest.raises(campaign_module.MetricCaptureError):
            campaign_module.run_primary_proxy_trial(Engine(),Campaign(),**kwargs)
        assert seen==[training]
    else:
        result=campaign_module.run_primary_proxy_trial(Engine(),Campaign(),**kwargs)
        assert seen==[training,validation] and result.metrics==(*training,*validation)
