from pathlib import Path
from dataclasses import replace
from decimal import Decimal
import tempfile
import pytest
from mora3.research import mcsa_campaign_bridge as m
from mora3.data.mcsa_role_list_admission import FAMILY,TASK,NATIVE
from mora3.research.matched_validation import MatchedValidationTask

@pytest.fixture(scope='module')
def bridge():
    root=Path.cwd();out=Path(tempfile.mkdtemp(prefix='TEST_ONLY_',dir=root/'outputs/implementation/readiness-20261108-combined60-campaign-integration-v1'));p=out/'TEST_ONLY_snapshot.json';p.write_bytes(b'{"TEST_ONLY_NOT_AUTHORITY":true}')
    args=dict(source_snapshot=dict(path=str(p.relative_to(root)),sha256=m.digest(p.read_bytes())),pass_policy=m.MCSAFinitePassPolicy(2,16,2026110201),max_windows=76,task_cycle=(TASK,),total_updates=76,max_validation_bytes=8*1024*1024)
    return root,args,m.build_mcsa_bridge(root,**args)

def test_real_public_bridge(bridge):
    root,args,b=bridge
    assert b.binding.route.records_per_pass==602 and b.binding.route.maximum_updates==76
    assert len(b.coordinates)==68 and b.observed_targets==2043
    assert b.evaluator.__self__ is b.binding.route and b.validation_sources.mcsa is b.binding.route
    assert b.plan.adapter_sha256!=b.evaluator_sha256
    assert not b.training_authorized and not b.allocation_authorized
    task=MatchedValidationTask(TASK,FAMILY,TASK,NATIVE,b.coordinates,2043,Decimal('1'),Decimal('1'),b.binding.route.loss_unit,b.binding.route.target_unit)
    batches=b.require_matched_task(task,read_batch_size=16,max_selected_bytes=8*1024*1024)
    assert sum(x.observed_targets for x in batches)==2043
    with pytest.raises(ValueError):b.require_matched_task(replace(task,observed_targets=2044),read_batch_size=16,max_selected_bytes=8*1024*1024)

def test_insufficient_pass_capacity(bridge):
    root,args,b=bridge
    with pytest.raises(ValueError):m.build_mcsa_bridge(root,**(args|dict(total_updates=77)))

def test_missing_pinned_snapshot_before_source(bridge,monkeypatch):
    root,args,b=bridge
    def forbidden(*a,**kw):raise AssertionError('Source construction reached')
    monkeypatch.setattr(m,'MCSACampaignTrainRoute',forbidden)
    with pytest.raises(ValueError):m.build_mcsa_bridge(root,**(args|dict(source_snapshot=dict(path=args['source_snapshot']['path'],sha256='a'*64))))
