"""Counted production launch boundary; test-only metadata, no optimizer updates."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from mora3.research import primary_campaign_entrypoint as entry


def test_legacy_launch_cannot_read_sources_or_reach_unaccounted_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'preflight_primary_worker_launch',lambda *a,**kw:SimpleNamespace(counted_campaign=None))
    seen=[]
    def forbidden(*a,**kw):
        seen.append('source-read');raise AssertionError('uncounted source/allocation path reached')
    monkeypatch.setattr(entry,'_load_repository_source_inputs',forbidden)
    result=entry.main(['--root',str(tmp_path),'execute','--release','outputs/unused.json','--release-sha256','a'*64])
    assert result==2 and not seen


def test_python_execution_entry_cannot_run_legacy_uncounted_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'preflight_primary_worker_launch',lambda *a,**kw:SimpleNamespace(counted_campaign=None))
    with pytest.raises(ValueError,match='[Cc]ounted'):
        entry.execute_primary_campaign(tmp_path,release_path='outputs/unused.json',release_sha256='a'*64,sources=object())


def test_production_entry_calls_counted_worker_not_direct_native_worker():
    tree=ast.parse(Path(entry.__file__).read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute_primary_campaign')
    calls=[n.func.id for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
    assert 'execute_counted_proxy_trial' in calls
    assert 'execute_one_pinned_trial' not in calls


@pytest.fixture
def launch_case(tmp_path,monkeypatch):
    """Synthetic metadata only; no source corpus or real owner qualification."""
    from dataclasses import asdict
    from hashlib import sha256
    from mora3.data.admet import canonical
    from mora3.model.full_owner import PrimaryConfiguration
    from mora3.research.campaign26 import Campaign26,RunBudget26
    from mora3.research.campaign26_transfer import TargetBlueprint
    from mora3.research.primary_campaign_assembly import ArtifactReference
    from mora3.model import primary_source_factory,source_assembly_inputs
    from mora3.research import (campaign26,campaign26_transfer,campaign26_worker,
        primary_campaign_construction62,primary_source_factory_preflight)
    planning=tmp_path/'planning';planning.mkdir()
    policy=dict(schema='Mora3.Campaign26UserDirective/v1',maximum_training_runs=26,proxy_sweep_runs=24,
        full_target_training_runs=2,target_seeds=[2026111001,2026111002],
        standard_parameterization_baseline_runs=0,intermediate_width_training_runs=0,
        naive_standard_parameterization_target_runs=0,
        primary_grid=dict(master_learning_rates=['0.0001','0.0003','0.001'],
            readout_multipliers=['0.5','1'],attention_key_multipliers=['0.5','1'],seeds=[2026110201,2026110202]))
    raw=canonical(policy);(planning/'policy.json').write_bytes(raw)
    pointer=dict(schema='Mora3.CurrentCampaignDirective/v1',path='planning/policy.json',sha256=sha256(raw).hexdigest())
    (planning/'current_mup_campaign.json').write_bytes(canonical(pointer))
    output=tmp_path/'outputs';output.mkdir()
    for name in ('controller','campaign','journal','failures','metadata'):(output/name).mkdir()
    def put(name,value):
        path=output/'metadata'/name;raw=canonical(value);path.write_bytes(raw)
        return dict(path=str(path.relative_to(tmp_path)),sha256=sha256(raw).hexdigest())
    target=PrimaryConfiguration(266,8,'b'*64,readout_mult=.5,attn_mult=.5)
    blueprint=TargetBlueprint(replace(target,readout_mult=1.,attn_mult=1.),80_100_000,
        'd'*64,'e'*64,'f'*64,20_000,86_400,('test-only-native-task',),1.,())
    plan=Campaign26(pointer['sha256'],'b'*64,'c'*64,target_specification_sha256=blueprint.sha256)
    budget=RunBudget26.create(output/'budget',plan)
    events=[]
    campaign=SimpleNamespace(sha256='c'*64,source_snapshot_sha256='b'*64,proxy_width=8,width_family_sha256='e'*64,
        task_cycle=blueprint.task_cycle,gradient_clip_max_norm=1.,decay_aliases=(),train_routes=(),
        selection_contract=lambda:None,require_next_trial=lambda *a:events.append('next-trial'))
    inputs=SimpleNamespace(campaign=campaign,source_snapshot=ArtifactReference('outputs/metadata/source.json','b'*64),
        configurations=(('base',PrimaryConfiguration(8,8,'b'*64,readout_mult=.5,attn_mult=.5)),
                        ('delta',PrimaryConfiguration(12,8,'b'*64,readout_mult=.5,attn_mult=.5)),('target',target)),
        family=SimpleNamespace(target=SimpleNamespace(total_numel=blueprint.parameter_count)),
        request=ArtifactReference(**put('request.json',dict(campaign=dict(path='outputs/metadata/campaign.json',sha256='c'*64)))),
        trial=SimpleNamespace(trial_id=plan.run_ids[0],seed=2026110201,learning_rate='0.0001'),evidence=())
    routes=put('routes.json',dict(schema=primary_campaign_construction62._SCHEMA,primary_request=asdict(inputs.request)))
    modules=dict(entrypoint=entry,route_construction=primary_campaign_construction62,
        source_assembly_inputs=source_assembly_inputs,source_factory=primary_source_factory,
        source_factory_preflight=primary_source_factory_preflight,counted_budget=campaign26,
        counted_worker=campaign26_worker,target_transfer=campaign26_transfer)
    declaration=dict(schema=entry._COUNTED_SCHEMA,route_request=routes,
        capacity=dict(max_identity_metadata_bytes=8192,max_replay_windows=6000),
        controller_directory='outputs/controller',campaign_directory='outputs/campaign',
        journal_path='outputs/journal/once.jsonl',run_id='test-only-run',segment_id='initial',device='cpu',
        geometry_identifier_axis=['test-only-geometry'],color_identifier_axis=['test-only-color'],
        profile_group_manifest=dict(path='outputs/metadata/group.json',sha256='6'*64),profile_private_seed=10,
        execution_root=str(tmp_path),implementation_sha256={k:sha256(Path(v.__file__).read_bytes()).hexdigest()for k,v in modules.items()},
        counted_campaign=dict(budget_directory='outputs/budget',campaign26_sha256=plan.sha256,
            target_blueprint=put('blueprint.json',blueprint.wire()),failure_artifact='outputs/failures/once.json'))
    holder={}
    def publish():
        request=ArtifactReference(**put('launch.json',declaration));holder['request']=request
        return request
    publish()
    def load(*a,**kw):
        events.append('reviewed-release')
        return (ArtifactReference('outputs/metadata/release.json','7'*64),
            dict(schema=entry._LAUNCH_RELEASE_SCHEMA,launch_request=asdict(holder['request'])),inputs,'9'*64)
    monkeypatch.setattr(entry,'load_reviewed_primary_release',load)
    def prefix(b,p,c,cp):
        assert b.plan==plan and p==output/'controller'and c==output/'campaign'and cp is campaign
        events.append('native-prefix')
        return None,SimpleNamespace(outcomes=())
    monkeypatch.setattr(campaign26_worker,'_prefix',prefix)
    return SimpleNamespace(root=tmp_path,put=put,publish=publish,declaration=declaration,events=events,
        inputs=inputs,budget=budget,plan=plan,blueprint=blueprint,holder=holder)


def preflight(case):
    return entry.preflight_primary_worker_launch(case.root,release_path='outputs/metadata/release.json',release_sha256='7'*64)


def test_complete_counted_preflight_is_repeatable_and_reserves_nothing(launch_case):
    c=launch_case
    before={str(p.relative_to(c.root)):p.read_bytes()for p in c.root.rglob('*')if p.is_file()}
    first=preflight(c);second=preflight(c)
    assert first==second and first.counted_campaign.plan==c.plan
    assert first.counted_campaign.target_blueprint==c.blueprint
    assert first.counted_campaign.budget_directory==c.root/'outputs/budget'
    assert c.budget.snapshot()['reserved_runs']==0
    assert before=={str(p.relative_to(c.root)):p.read_bytes()for p in c.root.rglob('*')if p.is_file()}
    assert c.events==['reviewed-release','native-prefix','reviewed-release','native-prefix']


@pytest.mark.parametrize('fault',('unknown-field','missing-accounting','wrong-genesis-pin','wrong-directive',
    'wrong-blueprint-pin','wrong-blueprint-count','wrong-family','wrong-target-config','wrong-source',
    'next-trial','unfinished-slot','failure-exists','failure-pending','failure-is-journal',
    'failure-in-budget','nested-budget','missing-counted-code-pin','changed-counted-code-pin','genesis-extra-field'))
def test_invalid_counted_launch_refuses_without_new_reservations(launch_case,fault):
    import json
    from mora3.data.admet import canonical
    c=launch_case;d=c.declaration;counted=d['counted_campaign']
    if fault=='unknown-field':d['unreviewed']=True
    elif fault=='missing-accounting':del d['counted_campaign']
    elif fault=='wrong-genesis-pin':counted['campaign26_sha256']='0'*64
    elif fault=='wrong-directive':(c.root/'planning/policy.json').write_bytes(b'{}')
    elif fault=='wrong-blueprint-pin':counted['target_blueprint']['sha256']='0'*64
    elif fault=='wrong-blueprint-count':c.inputs.family.target.total_numel+=1
    elif fault=='wrong-family':c.inputs.campaign.width_family_sha256='0'*64
    elif fault=='wrong-target-config':c.inputs.configurations=tuple((k,replace(v,width=264)if k=='target'else v)for k,v in c.inputs.configurations)
    elif fault=='wrong-source':c.inputs.source_snapshot=replace(c.inputs.source_snapshot,sha256='0'*64)
    elif fault=='next-trial':c.inputs.trial.trial_id=c.plan.run_ids[1]
    elif fault=='unfinished-slot':c.budget.reserve(c.plan.run_ids[0])
    elif fault=='failure-exists':(c.root/counted['failure_artifact']).write_bytes(b'preserve')
    elif fault=='failure-pending':(c.root/counted['failure_artifact']).with_suffix('.json.pending').write_bytes(b'preserve')
    elif fault=='failure-is-journal':counted['failure_artifact']=d['journal_path']
    elif fault=='failure-in-budget':counted['failure_artifact']='outputs/budget/failure.json'
    elif fault=='nested-budget':counted['budget_directory']='outputs/controller'
    elif fault=='missing-counted-code-pin':del d['implementation_sha256']['counted_worker']
    elif fault=='changed-counted-code-pin':d['implementation_sha256']['counted_worker']='0'*64
    elif fault=='genesis-extra-field':
        p=c.budget.directory/'genesis.json';value=json.loads(p.read_bytes());value['extra']=1;p.write_bytes(canonical(value))
    c.publish();before={p.name for p in(c.budget.directory/'runs').iterdir()}
    expected={
        'unknown-field':'versioned worker launch','missing-accounting':'versioned worker launch',
        'wrong-genesis-pin':'genesis','wrong-directive':'directive changed',
        'wrong-blueprint-pin':'digest differs','wrong-blueprint-count':'Preregistered target',
        'wrong-family':'Preregistered target','wrong-target-config':'Preregistered target',
        'wrong-source':'genesis','next-trial':'next unreserved','unfinished-slot':'next unreserved',
        'failure-exists':'failure artifact','failure-pending':'failure artifact',
        'failure-is-journal':'failure artifact','failure-in-budget':'Failure artifact',
        'nested-budget':'disjoint','missing-counted-code-pin':'implementation differs',
        'changed-counted-code-pin':'implementation differs','genesis-extra-field':'genesis'}
    with pytest.raises(ValueError,match=expected[fault]):preflight(c)
    assert before=={p.name for p in(c.budget.directory/'runs').iterdir()}
    assert 'native-prefix'not in c.events


def test_native_terminal_divergence_is_checked_before_any_source_read(launch_case,monkeypatch):
    from mora3.research import campaign26_worker
    def mismatch(*a,**kw):raise ValueError('native and budget terminal prefixes differ')
    monkeypatch.setattr(campaign26_worker,'_prefix',mismatch)
    with pytest.raises(ValueError,match='terminal prefixes'):preflight(launch_case)
    assert launch_case.budget.snapshot()['reserved_runs']==0


def test_actual_registry_refuses_before_metadata_or_source_access(tmp_path,monkeypatch):
    from mora3.model import full_owner_training_admission as admission
    assert not admission._REVIEWED_RELEASES
    def forbidden(*a,**kw):raise AssertionError('Unreviewed metadata/source access')
    monkeypatch.setattr(admission,'_metadata',forbidden)
    monkeypatch.setattr(entry,'_load_repository_source_inputs',forbidden)
    assert entry.main(['--root',str(tmp_path),'execute','--release','outputs/not-approved.json',
                       '--release-sha256','0'*64])==2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('valid_capacity',(False,True))
def test_native_execution_hands_off_to_counted_worker_only(launch_case,monkeypatch,valid_capacity):
    from unittest.mock import Mock
    from mora3.model import full_owner,full_owner_mup,mu_optimizer,profile_reward_stage
    from mora3.research import (mup_campaign_pin_store,mup_campaign_schedule,primary_campaign_construction62,
        primary_training,runtime_capacity_gate,campaign26_worker)
    c=launch_case;launch=preflight(c);seen=[]
    sources=full_owner.SourceAssemblyInputs.__new__(full_owner.SourceAssemblyInputs)
    object.__setattr__(sources,'reconstruction_source',dict(observed_group_manifest=dict(
        path=launch.profile_group_manifest.path,sha256=launch.profile_group_manifest.sha256)))
    monkeypatch.setattr(entry,'preflight_primary_worker_launch',lambda *a,**kw:launch)
    binding=SimpleNamespace(route=SimpleNamespace(initial_cursor=b'test-only-source-cursor'))
    routes=SimpleNamespace(inputs=launch.inputs,route_request=launch.route_request,
        routes=SimpleNamespace(plans=(),bindings=(('native',binding),),evaluators=(),validation=object()))
    monkeypatch.setattr(primary_campaign_construction62,'construct_scoped_primary_campaign62',lambda *a:routes)
    monkeypatch.setattr(mup_campaign_pin_store,'load_pinned_campaign',lambda *a,**kw:SimpleNamespace(pins=()))
    monkeypatch.setattr(runtime_capacity_gate,'require_runtime_window_capacity',
        lambda *a,**kw:SimpleNamespace(sha256='9'*64 if valid_capacity else'0'*64))
    owner=Mock();constructor=Mock(return_value=owner)
    monkeypatch.setattr(full_owner,'construct_primary_owner',constructor)
    monkeypatch.setattr(full_owner_mup,'full_owner_mup_parameter_roles',lambda *a,**kw:{})
    monkeypatch.setattr(mu_optimizer,'MuAdamW',lambda *a,**kw:object())
    monkeypatch.setattr(mup_campaign_schedule,'NativeMuWarmupCosine',lambda *a,**kw:object())
    monkeypatch.setattr(profile_reward_stage,'ProfileRewardStageController',lambda *a,**kw:object())
    engine=object();monkeypatch.setattr(primary_training,'NativeMuUpdateEngine',lambda *a,**kw:engine)
    def uncounted(*a,**kw):raise AssertionError('Direct uncounted worker bypass')
    monkeypatch.setattr(mup_campaign_pin_store,'execute_one_pinned_trial',uncounted)
    def counted(**kw):seen.append(kw);return 'test-only-handoff'
    monkeypatch.setattr(campaign26_worker,'execute_counted_proxy_trial',counted)
    kwargs=dict(release_path='outputs/metadata/release.json',release_sha256='7'*64,sources=sources)
    if not valid_capacity:
        with pytest.raises(ValueError,match='capacity'):entry.execute_primary_campaign(c.root,**kwargs)
        assert not seen;constructor.assert_not_called()
    else:
        assert entry.execute_primary_campaign(c.root,**kwargs)=='test-only-handoff'
        assert len(seen)==1
        kw=seen[0];assert kw['engine']is engine and kw['budget'].plan==c.plan
        assert kw['budget'].directory==c.budget.directory and kw['target_blueprint']==c.blueprint
        assert kw['failure_artifact']==c.root/'outputs/failures/once.json'
        assert kw['pin_directory']==launch.controller_directory and kw['campaign_directory']==launch.campaign_directory
        owner.authorize_primary_campaign_training.assert_called_once()
        owner.require_training_admission.assert_called_once()
    assert c.budget.snapshot()['reserved_runs']==0
