"""Replay native24-trial evidence and transfer its winner to two frozen targets.

This is the post-sweep handoff, not a new trainer or a source-admission authority.
The complete target blueprint is bound into the v2 run-budget genesis BEFORE any
proxy reservation. The selected master LR is retained; the existing MuAdamW
applies parameter-wise width multipliers exactly once. No checkpoint weights,
optimizer moments or target validation results are transferred as hyperparameters.

Artifact SHA256 (the published wrapper), result SHA256 (the inner selector), and
terminal-prefix SHA256 (the receipt chain) are deliberately separate identities.
Historical v1 budgets remain readable but cannot imply target preregistration.
"""
from dataclasses import asdict,dataclass,replace
from hashlib import sha256
import json
import math
from pathlib import Path

from mora3.data.admet import canonical
from mora3.model.full_owner import PrimaryConfiguration,SourceBoundOwner,unique_parameter_inventory
from .campaign26 import Campaign26,RunBudget26,TARGET_PARAMETER_BOUNDS,_digest,_positive,_safe,_read,_publish,target_run_specs
from .campaign26_worker import require_current_directive
from .mup_campaign_pin_store import load_pinned_campaign
from .mup_campaign_recovery import recover_terminal_prefix
from .mup_sweep_selection import FrozenProxySelection,select_proxy_transfer


@dataclass(frozen=True)
class TargetBlueprint:
    """Predeclared non-tuned target settings, not measured execution approval.

    readout/attention values in the template must be1: only the native selected
    winner fills those slots. Width/base, all-head count, source, operator family,
    schedule horizon, task cycle, clipping and decay ownership are fixed.
    """
    configuration: PrimaryConfiguration
    parameter_count: int
    complete_owner_sha256: str
    width_family_sha256: str
    runtime_profile_sha256: str
    maximum_updates: int
    maximum_seconds: int
    task_cycle: tuple[str,...]
    gradient_clip_max_norm: float
    decay_aliases: tuple[str,...]

    def __post_init__(self):
        if type(self.configuration)is not PrimaryConfiguration:
            raise ValueError('Exact native target configuration template required')
        self.configuration.__post_init__()
        if self.configuration.readout_mult!=1. or self.configuration.attn_mult!=1.:
            raise ValueError('Target multiplier slots must remain neutral until native validation selection')
        for value in(self.complete_owner_sha256,self.width_family_sha256,self.runtime_profile_sha256):_digest(value)
        for value in(self.parameter_count,self.maximum_updates,self.maximum_seconds):_positive(value)
        from .mup_campaign_schedule import _warmup_updates
        _warmup_updates(self.maximum_updates)
        if not TARGET_PARAMETER_BOUNDS[0]<=self.parameter_count<=TARGET_PARAMETER_BOUNDS[1]:
            raise ValueError('Complete approximately80M all-head target count required')
        if (type(self.task_cycle)is not tuple or not self.task_cycle
                or any(type(x)is not str or not x.strip()for x in self.task_cycle)
                or type(self.decay_aliases)is not tuple
                or any(type(x)is not str or not x.strip()for x in self.decay_aliases)
                or self.decay_aliases!=tuple(sorted(set(self.decay_aliases)))
                or type(self.gradient_clip_max_norm)is not float
                or not math.isfinite(self.gradient_clip_max_norm)or self.gradient_clip_max_norm<=0):
            raise ValueError('Exact frozen task cycle, complete decay alias policy and positive gradient clipping required')

    def wire(self):
        self.__post_init__()
        return dict(schema='Mora3.TargetTransferBlueprint/v1',**asdict(self),
            selection_fields=('master_learning_rate','readout_multiplier','attention_key_multiplier'),
            parameterization='muP',target_validation_or_test_selection=False,
            source_and_operator_qualification_issued=False)

    @property
    def sha256(self):return sha256(canonical(self.wire())).hexdigest()

    @classmethod
    def from_wire(cls,document):
        if type(document)is not dict:raise ValueError('Complete target blueprint required')
        try:
            result=cls(PrimaryConfiguration(**document['configuration']),document['parameter_count'],
                document['complete_owner_sha256'],document['width_family_sha256'],document['runtime_profile_sha256'],
                document['maximum_updates'],document['maximum_seconds'],tuple(document['task_cycle']),
                document['gradient_clip_max_norm'],tuple(document['decay_aliases']))
        except (KeyError,TypeError)as exc:raise ValueError('Malformed complete target blueprint')from exc
        if canonical(result.wire())!=canonical(document):raise ValueError('Unknown or changed target blueprint fields')
        return result


def require_target_blueprint(plan,blueprint):
    if type(plan)is not Campaign26 or type(blueprint)is not TargetBlueprint:
        raise ValueError('Exact campaign and preregistered target blueprint required')
    plan.__post_init__();blueprint.__post_init__()
    if (plan.target_specification_sha256 is None or plan.target_specification_sha256!=blueprint.sha256
            or plan.source_snapshot_sha256!=blueprint.configuration.source_snapshot_sha256):
        raise ValueError('Target specification must match the immutable pre-sweep run-budget commitment')


def verify_counted_selection(*,budget,root,pin_directory,campaign_directory,selection,selection_path):
    """Re-read ALL24 native histories, chained pins and counted terminal results.

    Also works after a target has started: only the unchanged first24 slots are
    selection evidence. Actual target outcomes cannot influence the winner.
    No artifacts are adopted, rewritten, reserved, completed or retried here.
    """
    if type(budget)is not RunBudget26 or type(selection)is not FrozenProxySelection:
        raise ValueError('Exact persistent run budget and frozen native selector required')
    require_current_directive(root,budget.plan);selection.__post_init__()
    if selection.campaign_manifest_sha256!=budget.plan.proxy_campaign_sha256:
        raise ValueError('Selector and budget campaign commitments differ')
    with budget._locked():
        states,bound=budget._load()
        if len(states)<24 or any(not row[1]or row[2]is None for row in states[:24])or bound is None:
            raise ValueError('All24 entered terminal proxies and one bound native selection required')
        terminal_binding=canonical(states[:24]);selection_digest=bound['selection_sha256']
    pins=load_pinned_campaign(pin_directory,campaign_directory,selection=selection)
    prefix=recover_terminal_prefix(campaign_directory,selection=selection,pins=pins.pins)
    if len(prefix.outcomes)!=24:raise ValueError('Complete24-result native terminal prefix required')
    for (claim,entered,terminal),outcome in zip(states[:24],prefix.outcomes,strict=True):
        if (claim['run_id']!=outcome.trial_id or terminal['status']!=outcome.status
                or terminal['artifact_sha256']!=outcome.run_artifact_sha256):
            raise ValueError('Counted run and native terminal artifact disagree')
    result=select_proxy_transfer(selection,prefix.outcomes)
    if result['frozen_transfer_hyperparameters']is None:
        raise ValueError('No complete eligible paired-seed hyperparameter point; no target transfer')
    publication=_read(_safe(selection_path))
    expected=dict(schema='Mora3.PinnedProxyTransferSelection/v1',
        terminal_pin_prefix_sha256=pins.tip_sha256,selection=result)
    if canonical(publication)!=canonical(expected)or sha256(canonical(publication)).hexdigest()!=selection_digest:
        raise ValueError('Published winner is not the exact replay of the current pinned24-result prefix')
    with budget._locked():
        after,bound_after=budget._load()
        if canonical(after[:24])!=terminal_binding or bound_after['selection_sha256']!=selection_digest:
            raise ValueError('Terminal selection snapshot changed during readback')
    return dict(selection=result,publication=publication,selection_sha256=selection_digest,
        selection_result_sha256=sha256(canonical(result)).hexdigest(),terminal_pin_prefix_sha256=pins.tip_sha256,
        outcomes=prefix.outcomes)


def _transfer_document(*,blueprint,**evidence_args):
    budget=evidence_args['budget'];require_target_blueprint(budget.plan,blueprint)
    evidence=verify_counted_selection(**evidence_args)
    specs=target_run_specs(budget.plan,evidence_args['selection'],evidence['outcomes'],
        source_snapshot_sha256=budget.plan.source_snapshot_sha256,
        target_width=blueprint.configuration.width,target_parameter_count=blueprint.parameter_count,
        complete_owner_sha256=blueprint.complete_owner_sha256,width_family_sha256=blueprint.width_family_sha256,
        runtime_profile_sha256=blueprint.runtime_profile_sha256,maximum_updates=blueprint.maximum_updates,
        maximum_seconds=blueprint.maximum_seconds,selection_publication=canonical(evidence['publication']))
    targets=[]
    for spec in specs:
        hp=spec['hyperparameters']
        config=replace(blueprint.configuration,readout_mult=float(hp['readout_multiplier']),
                       attn_mult=float(hp['attention_key_multiplier']))
        optimizer=dict(base_lr=float(hp['learning_rate']),weight_decay=.01,betas=(.9,.999),eps=1e-8,
            decay_aliases=blueprint.decay_aliases,decoupled_wd=False,member='target')
        target=dict(spec,configuration=asdict(config),optimizer=optimizer,task_cycle=blueprint.task_cycle,
            gradient_clip_max_norm=blueprint.gradient_clip_max_norm,
            schedule=dict(name='native-warmup-cosine-10pct',total_updates=blueprint.maximum_updates))
        targets.append(dict(target,binding_sha256=sha256(canonical(target)).hexdigest()))
    return dict(schema='Mora3.CountedTargetTransfer/v1',campaign26_sha256=budget.plan.sha256,
        source_snapshot_sha256=budget.plan.source_snapshot_sha256,proxy_campaign_sha256=budget.plan.proxy_campaign_sha256,
        target_blueprint=blueprint.wire(),target_specification_sha256=blueprint.sha256,
        selection=evidence['selection'],selection_sha256=evidence['selection_sha256'],
        selection_result_sha256=evidence['selection_result_sha256'],terminal_pin_prefix_sha256=evidence['terminal_pin_prefix_sha256'],
        proxy_terminal_count=24,eligible_hyperparameter_points=evidence['selection']['eligible_hyperparameter_points'],
        target_seeds=budget.plan.target_seeds,targets=tuple(targets),target_training_started=False,launch_authorized=False,
        transfer_scope='validation-selected-native-hyperparameters-and-target-configuration;not-weights-or-performance')


def _output_path(path,args):
    path=_safe(path)
    if not path.parent.is_dir():raise ValueError('Existing separate target artifact parent required')
    for root in(args['budget'].directory,Path(args['pin_directory']),Path(args['campaign_directory'])):
        if path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Transfer artifact must remain outside immutable accounting/native receipt namespaces')
    return path


def publish_target_transfer(*,blueprint,output,**evidence_args):
    """Publish both target configurations once; never allocate or start either run."""
    budget=evidence_args['budget']
    if budget.snapshot()['reserved_runs']!=24:
        raise ValueError('Target transfer publication requires24 terminal proxies and no target reservations')
    path=_output_path(output,evidence_args)
    if path.exists()or path.with_name(path.name+'.pending').exists():
        raise ValueError('Fresh transfer publication path required; no overwrite or interrupted adoption')
    document=_transfer_document(blueprint=blueprint,**evidence_args)
    # Recheck the complete binding before publication; existing worker owns
    # execution exclusivity. This function creates no worker or lock lease.
    with budget._locked():
        states,selection=budget._load()
        if len(states)!=24 or selection['selection_sha256']!=document['selection_sha256']:
            raise ValueError('Target reservation changed while preparing transfer publication')
        _publish(path,document)
    raw=canonical(document)
    if path.read_bytes()!=raw:raise ValueError('Transferred target artifact readback differs')
    return sha256(raw).hexdigest()


def load_target_transfer(*,blueprint,path,expected_sha256,**evidence_args):
    """Revalidate selection, all histories and preregistration before consuming it."""
    _digest(expected_sha256);path=_output_path(path,evidence_args)
    document=_read(path)
    if sha256(canonical(document)).hexdigest()!=expected_sha256:
        raise ValueError('Pinned target transfer digest differs')
    expected=_transfer_document(blueprint=blueprint,**evidence_args)
    if canonical(document)!=canonical(expected):raise ValueError('Target settings differ from preregistered native transfer')
    return document


def transferred_configuration(document,run_id):
    """Return exact factory and MuAdamW inputs; callers must load/verify first."""
    if (type(document)is not dict or document.get('schema')!='Mora3.CountedTargetTransfer/v1'
            or document.get('proxy_terminal_count')!=24 or document.get('launch_authorized')is not False
            or type(document.get('targets'))not in(tuple,list)or len(document['targets'])!=2):
        raise ValueError('Complete native target transfer document required')
    blueprint=TargetBlueprint.from_wire(document['target_blueprint'])
    if blueprint.sha256!=document.get('target_specification_sha256'):
        raise ValueError('Transferred target blueprint digest differs')
    result=document['selection'];hp=result.get('frozen_transfer_hyperparameters')
    if (hp is None or result.get('full_terminal_trial_count')!=24
            or result.get('target_validation_or_test_used')is not False
            or sha256(canonical(result)).hexdigest()!=document.get('selection_result_sha256')):
        raise ValueError('Native validation-only selection binding required')
    seeds=document.get('target_seeds')
    if (type(seeds)not in(tuple,list)or len(seeds)!=2
            or any(type(seed)is not int or seed<0 for seed in seeds)or len(set(seeds))!=2):
        raise ValueError('Exactly two declared distinct target seeds required')
    expected_ids=tuple(f'full80m-{seed}'for seed in seeds)
    if tuple(t.get('run_id')for t in document['targets'])!=expected_ids:
        raise ValueError('Target order or seed identity differs from its frozen two-run specification')
    matches=[t for t in document['targets']if t.get('run_id')==run_id]
    if len(matches)!=1:raise ValueError('One of the two exact frozen target run IDs required')
    target=matches[0]
    raw={k:v for k,v in target.items()if k!='binding_sha256'}
    if (sha256(canonical(raw)).hexdigest()!=target.get('binding_sha256')
            or target.get('target_specification_sha256')!=blueprint.sha256
            or target.get('selection_sha256')!=document.get('selection_sha256')
            or target.get('selection_result_sha256')!=document.get('selection_result_sha256')
            or canonical(target.get('hyperparameters'))!=canonical(hp)):
        raise ValueError('Transferred run/configuration/selection binding differs')
    expected_config=replace(blueprint.configuration,readout_mult=float(hp['readout_multiplier']),
        attn_mult=float(hp['attention_key_multiplier']))
    expected_optimizer=dict(base_lr=float(hp['learning_rate']),weight_decay=.01,betas=(.9,.999),eps=1e-8,
        decay_aliases=blueprint.decay_aliases,decoupled_wd=False,member='target')
    expected_target=dict(schema='Mora3.Transferred80MTrainingSpec/v2',run_id=run_id,
        seed=seeds[expected_ids.index(run_id)],campaign26_sha256=document['campaign26_sha256'],
        source_snapshot_sha256=blueprint.configuration.source_snapshot_sha256,
        selection_sha256=document['selection_sha256'],selection_result_sha256=document['selection_result_sha256'],
        terminal_pin_prefix_sha256=document['terminal_pin_prefix_sha256'],target_specification_sha256=blueprint.sha256,
        hyperparameters=hp,parameterization='muP',target_width=blueprint.configuration.width,
        target_parameter_count=blueprint.parameter_count,complete_owner_sha256=blueprint.complete_owner_sha256,
        width_family_sha256=blueprint.width_family_sha256,runtime_profile_sha256=blueprint.runtime_profile_sha256,
        maximum_updates=blueprint.maximum_updates,maximum_seconds=blueprint.maximum_seconds,
        target_tuning_allowed=False,launch_authorized=False,configuration=asdict(expected_config),
        optimizer=expected_optimizer,task_cycle=blueprint.task_cycle,gradient_clip_max_norm=blueprint.gradient_clip_max_norm,
        schedule=dict(name='native-warmup-cosine-10pct',total_updates=blueprint.maximum_updates))
    if (canonical(raw)!=canonical(expected_target)
            or document['source_snapshot_sha256']!=blueprint.configuration.source_snapshot_sha256
            or document['proxy_campaign_sha256']!=result['campaign_manifest_sha256']):
        raise ValueError('Selected hyperparameters and every non-tuned target setting must match preregistration')
    return expected_config,expected_optimizer


def require_primary_target_configuration(owner,configuration):
    """Compare native factory input with its actual stored owner representation.

    Source identity lives outside primary_configuration. The constructor adds
    heads/layers/vocabulary identity and converts its numeric multipliers to
    floats. Comparing the two entire dictionaries would incorrectly reject every
    real factory owner. This comparison does not issue training admission.
    """
    if type(owner)is not SourceBoundOwner or type(configuration)is not PrimaryConfiguration:
        raise ValueError('Exact native owner and configuration required')
    configuration.__post_init__();state=owner.get_extra_state()
    requested=asdict(configuration);source=requested.pop('source_snapshot_sha256')
    for key in('readout_mult','attn_mult','encoder_var'):requested[key]=float(requested[key])
    native=state['primary_configuration']
    if (state['source_snapshot_sha256']!=source or type(native)is not dict
            or set(native)!=set(requested)|{'heads','layers','payload_vocabulary_sha256'}
            or canonical({k:native[k]for k in requested})!=canonical(requested)
            or type(native['heads'])is not int or native['heads']!=2
            or type(native['layers'])is not int or native['layers']!=2):
        raise ValueError('Actual native factory configuration differs from transferred target inputs')
    _digest(native['payload_vocabulary_sha256'])
    return state


def require_transfer_target(engine,document,run_id):
    """Validate an ALREADY admitted, freshly constructed real native target engine.

    Calls original admission before inspecting the transfer. Never changes that
    gate, initializes parameters, takes optimizer steps, reserves a run or loads
    proxy weights. Source/operator/runtime scientific qualification is external.
    """
    from .primary_training import NativeMuUpdateEngine
    from .mup_campaign_schedule import NativeMuWarmupCosine
    import torch
    if type(engine)is not NativeMuUpdateEngine or type(engine.model)is not SourceBoundOwner:
        raise ValueError('Exact native full owner and update engine required')
    engine.model.require_training_admission()
    config,request=transferred_configuration(document,run_id)
    target=next(t for t in document['targets']if t['run_id']==run_id)
    state=require_primary_target_configuration(engine.model,config);optimizer=engine.optimizer
    metadata=optimizer._mup_metadata
    if (state['source_snapshot_sha256']!=document['source_snapshot_sha256']
            or state['primary_arm']!='ordinary-attention-mup-primary-v1'
            or unique_parameter_inventory(engine.model)['total_parameters']!=target['target_parameter_count']
            or engine.binding_sha256!=target['binding_sha256']or engine.steps!=0 or optimizer.state
            or engine.task_cycle!=tuple(target['task_cycle'])or optimizer._mup_member!='target'
            or metadata['family_sha256']!=target['width_family_sha256']
            or any(canonical(metadata[key])!=canonical(request[key])for key in request if key!='member')
            or type(engine.scheduler)is not NativeMuWarmupCosine
            or engine.scheduler.state_dict()['total_updates']!=target['maximum_updates']
            or engine._update_policy()['gradient_clip_max_norm']!=target['gradient_clip_max_norm']
            or torch.initial_seed()!=target['seed']):
        raise ValueError('Actual fresh native target does not implement its selected zero-shot transfer configuration')
    optimizer._validate_live()
    return dict(run_id=run_id,target_binding_sha256=target['binding_sha256'],
        selection_sha256=target['selection_sha256'],actual_parameters=target['target_parameter_count'],
        parameterization='muP',master_learning_rate=request['base_lr'],optimizer_updates=0,
        reserves_or_launches_training=False)
