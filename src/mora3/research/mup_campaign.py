"""One finite primary proxy trial on an existing admitted model worker."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import re
import time

import torch

from mora3.data.admet import _json, canonical, digest
from mora3.data.finite_union_native_rows import full9
from mora3.model.full_owner import SourceBoundOwner, unique_parameter_inventory
from .matched_validation import (
    CampaignValidationSources, TextCampaignValidationSources, MatchedValidationCohort, MatchedValidationTask, matched_validation_source_policy_sha256,
)
from .mcsa_campaign_route import MCSACampaignTrainRoute, MCSAValidationBatch
from .source_text_campaign import ReviewedTextCampaignRoute,TextWindow
from .metric_records import MetricRecord
from .mup_campaign_schedule import NativeMuWarmupCosine
from .mup_sweep_selection import (
    FAMILY_MACRO_POLICY, MAX_GPU_SECONDS, MAX_SECONDS, MAX_UPDATES,
    ProxyRunOutcome, ValidationCoordinate, planned_proxy_trials,
)
from .primary_native_routes import (
    NativeMetricIdentity, NativeRouteBinding, run_native_primary_update,
)
from .native_cpu_prefetch import NativeRouteCPUStager
from .primary_training import NativeMuUpdateEngine


_SHA256 = re.compile(r'[a-f0-9]{64}\Z')
_REFERENCE = 'amelie-iska/mup@19814971934ef91dd546f88e913fc963e096d11c'
_EVIDENCE = ('source_role_schedule','complete_owner_evidence','runtime_profile','existing_worker')
_CAMPAIGN_V1 = 'Mora3.FrozenPrimaryProxyCampaign/v1'
_CAMPAIGN_V2 = 'Mora3.FrozenPrimaryProxyCampaign/v2'
_LEGACY_TASK_MACRO = 'task-macro-v1'


def _sha(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _text(value):
    return type(value) is str and bool(value.strip())


def _decimal(value):
    return (type(value) is Decimal and value.is_finite() and len(value.as_tuple().digits) <= 34
            and abs(value.adjusted()) <= 300)


@dataclass(frozen=True)
class NativeTrainRoutePlan:
    task_key: str
    initial_cursor_sha256: str
    adapter_sha256: str
    max_windows: int

    def __post_init__(self):
        if (
            not _text(self.task_key) or not _sha(self.initial_cursor_sha256) or not _sha(self.adapter_sha256)
            or type(self.max_windows) is not int or self.max_windows < 1
        ):
            raise ValueError('Exact train route source cursor, executable adapter and finite window budget required')


@dataclass(frozen=True)
class FrozenPrimaryProxyCampaign:
    source_snapshot_sha256: str
    width_family_sha256: str
    proxy_width: int
    proxy_parameter_count: int
    source_role_schedule_sha256: str
    complete_owner_evidence_sha256: str
    runtime_profile_sha256: str
    existing_worker_sha256: str
    validation: MatchedValidationCohort
    validation_adapter_sha256: tuple[tuple[str, str], ...]
    train_routes: tuple[NativeTrainRoutePlan, ...]
    task_cycle: tuple[str, ...]
    gradient_clip_max_norm: float
    decay_aliases: tuple[str, ...]
    max_one_update_seconds: Decimal
    max_matched_validation_seconds: Decimal
    validation_aggregation: str = FAMILY_MACRO_POLICY

    def __post_init__(self):
        if (
            not all(_sha(value) for value in (self.source_snapshot_sha256,self.width_family_sha256,
                *(getattr(self,key+'_sha256') for key in _EVIDENCE)))
            or type(self.proxy_width) is not int or self.proxy_width < 2 or self.proxy_width % 2
            or type(self.proxy_parameter_count) is not int or self.proxy_parameter_count < 1
            or type(self.validation_aggregation) is not str
            or self.validation_aggregation not in (FAMILY_MACRO_POLICY,_LEGACY_TASK_MACRO)
            or type(self.validation) is not MatchedValidationCohort
            or self.validation.source_snapshot_sha256 != self.source_snapshot_sha256
            or type(self.train_routes) is not tuple or not self.train_routes
            or any(type(route) is not NativeTrainRoutePlan for route in self.train_routes)
            or type(self.task_cycle) is not tuple or not self.task_cycle or any(not _text(x) for x in self.task_cycle)
            or type(self.gradient_clip_max_norm) is not float or not math.isfinite(self.gradient_clip_max_norm)
            or self.gradient_clip_max_norm <= 0
            or type(self.decay_aliases) is not tuple or any(not _text(alias) for alias in self.decay_aliases)
            or self.decay_aliases != tuple(sorted(set(self.decay_aliases)))
            or not all(_decimal(n) and n>0 for n in (self.max_one_update_seconds,self.max_matched_validation_seconds))
            or self.max_one_update_seconds+self.max_matched_validation_seconds > MAX_SECONDS
        ):
            raise ValueError('Exact full-source proxy owner, scheduler/decay and finite worker resource policy required')
        self.validation.__post_init__()
        keys = tuple(route.task_key for route in self.train_routes)
        selected = tuple(task.task_key for task in self.validation.tasks)
        if (
            keys != tuple(sorted(set(keys))) or set(keys) != set(self.task_cycle)
            or set(keys) != set(selected)|{task for task,_ in self.validation.train_only_dispositions}
            or type(self.validation_adapter_sha256) is not tuple
            or any(type(row) is not tuple or len(row)!=2 or not _text(row[0]) or not _sha(row[1])
                   for row in self.validation_adapter_sha256)
            or tuple(key for key,_ in self.validation_adapter_sha256) != selected
        ):
            raise ValueError('Complete exact executable train and matched validation/disposition task axes required')
        for route in self.train_routes:
            route.__post_init__()

    def _validation_family_axis(self):
        return tuple((task.task_key,task.source_family) for task in self.validation.tasks)

    def binding(self):
        self.__post_init__()
        grid = tuple(dict(
            trial_id=t.trial_id,hp_id=t.hp_id,seed=t.seed,learning_rate=str(t.learning_rate),
            readout_multiplier=str(t.readout_multiplier),attention_key_multiplier=str(t.attention_key_multiplier),
        ) for t in planned_proxy_trials())
        result = dict(
            schema=_CAMPAIGN_V1,source_snapshot_sha256=self.source_snapshot_sha256,
            width_family_sha256=self.width_family_sha256,proxy_width=self.proxy_width,
            proxy_parameter_count=self.proxy_parameter_count,primary_arm='ordinary-attention-mup-primary-v1',
            upstream_reference=_REFERENCE,validation_cohort_sha256=self.validation.sha256,
            validation_adapter_sha256=self.validation_adapter_sha256,comparison_update=self.validation.comparison_update,
            exact_trial_grid=grid,train_routes=tuple(asdict(route) for route in self.train_routes),task_cycle=self.task_cycle,
            per_trial_update_limit=MAX_UPDATES,per_trial_wall_seconds=str(MAX_SECONDS),
            aggregate_worker_seconds=str(MAX_GPU_SECONDS),max_one_update_seconds=str(self.max_one_update_seconds),
            max_matched_validation_seconds=str(self.max_matched_validation_seconds),
            optimizer=dict(name='MuAdamW',betas=(.9,.999),eps=1e-8,weight_decay=.01,decoupled_wd=False,
                           decay_aliases=self.decay_aliases,gradient_clip_max_norm=self.gradient_clip_max_norm),
            schedule=dict(name='native-warmup-cosine-10pct',total_updates=MAX_UPDATES,warmup_updates=MAX_UPDATES//10),
            evidence_sha256={key:getattr(self,key+'_sha256') for key in _EVIDENCE},
            admits_training=False,
        )
        if self.validation_aggregation==FAMILY_MACRO_POLICY:
            result.update(schema=_CAMPAIGN_V2,validation_aggregation=dict(
                policy=FAMILY_MACRO_POLICY,task_families=self._validation_family_axis(),
            ))
        return result

    def wire(self):
        return canonical(self.binding())

    @property
    def sha256(self):
        return digest(self.wire())

    @classmethod
    def load(cls,path,expected_sha256,*,validation):
        raw=Path(path).read_bytes()
        if not _sha(expected_sha256) or digest(raw)!=expected_sha256:
            raise ValueError('Exact pinned frozen campaign artifact required')
        try:
            value=_json(raw)
            if (type(validation) is not MatchedValidationCohort
                    or value['validation_cohort_sha256']!=validation.sha256):
                raise ValueError('Exact matched cohort must accompany the frozen campaign')
            schema=value['schema']
            if schema not in (_CAMPAIGN_V1,_CAMPAIGN_V2):
                raise ValueError('Unknown frozen primary campaign schema')
            aggregation = (_LEGACY_TASK_MACRO if schema==_CAMPAIGN_V1
                           else value['validation_aggregation']['policy'])
            optimizer=value['optimizer']
            evidence=value['evidence_sha256']
            result=cls(
                source_snapshot_sha256=value['source_snapshot_sha256'],
                width_family_sha256=value['width_family_sha256'],proxy_width=value['proxy_width'],
                proxy_parameter_count=value['proxy_parameter_count'],
                **{key+'_sha256':evidence[key] for key in _EVIDENCE},
                validation=validation,validation_adapter_sha256=tuple(tuple(x) for x in value['validation_adapter_sha256']),
                train_routes=tuple(NativeTrainRoutePlan(**route) for route in value['train_routes']),
                task_cycle=tuple(value['task_cycle']),gradient_clip_max_norm=optimizer['gradient_clip_max_norm'],
                decay_aliases=tuple(optimizer['decay_aliases']),
                max_one_update_seconds=Decimal(value['max_one_update_seconds']),
                max_matched_validation_seconds=Decimal(value['max_matched_validation_seconds']),
                validation_aggregation=aggregation,
            )
        except (KeyError,AttributeError,TypeError,UnicodeError,InvalidOperation) as exc:
            raise ValueError('Malformed exact frozen primary proxy campaign') from exc
        if result.wire()!=raw:
            raise ValueError('Complete exact canonical campaign wire, grid and policies required')
        return result

    def selection_contract(self):
        contract = self.validation.selection_contract(self.sha256)
        if self.validation_aggregation==FAMILY_MACRO_POLICY:
            return replace(contract,task_families=self._validation_family_axis())
        return contract

    def require_next_trial(self, trial_id, terminal_trial_ids):
        grid = planned_proxy_trials()
        if (
            type(terminal_trial_ids) is not tuple or len(terminal_trial_ids) >= len(grid)
            or terminal_trial_ids != tuple(trial.trial_id for trial in grid[:len(terminal_trial_ids)])
            or trial_id != grid[len(terminal_trial_ids)].trial_id
        ):
            raise ValueError('One next unrun exact planned trial after the complete ordered terminal prefix required')
        return grid[len(terminal_trial_ids)]


@dataclass(frozen=True)
class ValidationBatchReduction:
    loss_sum: Decimal
    observed_targets: int
    missing_reason: str | None

    def __post_init__(self):
        if not _decimal(self.loss_sum) or type(self.observed_targets) is not int or self.observed_targets < 0:
            raise ValueError('Exact finite native validation loss sum and observed denominator required')
        if self.observed_targets == 0:
            if self.loss_sum != 0 or not _text(self.missing_reason):
                raise ValueError('Missing validation targets require an explicit reason and zero contribution')
        elif self.missing_reason is not None:
            raise ValueError('Observed validation reduction cannot carry a missing-target reason')


def reduce_matched_validation(task, reductions):
    if type(task) is not MatchedValidationTask:
        raise ValueError('Exact matched validation task required')
    reductions = tuple(reductions)
    if not reductions or any(type(row) is not ValidationBatchReduction for row in reductions):
        raise ValueError('Every frozen validation batch requires an exact reduction or explicit missing reason')
    for row in reductions:
        row.__post_init__()
    observed = sum(row.observed_targets for row in reductions)
    if observed != task.observed_targets:
        raise ValueError('Observed validation target count differs from the predeclared matched cohort')
    with localcontext() as ctx:
        ctx.prec = 34
        mean = sum((row.loss_sum for row in reductions),Decimal(0))/observed
    return ValidationCoordinate(task.task_key,task.source_sha256,'validation',observed,mean)


def _file_sha256(path):
    before = full9(path)
    h = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''):
            h.update(block)
    if full9(path) != before:
        raise ValueError('Frozen campaign evidence artifact changed during verification')
    return h.hexdigest()


def executable_adapter_sha256(adapter, *, root):
    """Pin route code and any supported bound semantic state."""
    from .assay_campaign_routes import FreeSolvCampaignAdapter, NciCampaignAdapter
    from .gperturb_campaign_routes import GPerturbNativeCampaignAdapter
    from .pdb_campaign_routes import PDBCampaignAdapter
    from .deepbacs_campaign_routes import DeepBacsNativeRouteAdapter
    from .deepbacs_forward_campaign_route import DeepBacsForwardCampaignAdapter
    from .native_admet_reaction_campaign_routes import NativeAdmetReactionCampaignAdapter
    from .native_scalar_campaign_routes import TrainOnlyScalarCampaignAdapter
    from .native_annotation_campaign_routes import NativeSourceRouteAdapter
    from .uniprot_campaign_routes import UniProtNativeCampaignAdapter
    from .uniprot_full_record_campaign_route import UniProtFullRecordCampaignAdapter
    from .goa_campaign_route import GOACampaignAdapter
    from .rfam_campaign_routes import RfamCampaignAdapter
    from .rosetta_unconditional_campaign_routes import RosettaUnconditionalCampaignAdapter
    from .rosetta_cdrp_campaign_routes import RosettaCDRPCampaignAdapter
    from .lincs_pilot1_campaign_routes import LINCSCampaignAdapter
    from .native_prior_campaign import NativePriorCampaignAdapter
    from .tox21_campaign_routes import Tox21CampaignAdapter

    from .source_text_campaign import ReviewedTextCampaignRoute
    from mora3.extensions.scri.native_routes import SCRICampaignAdapter
    binding = None
    if inspect.isfunction(adapter):
        function = adapter
    elif inspect.ismethod(adapter) and type(adapter.__self__) is SCRICampaignAdapter:
        function=adapter.__func__
        if function not in (SCRICampaignAdapter.prepare,SCRICampaignAdapter.evaluate_validation):
            raise ValueError('Exact typed SCRI executable method required')
        binding=adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) is Tox21CampaignAdapter:
        function = adapter.__func__
        if function not in (Tox21CampaignAdapter.prepare,Tox21CampaignAdapter.evaluate_validation):
            raise ValueError('Exact typed Tox21 executable adapter method required')
        binding = adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) in (NciCampaignAdapter,FreeSolvCampaignAdapter):
        function = adapter.__func__
        adapter_type = type(adapter.__self__)
        if function not in (adapter_type.prepare,adapter_type.evaluate_validation):
            raise ValueError('Exact typed native assay executable adapter method required')
        binding = adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) is NativePriorCampaignAdapter:
        function = adapter.__func__
        if function is not NativePriorCampaignAdapter.prepare:
            raise ValueError('Exact train-only native prior executable adapter method required')
        binding = adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) in (
        NativeSourceRouteAdapter,UniProtNativeCampaignAdapter,DeepBacsNativeRouteAdapter,
        NativeAdmetReactionCampaignAdapter,TrainOnlyScalarCampaignAdapter,
        GPerturbNativeCampaignAdapter,PDBCampaignAdapter,
        UniProtFullRecordCampaignAdapter,GOACampaignAdapter,RfamCampaignAdapter,
        RosettaUnconditionalCampaignAdapter,RosettaCDRPCampaignAdapter,LINCSCampaignAdapter,
        DeepBacsForwardCampaignAdapter,
    ):
        function = adapter.__func__
        if function is not type(adapter.__self__).prepare:
            raise ValueError('Exact train-only annotation preparation method required')
        binding = adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) is ReviewedTextCampaignRoute:
        function=adapter.__func__
        if function not in (ReviewedTextCampaignRoute.prepare,ReviewedTextCampaignRoute.evaluate_validation):
            raise ValueError('Exact reviewed text executable method required')
        binding=adapter.__self__.executable_binding()
    elif inspect.ismethod(adapter) and type(adapter.__self__) is MCSACampaignTrainRoute:
        function = adapter.__func__
        if function not in (MCSACampaignTrainRoute.prepare,MCSACampaignTrainRoute.evaluate_validation):
            raise ValueError('Exact typed M-CSA executable adapter method required')
        binding = dict(route=adapter.__self__.binding(),validation_policy_sha256=adapter.__self__.validation_policy_sha256)
        materialization=adapter.__self__.train_materialization_binding()
        if materialization is not None:
            binding['train_materialization']=materialization
    else:
        raise ValueError('Exact module function or supported typed executable route adapter required')
    if (
        not inspect.isfunction(function) or function.__closure__ is not None
        or any(part in function.__qualname__ for part in ('<locals>','<lambda>'))
        or not function.__module__.startswith('mora3.')
    ):
        raise ValueError('Exact inspectable project executable route adapter required')
    root = Path(root).resolve()
    path = Path(inspect.getfile(function)).resolve()
    path.relative_to(root/'src'/'mora3')
    identity = [function.__module__,function.__qualname__,_file_sha256(path)]
    if binding is not None:
        identity.append(binding)
    return digest(canonical(identity))


def _prefetch_runtime_options(path, expected_sha256, planned_tasks):
    # Only an explicitly enabled profile pinned by the same campaign can
    # change the worker settings used for its timed trial comparisons.
    with path.open('rb') as handle:
        raw = handle.read(2*1024*1024+1)
    if len(raw)>2*1024*1024:
        return None
    if sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Exact runtime profile bytes changed during CPU staging setup')
    try:
        profile = json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError):
        return None
    if type(profile) is not dict or 'native_cpu_prefetch' not in profile:
        return None
    options = profile['native_cpu_prefetch']
    if type(options) is not dict or set(options)!={'workers','task_keys'}:
        raise ValueError('Exact native CPU prefetch profile settings required')
    workers,keys = options['workers'],options['task_keys']
    if (type(workers) is not int or not 1<=workers<=4
            or type(keys) is not list or not keys
            or any(type(key) is not str for key in keys)
            or len(set(keys))!=len(keys) or not set(keys)<=set(planned_tasks)):
        raise ValueError('Bounded CPU workers and distinct admitted prefetch routes required')
    return workers,tuple(keys)


def _require_runtime(engine,campaign,trial,bindings,validation_view,validation_evaluators,metric_identity,root,evidence_paths):
    root = Path(root).resolve()
    if type(evidence_paths) is not dict or set(evidence_paths) != set(_EVIDENCE):
        raise ValueError('Exact existing worker, full owner, source schedule and real runtime evidence paths required')
    for key in _EVIDENCE:
        path = (root/evidence_paths[key]).resolve()
        path.relative_to(root)
        if _file_sha256(path) != getattr(campaign,key+'_sha256'):
            raise ValueError('Pinned full campaign source/owner/runtime/worker evidence differs')
    configuration = engine.model.get_extra_state()
    declared = configuration['primary_configuration']
    optimizer = engine.optimizer
    settings = optimizer._mup_metadata
    if (
        configuration['source_snapshot_sha256'] != campaign.source_snapshot_sha256
        or configuration['primary_arm'] != 'ordinary-attention-mup-primary-v1'
        or declared['width'] != campaign.proxy_width or declared['base_width'] != campaign.proxy_width
        or declared['readout_mult'] != float(trial.readout_multiplier)
        or declared['attn_mult'] != float(trial.attention_key_multiplier)
        or unique_parameter_inventory(engine.model)['total_parameters'] != campaign.proxy_parameter_count
        or engine.binding_sha256 != campaign.sha256 or engine.task_cycle != campaign.task_cycle
        or engine.steps != 0 or optimizer.state or optimizer._mup_member != 'base'
        or settings['reference'] != _REFERENCE or settings['family_sha256'] != campaign.width_family_sha256
        or settings['base_lr'] != float(trial.learning_rate) or settings['weight_decay'] != .01
        or settings['betas'] != (.9,.999) or settings['eps'] != 1e-8 or settings['decoupled_wd'] is not False
        or tuple(sorted(settings['decay_aliases'])) != campaign.decay_aliases
        or type(engine.scheduler) is not NativeMuWarmupCosine
        or engine._update_policy()['gradient_clip_max_norm'] != campaign.gradient_clip_max_norm
        or type(metric_identity) is not NativeMetricIdentity or metric_identity.trial_id != trial.trial_id
        or metric_identity.seed != trial.seed or torch.initial_seed() != trial.seed
    ):
        raise ValueError('Exact fresh source-bound proxy owner, trial optimizer, schedule and RNG binding required')
    metric_identity.__post_init__()
    optimizer._validate_live()
    scheduler = engine.scheduler.state_dict()
    if scheduler['total_updates']!=MAX_UPDATES or scheduler['warmup_updates']!=MAX_UPDATES//10:
        raise ValueError('Exact preregistered proxy schedule horizon required')
    if engine.profile_stage_controller is None:
        raise ValueError('Active primary ProfileGFN requires its exact frozen-stage controller')
    if type(bindings) is not dict or set(bindings) != {route.task_key for route in campaign.train_routes}:
        raise ValueError('Every planned training task requires a real executable route binding')
    for planned in campaign.train_routes:
        binding = bindings[planned.task_key]
        if (
            type(binding) is not NativeRouteBinding or binding.route.task_key != planned.task_key
            or binding.route.source_snapshot_sha256 != campaign.source_snapshot_sha256
            or engine.source_cursors[planned.task_key] != binding.route.initial_cursor
            or digest(binding.route.initial_cursor) != planned.initial_cursor_sha256
            or executable_adapter_sha256(binding.prepare,root=root) != planned.adapter_sha256
        ):
            raise ValueError('Exact original train cursor/source and live route executable required')
        binding.__post_init__()
        if type(binding.route) is MCSACampaignTrainRoute:
            binding.route.require_task_cycle_capacity(campaign.task_cycle,MAX_UPDATES)
    if (
        matched_validation_source_policy_sha256(validation_view) != campaign.validation.view_policy_sha256
        or type(validation_evaluators) is not dict or set(validation_evaluators) != {k for k,_ in campaign.validation_adapter_sha256}
        or any(executable_adapter_sha256(validation_evaluators[k],root=root) != value
               for k,value in campaign.validation_adapter_sha256)
    ):
        raise ValueError('Exact live original validation source and executable task evaluators required')
    mcsa_validation_keys = {key for key,binding in bindings.items()
                            if type(binding.route) is MCSACampaignTrainRoute and key in validation_evaluators}
    legacy_view=validation_view.legacy if type(validation_view) is TextCampaignValidationSources else validation_view
    if type(validation_view) is TextCampaignValidationSources:
        for route in validation_view.text_routes:
            evaluator=validation_evaluators.get(route.task_key)
            if (route.task_key not in bindings or bindings[route.task_key].route is not route
                or not inspect.ismethod(evaluator) or evaluator.__self__ is not route
                or evaluator.__func__ is not ReviewedTextCampaignRoute.evaluate_validation):
                raise ValueError('Exact common reviewed text train and validation route required')
    if mcsa_validation_keys and type(legacy_view) is not CampaignValidationSources:
        raise ValueError('Matched M-CSA evaluation requires the exact mixed validation source')
    if type(legacy_view) is CampaignValidationSources:
        route = legacy_view.mcsa
        evaluator = validation_evaluators.get(route.task_key)
        if (route.task_key not in bindings or bindings[route.task_key].route is not route
                or not inspect.ismethod(evaluator) or evaluator.__self__ is not route
                or evaluator.__func__ is not MCSACampaignTrainRoute.evaluate_validation):
            raise ValueError('Exact common reviewed M-CSA train and matched validation route required')
    return _prefetch_runtime_options((root/evidence_paths['runtime_profile']).resolve(),
                                    campaign.runtime_profile_sha256,bindings)


@dataclass(frozen=True)
class ValidationTaskExecution:
    coordinate: ValidationCoordinate
    original_source_rows: int
    reductions: tuple[ValidationBatchReduction, ...]


def _validation_metrics(cohort, details, identity):
    records = []
    for task,detail in zip(cohort.tasks,details,strict=True):
        coordinate = detail.coordinate
        if (coordinate.task_key,coordinate.source_sha256,coordinate.split,coordinate.observed_targets)!=(
            task.task_key,task.source_sha256,'validation',task.observed_targets,
        ):
            raise ValueError('Exact matched validation metric coordinate required')
        evidence = digest(canonical([
            cohort.sha256,task.source_sha256,detail.original_source_rows,
            tuple((str(row.loss_sum),row.observed_targets,row.missing_reason) for row in detail.reductions),
        ]))
        common = dict(
            run_id=identity.run_id,trial_id=identity.trial_id,segment_id=identity.segment_id,seed=identity.seed,
            step=cohort.comparison_update,x_axis='optimizer_step',task=task.task_key,split='validation',
            scope='validation',source_sha256=task.source_sha256,evidence_sha256=evidence,
        )
        records.extend((
            MetricRecord(**common,name='matched_native_validation_objective',value=float(coordinate.mean_loss),
                         unit=task.loss_unit,direction='minimize',
                         numerator=float(sum((row.loss_sum for row in detail.reductions),Decimal(0))),
                         denominator=coordinate.observed_targets),
            MetricRecord(**common,name='matched_validation_observed_targets',value=coordinate.observed_targets,
                         unit=task.target_unit,direction='none'),
            MetricRecord(**common,name='matched_validation_original_rows',value=detail.original_source_rows,
                         unit='original source rows',direction='none'),
        ))
    return tuple(records)


def _evaluate(model,cohort,view,evaluators):
    modes = tuple((module,module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            results = []
            for task in cohort.tasks:
                batches = cohort.batches(view,task.task_key)
                reductions = tuple(evaluators[task.task_key](model,batch) for batch in batches)
                coordinate = reduce_matched_validation(task,reductions)
                results.append(ValidationTaskExecution(
                    coordinate,sum(batch.original_source_rows if type(batch) in (MCSAValidationBatch,TextWindow)
                                   else batch.native_rows.num_rows for batch in batches),reductions,
                ))
            return tuple(results)
    finally:
        for module,mode in modes:
            module.training = mode


@dataclass(frozen=True)
class ProxyTrialExecution:
    trial_id: str
    campaign_manifest_sha256: str
    status: str
    completed_updates: int
    measured_wall_seconds: Decimal
    validation_exposure_sha256: str | None
    comparison_update: int | None
    validation: tuple[ValidationCoordinate, ...]
    validation_details: tuple[ValidationTaskExecution, ...]
    metrics: tuple[MetricRecord, ...]
    failure_reason: str | None

    def terminal_outcome(self, *, run_artifact_sha256):
        if not _sha(run_artifact_sha256) or self.measured_wall_seconds > MAX_SECONDS:
            raise ValueError('Immutable within-budget terminal run artifact required for the selector')
        return ProxyRunOutcome(
            self.trial_id,self.campaign_manifest_sha256,run_artifact_sha256,self.status,self.completed_updates,
            self.measured_wall_seconds,self.validation_exposure_sha256,self.comparison_update,
            self.validation,self.failure_reason,
        )


class MetricCaptureError(RuntimeError):
    """Durable evidence capture failed; never classify as a native data refusal."""


def _capture_metrics(capture,records):
    if capture is None:
        return
    try:
        capture(records)
    except MetricCaptureError:
        raise
    except Exception as error:
        raise MetricCaptureError('Durable metric capture failed') from error


def run_primary_proxy_trial(engine,campaign,*,trial_id,terminal_trial_ids,bindings,validation_view,
        validation_evaluators,metric_identity,root,evidence_paths=None,metric_capture=None):
    """Run one fresh admitted trial; existing worker owns timeout and durable receipts.

    Optional capture receives each update/validation metric batch synchronously;
    its cost is included in measured runtime envelopes. Capture failures propagate
    outside generic native data refusals. Existing callers may omit the hook.
    """
    if metric_capture is not None and not callable(metric_capture):
        raise ValueError('Metric capture must be callable or absent')
    if type(engine) is not NativeMuUpdateEngine or type(engine.model) is not SourceBoundOwner:
        raise ValueError('Exact native update engine and registered source-bound model required')
    engine.model.require_training_admission()
    if type(campaign) is not FrozenPrimaryProxyCampaign:
        raise ValueError('Exact frozen primary proxy campaign required')
    engine.model.require_training_admission(campaign_sha256=campaign.sha256,trial_id=trial_id)
    started = time.monotonic()
    trial = campaign.require_next_trial(trial_id,terminal_trial_ids)
    prefetch = _require_runtime(engine,campaign,trial,bindings,validation_view,validation_evaluators,
                                metric_identity,root,evidence_paths)
    limits = {route.task_key:route.max_windows for route in campaign.train_routes}
    validation, validation_details, metrics = (), (), []
    status, reason = 'completed', None
    elapsed = lambda: Decimal(str(time.monotonic()-started))
    staging = nullcontext() if prefetch is None else NativeRouteCPUStager(workers=prefetch[0])
    lookahead = {} if prefetch is None else {key:limits[key] for key in prefetch[1]}
    with staging as stager:
        while engine.steps < MAX_UPDATES:
            needs_validation = engine.steps+1 == campaign.validation.comparison_update
            reserve = campaign.max_one_update_seconds
            if needs_validation:
                reserve += campaign.max_matched_validation_seconds
            if elapsed()+reserve > MAX_SECONDS:
                status = 'wall_time_censored' if validation else 'insufficient_exposure'
                reason = None if validation else 'worker wall budget ended before matched validation exposure'
                break
            try:
                tick = elapsed()
                kwargs = {} if stager is None else dict(cpu_stager=stager,
                    prefetch_limits=lookahead if engine.steps+1<MAX_UPDATES else {})
                result = run_native_primary_update(engine,bindings,metric_identity=metric_identity,
                                                   max_windows=limits[engine.next_task],**kwargs)
                metrics.extend(result.metrics)
                _capture_metrics(metric_capture,result.metrics)
                if elapsed()-tick > campaign.max_one_update_seconds:
                    status,reason = 'refused','exact measured single-update runtime envelope exceeded'
                    break
                if result.status != 'updated':
                    status,reason = 'insufficient_exposure',result.status
                    break
                if needs_validation:
                    tick = elapsed()
                    validation_details = _evaluate(engine.model,campaign.validation,validation_view,validation_evaluators)
                    validation_metrics = _validation_metrics(campaign.validation,validation_details,metric_identity)
                    metrics.extend(validation_metrics)
                    _capture_metrics(metric_capture,validation_metrics)
                    validation = tuple(item.coordinate for item in validation_details)
                    if elapsed()-tick > campaign.max_matched_validation_seconds:
                        status,reason = 'missing_metrics','matched validation exceeded its runtime envelope'
                        break
            except torch.OutOfMemoryError:
                status,reason = 'oom','existing worker allocation failed within this trial'
                break
            except ValueError:
                status,reason = 'refused','exact native source, objective or update preflight refused'
                break
            except KeyboardInterrupt:
                status,reason = 'interrupted','existing worker interrupted at current recorded update'
                break
    spent = elapsed()
    if spent > MAX_SECONDS:
        status,reason = 'refused','existing worker wall-time bound exceeded'
    if status in ('completed','wall_time_censored') and not validation:
        status,reason = 'missing_metrics','complete matched validation exposure absent'
    if status not in ('completed','wall_time_censored'):
        validation = ()
    return ProxyTrialExecution(
        trial.trial_id,campaign.sha256,status,engine.steps,spent,campaign.validation.sha256 if validation else None,
        campaign.validation.comparison_update if validation else None,validation,validation_details,tuple(metrics),reason,
    )
