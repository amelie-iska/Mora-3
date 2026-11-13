"""Join the actual v6 schedule and measured source baselines before allocation."""
from collections import Counter
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from mora3.model.full_owner import PrimaryConfiguration
from mora3.research.matched_validation import (
    MatchedValidationCohort, TextValidationSourceRow, ValidationSourceRow,
)
from mora3.research.mup_campaign import NativeTrainRoutePlan
from mora3.research.mup_sweep_selection import MAX_UPDATES
from mora3.research.primary_campaign_assembly import (
    ArtifactReference, _document, _read, _reference, load_primary_campaign_request,
)

SCHEMA = 'Mora3.PrimaryCampaignV6MetadataContextRequest/v1'


@dataclass(frozen=True)
class DiagnosticValidationBaseline:
    task_key: str
    source_sha256: str
    source_family: str
    native_task_name: str
    native_schema: str
    coordinates: tuple[ValidationSourceRow | TextValidationSourceRow, ...]
    observed_targets: int
    loss_unit: str
    target_unit: str
    mean_loss: Decimal
    receipt: ArtifactReference


@dataclass(frozen=True)
class PrimaryCampaignMetadataContext:
    request: ArtifactReference
    source_snapshot: ArtifactReference
    route_replay: ArtifactReference
    route_plan: ArtifactReference
    reference_evidence: ArtifactReference
    coordinate_template: ArtifactReference
    train_routes: tuple[NativeTrainRoutePlan, ...]
    task_cycle: tuple[str, ...]
    prior_observed_updates: int
    prior_consumed_windows: int
    validation_evaluators: tuple[tuple[str, str], ...]
    validation_view_policy_sha256: str
    validation_read_batch_size: int
    baselines: tuple[DiagnosticValidationBaseline, ...]
    reference_configuration: PrimaryConfiguration
    reference_seed: int
    reference_parameter_sha256: str
    measured_scope_width_family_sha256: str

    @property
    def allocation_authorized(self):
        return False

    @property
    def training_authorized(self):
        return False

    @property
    def unresolved(self):
        return (
            'complete_source_runtime_and_verified_full_owner_width_family',
            'semantic_owner_operator_recovery_and_resource_admission',
            'production_executable_v6_route_and_validation_registry',
            'global_split_and_reference_selection_admission',
            'live_runtime_capacity_and_worker_ownership',
        )

    def require_worker_ready(self):
        raise ValueError('Primary campaign remains preallocation only: ' + ', '.join(self.unresolved))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _metadata_ref(value):
    _require(type(value) is dict and set(value) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
             'Exact metadata artifact reference required')
    ref = _reference({'path':value['path'], 'sha256':value['sha256']})
    path = Path(ref.path)
    _require(path.is_relative_to(Path('outputs')) and path.suffix=='.json' and path.name!='private-packet.json'
             and not {'data-intake','data-sources','shards','validation','heldout','held-out','test'}.intersection(path.parts),
             'Only repository output JSON metadata outside source payload roots is supported')
    if 'bytes' in value:
        _require(type(value['bytes']) is int and value['bytes']>0,'Exact positive metadata extent required')
    return ref


def _metadata(root, value, *, canonical_required=False):
    ref = _metadata_ref(value)
    raw = _read(root,ref)
    if 'bytes' in value:
        _require(len(raw)==value['bytes'],'Pinned metadata byte extent differs')
    return ref,_document(raw,canonical_required=canonical_required)


def _sha(value):
    _reference({'path':'digest','sha256':value})
    return value


def _positive(value):
    return type(value) is int and value>0


def _rows_by_key(rows):
    _require(type(rows) is list and all(type(r) is dict and type(r.get('task_key')) is str for r in rows),
             'Exact task row metadata required')
    result = {row['task_key']:row for row in rows}
    _require(len(rows)==len(result),'Duplicate task metadata')
    return result


def _scheduled_scope(root, replay):
    snapshot_ref,snapshot = _metadata(root,replay['scoped_registry_source_snapshot'])
    plan_ref,plan = _metadata(root,replay['route_binding_plan'])
    _require(replay['schema']=='Mora3.ScopedActualFamilyRouteBindingReplay/v1'
             and plan['schema']=='Mora3.ScopedNativeRoutePlan/v1'
             and snapshot['schema']=='Mora3.ScopedModelAndSourceRegistrySnapshot/v1'
             and _metadata_ref(plan['source_snapshot'])==snapshot_ref
             and replay['actual_measured_model_family_sha256']==snapshot['measured_width_family_sha256']
             and replay['same_new_source_snapshot_for_all_metadata_and_routes'] is True
             and replay['exact_train_eval_shared_mcsa_object'] is plan['exact_train_eval_shared_mcsa_object'] is True
             and plan['matched_holdout_coordinates_preserved'] is True
             and all(replay[k] is False for k in ('training_authorized','global_leakage_safe','validation_reference_qualified','role_changes'))
             and all(plan[k] is False for k in ('training_authorized','measured_initial_reference_losses_bound'))
             and all(snapshot[k] is False for k in ('training_authorized','global_split_accepted','role_changes'))
             and replay['gpu_calls']==replay['optimizer_updates']==0,
             'Exact inactive same-snapshot v6 plan and replay required')
    raw = _rows_by_key(plan['train_routes'])
    _require(all(set(row)=={'task_key','initial_cursor_sha256','adapter_sha256','max_windows'} for row in raw.values()),
             'Exact native route plan fields required')
    routes = tuple(NativeTrainRoutePlan(**row) for row in raw.values())
    _require(len(routes)==replay['source_routes']==62 and tuple(raw)==tuple(sorted(raw)),
             'Complete ordered 62-route plan required')
    cycle = tuple(plan['task_cycle'])
    scheduled = Counter(cycle)
    _require(len(cycle)==MAX_UPDATES==replay['replayed_source_updates'] and set(scheduled)==set(raw)
             and scheduled==plan['required_updates'],'Exact full v6 scheduled update denominator required')
    traces = _rows_by_key(replay['replay_receipts'])
    _require(traces.keys()==raw.keys(),'Complete matching v6 replay route axis required')
    total=0
    for route in routes:
        row=traces[route.task_key]
        updates=row['observed_updates']
        windows=row['consumed_windows']
        _require(set(row)=={'task_key','observed_updates','consumed_windows','reason','trace_sha256'}
                 and row['reason'] is None and _positive(updates) and _positive(windows)
                 and updates==scheduled[route.task_key] and updates<=windows<=updates*route.max_windows
                 and _sha(row['trace_sha256']),'Exact bounded completed per-route replay required')
        total+=windows
    _require(total==replay['source_windows_consumed'],'Exact consumed-window denominator required')
    evaluators=tuple(sorted(plan['validation_evaluator_hashes'].items()))
    _require(len(evaluators)==replay['matched_validation_scopes']==5 and all(key in raw and _sha(h) for key,h in evaluators)
             and _sha(plan['validation_view_policy_sha256']) and plan['finite_validation_read_batch_size']==1,
             'Exact five matched validation scopes and bounded read size required')
    return snapshot_ref,snapshot,plan_ref,plan,routes,cycle,total,evaluators


def _coordinate_template(root,snapshot):
    _,recipe=_metadata(root,snapshot['verified_source_factory_recipe'])
    _,primary=_metadata(root,recipe['primary_request'],canonical_required=True)
    _require(primary['schema']=='Mora3.PrimaryCampaignConstructionRequest/v1',
             'Exact original typed coordinate container required')
    ref=_metadata_ref(primary['validation'])
    _read(root,ref)
    return ref,MatchedValidationCohort.load(root/ref.path,ref.sha256)


def _number(value):
    _require(type(value) is str,'Exact measured decimal wire required')
    try:
        result=Decimal(value)
    except InvalidOperation as error:
        raise ValueError('Malformed measured decimal') from error
    _require(result.is_finite() and len(result.as_tuple().digits)<=34 and abs(result.adjusted())<=300,
             'Finite measured decimal with bounded precision required')
    return result


def _baselines(root,reference,template,contract):
    rows={}
    for value in reference['task_receipts']:
        ref,row=_metadata(root,value)
        key=row['task_key']
        _require(type(key) is str and key not in rows,'Duplicate measured validation scope')
        rows[key]=(ref,row)
    _require(rows.keys()==contract['ordered_task_source_sha256'].keys()=={t.task_key for t in template.tasks},
             'Measured reference and exact coordinate task axis differ')
    result=[]
    for task in template.tasks:
        ref,row=rows[task.task_key]
        loss=_number(row['initial_reference_loss'])
        loss_sum=_number(row['loss_sum'])
        _require(row['complete'] is True and row['blocked_reason'] is None and _positive(row['completed_batches'])
                 and _positive(row['observed_targets'])
                 and task.observed_targets==row['observed_targets']==row['expected_targets']
                 and task.source_sha256==row['source_sha256']==contract['ordered_task_source_sha256'][task.task_key]
                 and (task.loss_unit,task.target_unit)==(row['loss_unit'],row['target_unit']),
                 'Measured baseline coordinate, target denominator or units differ')
        with localcontext() as ctx:
            ctx.prec=34
            _require(loss>0 and loss==loss_sum/task.observed_targets,'Measured native reduction does not close')
        result.append(DiagnosticValidationBaseline(task.task_key,task.source_sha256,task.source_family,task.native_task_name,
            task.native_schema,task.coordinates,task.observed_targets,task.loss_unit,task.target_unit,loss,ref))
    return tuple(result)


def _reference_scope(root,request,replay_ref,snapshot_ref,plan,template,evaluators):
    evidence_ref,reference=_metadata(root,request['initial_reference_receipt'])
    _,independent=_metadata(root,request['initial_reference_readback'])
    _,contract=_metadata(root,reference['contract'])
    _require(reference['schema']=='Mora3.ScopedNativeInitialReferenceEvidence/v1'
             and independent['schema']=='Mora3.IndependentV6NativeReferenceReadback/v1'
             and contract['schema']=='Mora3.ScopedInitialReferenceEvaluationDiagnostic/v1'
             and _metadata_ref(contract['source_replay'])==replay_ref
             and _metadata_ref(contract['source_snapshot'])==snapshot_ref
             and independent['v6_snapshot_sha256']==snapshot_ref.sha256
             and independent['reference_sha256']==evidence_ref.sha256
             and dict(evaluators)==contract['exact_evaluators']
             and plan['validation_view_policy_sha256']==contract['view_policy_sha256']
             and plan['finite_validation_read_batch_size']==reference['finite_read_batch_size']==contract['finite_read_batch_size']
             and reference['complete_reference_tasks']==reference['required_reference_tasks']==independent['exact_scope_count']==5
             and reference['exact_train_eval_shared_mcsa_object'] is True
             and reference['original_validation_coordinates_unchanged'] is True
             and reference['native_model_state_unchanged'] is reference['rng_unchanged'] is True
             and independent['same_parameter_bytes'] is independent['native_state_unchanged'] is independent['no_updates'] is True
             and contract['candidate_selection_reference_only'] is contract['no_hyperparameter_selection'] is True
             and type(contract['comparison_step']) is int and contract['comparison_step']==0
             and reference['gpu_calls']==reference['optimizer_updates']==0
             and all(reference[k] is False for k in ('global_split_accepted','training_authorized','reference_losses_admitted_for_sweep_selection'))
             and all(independent[k] is False for k in ('training_authorized','selection_admitted'))
             and contract['training_authorized'] is contract['global_split_accepted'] is False,
             'Exact independently read back diagnostic reference scope required')
    config=contract['model_configuration']
    _require(type(config) is dict and set(config)=={f.name for f in fields(PrimaryConfiguration)},
             'Complete explicit diagnostic reference owner configuration required')
    configuration=PrimaryConfiguration(**config)
    seed=contract['initialization_seed']
    _require(configuration.source_snapshot_sha256==snapshot_ref.sha256
             and _positive(seed) and seed==independent['exact_reference_seed']
             and _sha(contract['model_parameter_sha256']),'Exact measured initial owner and RNG binding required')
    baselines=_baselines(root,reference,template,contract)
    comparisons=_rows_by_key(independent['comparisons'])
    _require(comparisons.keys()=={r.task_key for r in baselines}
             and all(comparisons[r.task_key]['positions']==r.observed_targets
                     and 0<=_number(comparisons[r.task_key]['relative_error'])<=Decimal('0.00002') for r in baselines)
             and sum(t.observed_targets for t in baselines)==reference['observed_targets']
             ==reference['expected_targets']==independent['exact_positions'],
             'Independent matched reference and original denominators differ')
    return evidence_ref,baselines,configuration,seed,contract['model_parameter_sha256']


def build_primary_campaign_context(root, path, expected_sha256):
    """Assemble only the exact authenticated metadata already available for v6."""
    root=Path(root).resolve()
    request_ref,request=_metadata(root,{'path':path,'sha256':expected_sha256},canonical_required=True)
    _require(set(request)=={'schema','source_route_replay','initial_reference_receipt','initial_reference_readback'}
             and request['schema']==SCHEMA,'Exact v6 preallocation context request required')
    replay_ref,replay=_metadata(root,request['source_route_replay'])
    snapshot,snapshot_doc,plan_ref,plan,routes,cycle,windows,evaluators=_scheduled_scope(root,replay)
    coordinate_ref,template=_coordinate_template(root,snapshot_doc)
    evidence,baselines,config,seed,parameters=_reference_scope(root,request,replay_ref,snapshot,plan,template,evaluators)
    return PrimaryCampaignMetadataContext(request_ref,snapshot,replay_ref,plan_ref,evidence,coordinate_ref,routes,cycle,
        len(cycle),windows,evaluators,plan['validation_view_policy_sha256'],plan['finite_validation_read_batch_size'],
        baselines,config,seed,parameters,snapshot_doc['measured_width_family_sha256'])


def check_primary_request_metadata(root, path, expected_sha256, *, context):
    """Match a separately frozen declaration to the scoped evidence without issuing admission."""
    _require(type(context) is PrimaryCampaignMetadataContext,'Exact typed preallocation context required')
    root=Path(root).resolve()
    _require(context==build_primary_campaign_context(root,context.request.path,context.request.sha256),
             'Declared context differs from its pinned v6 evidence')
    _,declaration=_metadata(root,{'path':path,'sha256':expected_sha256},canonical_required=True)
    # The canonical loader authenticates the bytes and semantics after this
    # metadata-only boundary has checked every path it would open.
    for key in ('source_snapshot','campaign','validation','width_family'):
        _metadata_ref(declaration[key])
    for key in ('configurations','evidence'):
        for reference in declaration[key].values():
            _metadata_ref(reference)
    inputs=load_primary_campaign_request(root,path,expected_sha256)
    campaign=inputs.campaign
    cohort=campaign.validation
    _require(inputs.source_snapshot==context.source_snapshot and inputs.source_view_policy=='effective-study-v2'
             and campaign.width_family_sha256==context.measured_scope_width_family_sha256,
             'Declared primary request differs from actual v6 source and measured family')
    _require(campaign.train_routes==context.train_routes and campaign.task_cycle==context.task_cycle
             and campaign.validation_adapter_sha256==context.validation_evaluators
             and cohort.view_policy_sha256==context.validation_view_policy_sha256
             and cohort.read_batch_size==context.validation_read_batch_size
             and cohort.initial_reference_artifact_sha256==context.reference_evidence.sha256,
             'Declared campaign routes, evaluators or measured validation evidence differ')
    _require(len(cohort.tasks)==len(context.baselines) and all(
        (task.task_key,task.source_sha256,task.observed_targets,task.initial_reference_loss)==
        (baseline.task_key,baseline.source_sha256,baseline.observed_targets,baseline.mean_loss)
        for task,baseline in zip(cohort.tasks,context.baselines,strict=True)),
        'Declared validation coordinates, target counts or measured means differ')
    return inputs
