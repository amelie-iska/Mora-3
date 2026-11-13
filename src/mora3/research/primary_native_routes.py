"""Deterministic train-only source windows and an admission-gated update entry point."""

from dataclasses import dataclass, replace
from itertools import islice
import inspect
import re
from typing import Callable

from mora3.data.admet import _json, canonical, digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, SelectedNativeSourceBatch
from mora3.model.full_owner import SourceBoundOwner
from .mcsa_campaign_route import MCSACampaignTrainRoute, MCSATrainPassWindow
from .source_text_campaign import ReviewedTextCampaignRoute
from .metric_records import MetricRecord
from .primary_training import NativeMuUpdateEngine


_SHA256 = re.compile(r'[a-f0-9]{64}\Z')
_CURSOR_SCHEMA = 'Mora3.FiniteNativeTrainRouteCursor/v1'
_ROUTE_FIELDS = ('source_family', 'task_name', 'native_schema', 'source_revision')


def _sha256(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _text(value):
    return type(value) is str and bool(value.strip())


def finite_view_policy_sha256(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact finite native source view required')
    return digest(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


@dataclass(frozen=True)
class NativeRouteWindow:
    batch: SelectedNativeSourceBatch
    cursor_after: bytes
    evidence_sha256: str

    @property
    def original_source_rows(self):
        return self.batch.native_rows.num_rows


class FiniteNativeTrainRoute:
    """One finite pass over an explicit immutable source-shard order.

    The cursor uses literal absolute original ordinals, so excluded rows and
    other task directions cannot become the next source record. The reader
    resumes by reopening only the current shard and any later ordered shards.
    """

    def __init__(self, view, *, task_key, source_snapshot_sha256, native_view_policy_sha256,
                 source_family, task_name, native_schema, source_revisions, source_paths, batch_size):
        if (
            not all(_text(value) for value in (task_key, source_family, task_name, native_schema))
            or not _sha256(source_snapshot_sha256) or not _sha256(native_view_policy_sha256)
            or finite_view_policy_sha256(view) != native_view_policy_sha256
            or type(source_revisions) is not tuple or not source_revisions
            or any(not _text(value) for value in source_revisions) or len(set(source_revisions)) != len(source_revisions)
            or type(source_paths) is not tuple or not source_paths
            or any(not _text(value) for value in source_paths) or len(set(source_paths)) != len(source_paths)
            or type(batch_size) is not int or batch_size < 1
        ):
            raise ValueError('Exact train route, source snapshot/view, ordered paths and batch size required')
        identities = frozenset((source_family,task_name,native_schema,rev) for rev in source_revisions)
        path_set = set(source_paths)
        by_path = {}
        observed_revisions = set()
        for descriptor in view.descriptors:
            shard = descriptor['source_shard']
            if shard['path'] not in path_set:
                continue
            eligible_revisions = {row['source_revision'] for row in descriptor['source_task_schema_counts']
                if tuple(row[field] for field in _ROUTE_FIELDS) in identities
                and row['split'] == 'train' and row['eligible_view_disposition'] == 0
                and type(row['rows']) is int and row['rows'] > 0}
            if not _sha256(shard['sha256']) or type(shard['rows']) is not int or shard['rows'] < 1 or not eligible_revisions:
                raise ValueError('Exact original eligible train task and shard identity required')
            by_path[shard['path']] = canonical(shard)
            observed_revisions.update(eligible_revisions)
        if set(by_path) != path_set or observed_revisions != set(source_revisions):
            raise ValueError('Every explicit route path and release must exist in the bound original view')
        self.task_key = task_key
        self.source_snapshot_sha256 = source_snapshot_sha256
        self.native_view_policy_sha256 = native_view_policy_sha256
        self._view = view
        self._identity = identities
        self._paths = source_paths
        self._shard_refs = tuple(by_path[path] for path in source_paths)
        self._batch_size = batch_size
        self._route_sha256 = digest(canonical(dict(
            schema='Mora3.FiniteNativeTrainRoute/v1', role='train', task_key=task_key,
            source_snapshot_sha256=source_snapshot_sha256, native_view_policy_sha256=native_view_policy_sha256,
            identities=sorted(identities), original_source_shards=[_json(ref) for ref in self._shard_refs],
            raw_batch_size=batch_size, epoch_policy='single-explicit-source-pass',
        )))

    def _cursor(self, shard_index, next_ordinal):
        return canonical(dict(schema=_CURSOR_SCHEMA, route_sha256=self._route_sha256, role='train',
                              shard_index=shard_index, next_original_ordinal=next_ordinal))

    @property
    def initial_cursor(self):
        return self._cursor(0,0)

    def _cursor_position(self, raw):
        if type(raw) is not bytes:
            raise ValueError('Exact serialized native train cursor required')
        try:
            state = _json(raw)
        except (ValueError,UnicodeError) as exc:
            raise ValueError('Exact serialized native train cursor required') from exc
        if type(state) is not dict:
            raise ValueError('Exact serialized native train cursor required')
        index, ordinal = state.get('shard_index'), state.get('next_original_ordinal')
        if (
            type(index) is not int or not 0 <= index <= len(self._paths)
            or type(ordinal) is not int or ordinal < 0 or raw != self._cursor(index,ordinal)
            or index == len(self._paths) and ordinal != 0
            or index < len(self._paths) and ordinal >= _json(self._shard_refs[index])['rows']
        ):
            raise ValueError('Native train cursor route/snapshot or original ordinal differs')
        return index,ordinal

    def iter_from(self, cursor):
        import pyarrow as pa

        if finite_view_policy_sha256(self._view) != self.native_view_policy_sha256:
            raise ValueError('Native train route source view snapshot changed')
        first, minimum = self._cursor_position(cursor)
        for shard_index in range(first,len(self._paths)):
            path = self._paths[shard_index]
            reference = self._shard_refs[shard_index]
            shard_rows = _json(reference)['rows']
            lower = minimum if shard_index == first else 0
            for original in self._view.iter_native_task_batches(
                split='train',source_paths=(path,),batch_size=self._batch_size,
            ):
                if canonical(original.source_shard) != reference:
                    raise ValueError('Native train shard reference differs from exact route snapshot')
                rows = original.native_rows
                required = set(_ROUTE_FIELDS)|{'task_id','native_payload_sha256','split'}
                if not required <= set(rows.schema.names):
                    raise ValueError('Original complete train task coordinate fields required')
                columns = [rows[name].to_pylist() for name in _ROUTE_FIELDS]
                identity = zip(*columns,strict=True)
                positions = [i for i,(ordinal,task) in enumerate(zip(original.source_row_ordinals,identity,strict=True))
                             if ordinal >= lower and task in self._identity]
                if not positions:
                    continue
                ordinals = tuple(original.source_row_ordinals[i] for i in positions)
                batch = SelectedNativeSourceBatch(original.source_shard,ordinals,'train',
                    rows.take(pa.array(positions,type=pa.int64())))
                self._view.require_native_rows(batch,split='train')
                ids = batch.native_rows['task_id'].to_pylist()
                payloads = batch.native_rows['native_payload_sha256'].to_pylist()
                if any(not _text(value) for value in ids) or any(not _sha256(value) for value in payloads):
                    raise ValueError('Complete native train task and payload provenance required')
                next_ordinal = original.source_row_ordinals[-1]+1
                after = (self._cursor(shard_index+1,0) if next_ordinal==shard_rows
                         else self._cursor(shard_index,next_ordinal))
                evidence = digest(canonical([
                    self._route_sha256,path,original.source_shard['sha256'],
                    tuple(zip(ordinals,ids,payloads,strict=True)),
                ]))
                yield NativeRouteWindow(batch,after,evidence)


@dataclass(frozen=True)
class RepeatedFiniteNativeTrainRoute:
    """Repeat the exact original source order under an explicit finite pass budget."""
    original: FiniteNativeTrainRoute
    max_passes: int

    def __post_init__(self):
        if type(self.original) is not FiniteNativeTrainRoute or type(self.max_passes) is not int or self.max_passes<2:
            raise ValueError('Exact original single-pass route and explicit finite repeat count required')

    @property
    def task_key(self):
        return self.original.task_key

    @property
    def source_snapshot_sha256(self):
        return self.original.source_snapshot_sha256

    @property
    def native_view_policy_sha256(self):
        return self.original.native_view_policy_sha256

    @property
    def _pass_policy_sha256(self):
        return digest(canonical(dict(schema='Mora3.RepeatedFiniteOriginalTrainPasses/v1',
            single_pass_initial_cursor_sha256=digest(self.original.initial_cursor),
            maximum_passes=self.max_passes,order='same-original-ordered-shard-and-absolute-ordinal-per-pass',
            role='train')))

    def _cursor(self, pass_index, inner):
        return canonical(dict(schema='Mora3.RepeatedFiniteOriginalTrainCursor/v1',
            pass_policy_sha256=self._pass_policy_sha256,role='train',pass_index=pass_index,
            inner_cursor_hex=inner.hex()))

    @property
    def initial_cursor(self):
        return self._cursor(0,self.original.initial_cursor)

    def _position(self, cursor):
        if type(cursor) is not bytes:
            raise ValueError('Exact serialized repeated original train cursor required')
        try:
            document = _json(cursor)
            pass_index = document['pass_index']
            inner = bytes.fromhex(document['inner_cursor_hex'])
        except (KeyError,TypeError,ValueError,UnicodeError) as exc:
            raise ValueError('Exact serialized repeated original train cursor required') from exc
        if (type(pass_index) is not int or not 0 <= pass_index < self.max_passes
                or cursor!=self._cursor(pass_index,inner)):
            raise ValueError('Exact original train cursor, pass and repeated source policy required')
        self.original._cursor_position(inner)
        return pass_index,inner

    def iter_from(self, cursor):
        first,inner = self._position(cursor)
        for pass_index in range(first,self.max_passes):
            start = inner if pass_index==first else self.original.initial_cursor
            for window in self.original.iter_from(start):
                yield NativeRouteWindow(window.batch,self._cursor(pass_index,window.cursor_after),
                    digest(canonical([self._pass_policy_sha256,pass_index,window.evidence_sha256])))


@dataclass(frozen=True)
class PreparedNativeTargets:
    observed_positions: int
    unavailable_reason: str | None
    evaluate: Callable | None

    def __post_init__(self):
        if type(self.observed_positions) is not int or self.observed_positions < 0:
            raise ValueError('Exact observed native target count required')
        if self.observed_positions == 0:
            if not _text(self.unavailable_reason) or self.evaluate is not None:
                raise ValueError('Unavailable native targets require a reason and no model execution')
        elif self.unavailable_reason is not None or not callable(self.evaluate):
            raise ValueError('Observed native targets require one prepared objective')


@dataclass(frozen=True)
class NativeRouteBinding:
    route: FiniteNativeTrainRoute | RepeatedFiniteNativeTrainRoute | MCSACampaignTrainRoute | ReviewedTextCampaignRoute
    prepare: Callable
    objective_unit: str
    target_unit: str

    def __post_init__(self):
        if (
            type(self.route) not in (FiniteNativeTrainRoute,RepeatedFiniteNativeTrainRoute,MCSACampaignTrainRoute,ReviewedTextCampaignRoute) or not callable(self.prepare)
            or not _text(self.objective_unit) or not _text(self.target_unit)
        ):
            raise ValueError('Exact native train route, target preparation and declared metric units required')
        if type(self.route) is ReviewedTextCampaignRoute and (
            not inspect.ismethod(self.prepare) or self.prepare.__self__ is not self.route
            or self.prepare.__func__ is not ReviewedTextCampaignRoute.prepare
            or (self.objective_unit,self.target_unit)!=(self.route.loss_unit,self.route.target_unit)):
            raise ValueError('Exact reviewed native text train method and metric units required')
        if type(self.route) is MCSACampaignTrainRoute and (
            not inspect.ismethod(self.prepare) or self.prepare.__self__ is not self.route
            or self.prepare.__func__ is not MCSACampaignTrainRoute.prepare
            or (self.objective_unit,self.target_unit)!=(self.route.loss_unit,self.route.target_unit)
        ):
            raise ValueError('Exact reviewed M-CSA train objective and native target units required')


def repeat_native_train_route(binding, *, max_passes):
    """Opt in without changing a historical source route, adapter or single-pass cursor."""
    if type(binding) is not NativeRouteBinding:
        raise ValueError('Exact native train binding required for finite source repetition')
    return replace(binding,route=RepeatedFiniteNativeTrainRoute(binding.route,max_passes))


@dataclass(frozen=True)
class UnavailableNativeWindow:
    evidence_sha256: str
    original_source_rows: int
    reason: str


@dataclass(frozen=True)
class NativeRouteSelection:
    status: str
    window: NativeRouteWindow | MCSATrainPassWindow | None
    targets: PreparedNativeTargets | None
    unavailable: tuple[UnavailableNativeWindow, ...]


def select_native_route_targets(binding, cursor, *, max_windows):
    """Bounded source-only search. This never evaluates a model or commits a cursor."""
    if type(binding) is not NativeRouteBinding or type(max_windows) is not int or max_windows < 1:
        raise ValueError('Exact native route and finite positive window search budget required')
    binding.__post_init__()
    unavailable = []
    for window in islice(binding.route.iter_from(cursor),max_windows):
        prepared = binding.prepare(window)
        if type(prepared) is not PreparedNativeTargets:
            raise ValueError('Exact prepared native target disposition required')
        prepared.__post_init__()
        if prepared.observed_positions:
            return NativeRouteSelection('observed',window,prepared,tuple(unavailable))
        unavailable.append(UnavailableNativeWindow(
            window.evidence_sha256,window.original_source_rows,prepared.unavailable_reason,
        ))
    status = 'unavailable-scan-limit' if len(unavailable)==max_windows else 'source-exhausted'
    return NativeRouteSelection(status,None,None,tuple(unavailable))


@dataclass(frozen=True)
class NativeMetricIdentity:
    run_id: str
    trial_id: str
    segment_id: str
    seed: int

    def __post_init__(self):
        if not all(_text(value) for value in (self.run_id,self.trial_id,self.segment_id)) or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Exact canonical run/trial/resume/seed identity required')


@dataclass(frozen=True)
class NativePrimaryRouteResult:
    status: str
    cursor_before: bytes
    cursor_after: bytes
    unavailable: tuple[UnavailableNativeWindow, ...]
    metrics: tuple[MetricRecord, ...]


def _metrics(identity,binding,selection,*,step,evidence,update=None):
    common = dict(
        run_id=identity.run_id,trial_id=identity.trial_id,segment_id=identity.segment_id,seed=identity.seed,
        step=step,x_axis='optimizer_step',task=binding.route.task_key,split='train',scope='train',
        source_sha256=binding.route.native_view_policy_sha256,evidence_sha256=evidence,
    )
    missing = sum(row.original_source_rows for row in selection.unavailable)
    records = [MetricRecord(**common,name='unavailable_native_source_rows',value=missing,
                            unit='original source rows',direction='none')]
    if update is None:
        records.append(MetricRecord(**common,name='native_objective_unavailable',value=None,
                                    unit=binding.objective_unit,direction='minimize',missing_reason=selection.status))
    else:
        records.extend((
            MetricRecord(**common,name='native_objective',value=update['loss'],unit=binding.objective_unit,direction='minimize'),
            MetricRecord(**common,name='observed_native_targets',value=selection.targets.observed_positions,
                         unit=binding.target_unit,direction='none'),
            MetricRecord(**common,name='native_source_rows_for_update',value=selection.window.original_source_rows,
                         unit='original source rows',direction='none'),
        ))
    return tuple(records)


def require_native_route_cursor_map(bindings, source_cursors):
    """Validate ALL native task cursors without reading targets or moving readers.

    A restored checkpoint can carry corrupt cursors for tasks other than the
    imminent update. Inspect every exact route policy and source view BEFORE
    any source preparation, generator draw or optimizer action. This is one
    checkpoint-boundary requirement, not proof of completed update replay.
    """
    if (type(bindings) is not dict or not bindings or type(source_cursors) is not dict
            or set(bindings) != set(source_cursors)):
        raise ValueError('Complete exact native task cursor map required')
    for key, binding in bindings.items():
        if (type(key) is not str or type(binding) is not NativeRouteBinding
                or binding.route.task_key != key or type(source_cursors[key]) is not bytes):
            raise ValueError('Exact native task binding and serialized cursor required')
        binding.__post_init__()
        route = binding.route
        if type(route) is FiniteNativeTrainRoute:
            if finite_view_policy_sha256(route._view) != route.native_view_policy_sha256:
                raise ValueError('Native train route source view snapshot changed')
            route._cursor_position(source_cursors[key])
        elif type(route) is RepeatedFiniteNativeTrainRoute:
            if finite_view_policy_sha256(route.original._view) != route.native_view_policy_sha256:
                raise ValueError('Repeated native train route source view snapshot changed')
            route._position(source_cursors[key])
        elif type(route) in (MCSACampaignTrainRoute, ReviewedTextCampaignRoute):
            route._position(source_cursors[key])
        else:
            raise ValueError('Unsupported native route cursor type')


def run_native_primary_update(engine, bindings, *, metric_identity, max_windows,
                              cpu_stager=None, prefetch_limits=None):
    """Execute one authorized native update, or return an explicit no-target result.

    Whole-owner admission is mandatory before a source window is read. An
    unavailable window advances the checkpointed cursor only if the same
    bounded search finds an observed objective and its optimizer update commits.
    Canonical records are returned for caller-owned durable reporting.
    """
    if type(engine) is not NativeMuUpdateEngine or type(engine.model) is not SourceBoundOwner:
        raise ValueError('Exact native engine and real registered source-bound owner required')
    engine.model.require_training_admission()
    if type(metric_identity) is not NativeMetricIdentity:
        raise ValueError('Exact canonical metric identity required')
    metric_identity.__post_init__()
    engine.model.require_training_admission(campaign_sha256=engine.binding_sha256,trial_id=metric_identity.trial_id)
    source_snapshot = engine.model.get_extra_state()['source_snapshot_sha256']
    if (
        type(bindings) is not dict or set(bindings) != set(engine.task_cycle)
        or any(type(binding) is not NativeRouteBinding or binding.route.task_key != key
               or binding.route.source_snapshot_sha256 != source_snapshot for key,binding in bindings.items())
    ):
        raise ValueError('Exact task cycle and same model/source snapshot route bindings required')
    require_native_route_cursor_map(bindings, engine.source_cursors)
    task = engine.next_task
    binding = bindings[task]
    before = engine.source_cursors[task]
    if cpu_stager is None:
        if prefetch_limits is not None:
            raise ValueError('Prefetch limits require an exact native CPU stager')
        selected = select_native_route_targets(binding,before,max_windows=max_windows)
    else:
        from .native_cpu_prefetch import NativeRouteCPUStager
        if (type(cpu_stager) is not NativeRouteCPUStager or type(prefetch_limits) is not dict
                or not set(prefetch_limits)<=set(bindings)
                or any(type(n) is not int or n<1 for n in prefetch_limits.values())):
            raise ValueError('Exact admitted CPU stager and per-route limits required')
        selected = cpu_stager.select(binding,before,max_windows=max_windows)
    window = selected.window
    after = before if window is None else window.cursor_after
    evidence = digest(canonical([
        engine.binding_sha256,source_snapshot,task,before.hex(),after.hex(),selected.status,
        tuple((row.evidence_sha256,row.original_source_rows,row.reason) for row in selected.unavailable),
        None if window is None else window.evidence_sha256,
    ]))
    if window is None:
        return NativePrimaryRouteResult(selected.status,before,before,selected.unavailable,
            _metrics(metric_identity,binding,selected,step=engine.steps,evidence=evidence))
    if cpu_stager is not None and prefetch_limits:
        seen = {task}
        for offset in range(1,len(engine.task_cycle)):
            upcoming = engine.task_cycle[(engine.steps+offset)%len(engine.task_cycle)]
            if upcoming in seen or upcoming not in prefetch_limits:
                continue
            seen.add(upcoming)
            cpu_stager.prefetch(bindings[upcoming],engine.source_cursors[upcoming],
                                max_windows=prefetch_limits[upcoming])
            if len(seen)==5:
                break
    if engine.profile_stage_controller is not None:
        engine.prepare_profile_stage()
    loss = selected.targets.evaluate(engine.model)
    event = engine.update(task,loss,cursor_before=before,cursor_after=after)
    return NativePrimaryRouteResult('updated',before,after,selected.unavailable,
        _metrics(metric_identity,binding,selected,step=event['step'],evidence=evidence,update=event))
