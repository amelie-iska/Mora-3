"""Exact original-window capacities and explicit integer-weight schedules.

Window availability is not proof of nonempty observed targets or optimizer work.
No passes, weights or source roles are added or inferred.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import full9,decode_disposition,read_exact
from mora3.research.primary_native_routes import FiniteNativeTrainRoute,NativeRouteBinding,RepeatedFiniteNativeTrainRoute
from mora3.research.mcsa_campaign_route import MCSACampaignTrainRoute
from mora3.research.source_text_campaign import ReviewedTextCampaignRoute,SPECS
from mora3.research.mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.primary_campaign_assembly import _read,_reference

@dataclass(frozen=True)
class Capacity:
    task_key:str
    original_windows:int
    per_update_scan_limit:int
    source_window_upper_bound:int
    initial_cursor_sha256:str
    adapter_sha256:str
    evidence_sha256:str

class ScheduleInfeasible(ValueError):
    pass

def _grouped_native_windows(route, descriptor, *, root, max_metadata_bytes):
    """Count actual nonempty projected native windows without decoding payloads."""
    from pyarrow.parquet import ParquetFile

    shard = descriptor['source_shard']
    path = shard['path']
    source = root / path
    fingerprint = tuple(descriptor['source_full9'])
    if full9(source) != fingerprint:
        raise ValueError('Pinned original source fingerprint changed')
    read_exact(descriptor['original_group_sidecar'], root=root)
    codes = decode_disposition(descriptor, root=root)
    excluded = {ordinal for p, ordinal in route._view._ordinal_exclusions if p == path}
    names = ('source_family','task_name','native_schema','source_revision','split','task_id')
    offset = used = windows = retained = 0
    for batch in ParquetFile(source).iter_batches(batch_size=route._batch_size, columns=names):
        used += batch.nbytes
        if used > max_metadata_bytes:
            raise ValueError('Grouped source identity metadata budget exceeded')
        stop = offset + batch.num_rows
        if stop > len(codes):
            raise ValueError('Original metadata extends beyond the pinned disposition')
        selected = tuple(i for i in range(offset,stop) if codes[i] == 0 and i not in excluded)
        if selected:
            fields = tuple(batch[name].to_pylist() for name in names)
            matching = 0
            for i in selected:
                *identity, role, task_id = (field[i-offset] for field in fields)
                if role != 'train' or task_id is None or identity[0] is None:
                    raise ValueError('Selected original task identity or train role differs')
                if tuple(identity) in route._identity:
                    matching += 1
            retained += matching
            windows += bool(matching)
        offset = stop
    if offset != len(codes) or full9(source) != fingerprint:
        raise ValueError('Original source identity changed during grouped metadata read')
    return windows,retained,used


def exact_capacities(root,*,bindings,plans,source_snapshot,max_identity_metadata_bytes):
    root=Path(root).resolve();sr=_reference(source_snapshot);raw=_read(root,sr)
    if type(max_identity_metadata_bytes) is not int or max_identity_metadata_bytes<1:raise ValueError('Explicit metadata byte budget required')
    if type(bindings) is not tuple or type(plans) is not tuple or tuple(k for k,_ in bindings)!=tuple(p.task_key for p in plans) or len(set(k for k,_ in bindings))!=len(bindings):raise ValueError('Exact paired unique ordered bindings/plans required')
    cache={};used=0;result=[]
    def excluded_identity(view,path,ordinal):
        nonlocal used
        import pyarrow.parquet as pq
        key=(path,ordinal)
        if key in cache:return cache[key]
        source=root/path;before=full9(source);parquet=pq.ParquetFile(source);offset=0
        for i in range(parquet.num_row_groups):
            count=parquet.metadata.row_group(i).num_rows
            if offset<=ordinal<offset+count:
                fields=('source_family','task_name','native_schema','source_revision')
                table=parquet.read_row_group(i,columns=list(fields),use_threads=False)
                used+=table.nbytes
                if used>max_identity_metadata_bytes:raise ValueError('Excluded-row identity metadata budget exceeded')
                row=table.slice(ordinal-offset,1).to_pylist()[0];identity=tuple(row[k] for k in fields)
                if full9(source)!=before:raise ValueError('Source changed during identity-only read')
                cache[key]=identity;return identity
            offset+=count
        raise ValueError('Excluded ordinal outside pinned source')
    for (key,binding),plan in zip(bindings,plans,strict=True):
        if type(binding) is not NativeRouteBinding or type(plan) is not NativeTrainRoutePlan:raise ValueError('Exact typed binding/plan required')
        route=binding.route;plan.__post_init__();binding.__post_init__()
        if route.source_snapshot_sha256!=sr.sha256 or route.task_key!=key or digest(route.initial_cursor)!=plan.initial_cursor_sha256 or executable_adapter_sha256(binding.prepare,root=root)!=plan.adapter_sha256:raise ValueError('Exact snapshot/cursor/executable plan changed')
        evidence=[]
        if type(route) is MCSACampaignTrainRoute:
            passes = route._policy.max_passes
            limit = route.maximum_updates
            capacity = limit // passes
            evidence = route.binding()
        elif type(route) is ReviewedTextCampaignRoute:
            state=route.executable_binding();route.budget.__post_init__()
            task,train_rows,validation_rows,_=SPECS[route.family]
            if (state['task']!=task or state['train_rows']!=train_rows
                or len(route._train)!=train_rows or state['validation_rows']!=validation_rows
                or len(route._validation)!=validation_rows):
                raise ValueError('Exact reviewed text role denominator changed')
            coordinates=[]
            for issued in route._train:
                route.view.require_row(issued,role='train')
                row=issued.native_row
                if row.source_family!=route.family or row.evaluation_only or row.condition_inputs()['task']!=task:
                    raise ValueError('Exact reviewed native text train family/task required')
                used+=len(row.source_coordinate_json)
                if used>max_identity_metadata_bytes:raise ValueError('Text source coordinate metadata budget exceeded')
                coordinates.append(digest(row.source_coordinate_json))
            if len(set(coordinates))!=train_rows:raise ValueError('Duplicate text source coordinates cannot enlarge capacity')
            passes=route.budget.max_passes
            capacity=(train_rows+route.budget.batch_size-1)//route.budget.batch_size
            limit=capacity*passes
            if limit!=route.maximum_windows:raise ValueError('Declared finite text pass capacity changed')
            evidence=dict(route=state,train_coordinate_order_sha256=digest(canonical(coordinates)),
                original_windows=capacity,declared_passes=passes,
                incomplete_last_batch_preserved=True)
        elif type(route) in (FiniteNativeTrainRoute, RepeatedFiniteNativeTrainRoute):
            passes = route.max_passes if type(route) is RepeatedFiniteNativeTrainRoute else 1
            route = route.original if passes > 1 else route
            capacity=0
            for path in route._paths:
                desc=json.loads(route._view._sealed_by_path[path])
                if route._batch_size > 1:
                    windows,retained,read_bytes = _grouped_native_windows(
                        route,desc,root=root,max_metadata_bytes=max_identity_metadata_bytes-used
                    )
                    used += read_bytes
                    capacity += windows
                    evidence.append((path,desc['source_shard']['sha256'],retained,windows,route._batch_size))
                    continue
                if full9(root/path)!=tuple(desc['source_full9']):raise ValueError('Pinned original source fingerprint changed')
                rows=sum(r['rows'] for r in desc['source_task_schema_counts'] if (r['source_family'],r['task_name'],r['native_schema'],r['source_revision']) in route._identity and r['split']=='train' and r['eligible_view_disposition']==0)
                exclusions=tuple(o for p,o in route._view._ordinal_exclusions if p==path)
                removed=0
                if exclusions:
                    codes=decode_disposition(desc,root=root)
                    eligible={tuple(r[f] for f in ('source_family','task_name','native_schema','source_revision')) for r in desc['source_task_schema_counts'] if r['split']=='train' and r['eligible_view_disposition']==0}
                    for ordinal in exclusions:
                        if codes[ordinal]!=0:continue
                        if eligible<=route._identity or excluded_identity(route._view,path,ordinal) in route._identity:removed+=1
                if rows<removed:raise ValueError('Exclusions exceed exact declared original rows')
                capacity+=rows-removed
                evidence.append((path,desc['source_shard']['sha256'],rows,removed))
            evidence = dict(passes=passes, original_shards=evidence)
            limit = capacity * passes
        else:raise ValueError('Unsupported route requires separately reviewed exact pass-capacity evidence')
        result.append(Capacity(key,capacity,plan.max_windows,limit,plan.initial_cursor_sha256,plan.adapter_sha256,digest(canonical(evidence))))
    if _read(root,sr)!=raw:raise ValueError('Pinned source snapshot changed')
    return tuple(result)

def build_exact_weight_schedule(capacities,*,weights,total_updates):
    """Preserve exact caller integer weight ratios; refuse rounding or redistribution.

    Smooth weighted round-robin with lexical ties is deterministic. Zero-weight
    omission is unsupported because every declared campaign key must be present.
    """
    if type(capacities) is not tuple or not capacities or any(type(c) is not Capacity for c in capacities):raise ValueError('Exact capacity tuple required')
    keys=tuple(c.task_key for c in capacities)
    if keys!=tuple(sorted(set(keys))) or type(weights) is not dict or set(weights)!=set(keys) or any(type(w) is not int or w<1 for w in weights.values()) or type(total_updates) is not int or total_updates<1:raise ValueError('Explicit exact positive integer weight per route and total required')
    for c in capacities:
        if (any(type(x) is not int for x in (c.original_windows,c.per_update_scan_limit,c.source_window_upper_bound))
                or not 0<=c.original_windows<=c.source_window_upper_bound or c.per_update_scan_limit<1):
            raise ValueError('Exact nonnegative source windows and positive per-update scan bound required')
    if sum(c.source_window_upper_bound for c in capacities)<total_updates:raise ScheduleInfeasible('Total pinned source-window capacity is below requested exact length')
    mass=sum(weights.values())
    if any(total_updates*weights[k]%mass for k in keys):raise ScheduleInfeasible('Requested exact integer weight ratios do not yield integral route counts; explicit revised weights required')
    counts={k:total_updates*weights[k]//mass for k in keys}
    if any(counts[c.task_key]>c.source_window_upper_bound for c in capacities):raise ScheduleInfeasible('Requested exact weights exceed a route capacity; no redistribution or repetition authorized')
    credit=dict.fromkeys(keys,0);left=dict(counts);schedule=[]
    for _ in range(total_updates):
        for k in keys:credit[k]+=counts[k]
        key=min((k for k in keys if left[k]),key=lambda k:(-credit[k],k))
        credit[key]-=total_updates;left[key]-=1;schedule.append(key)
    if any(left.values()):raise AssertionError('Exact deterministic allocation incomplete')
    return tuple(schedule)
