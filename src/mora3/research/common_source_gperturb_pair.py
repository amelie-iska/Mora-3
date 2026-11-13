"""Genuine sci-Plex2 two-direction source-cell conservation under native29 bindings.

The source contains paired observational reverse-task views of the *same cells*,
not independent intervention/response experiments or evidence of causal effects.
No model/optimizer, full source admission, target leakage or training release.
"""
from hashlib import sha256
from pathlib import Path
import json
import math
import re

from mora3.data.admet import canonical
from mora3.data.sciplex_admission import ObservationalSciPlexTask
from mora3.research.gperturb_campaign_routes import GPerturbNativeCampaignAdapter
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref,document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.common_source_native_replay import (
    _source_fingerprint,_first,_checked_rebound_position,_checked_two_window_transition,
)

_FAMILY='GPerturb-sciPlex2'
_SCHEMA='mora3.ObservationalSciPlexTask/v1'
_DIRECTIONS=('intervention','response')
_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
_CONTEXT=dict(study='GSM4150377',cell_line='A549',exposure_hours=24,
              baseline=None,desired_response=None)
_GROUP='study:GSE139944:GSM4150377'
_INTERVENTIONS=('Dex','Nutlin','BMS','SAHA','Vehicle')
_HEX=re.compile('[0-9a-f]{64}')


def _sha(blob):return sha256(blob).hexdigest()


def require_paired_source_semantics(response,intervention):
    """Join only fully identical original cell/context/axes, with inverse roles.

    A source-side identity match is not a global leakage-group certificate and
    never establishes causal pairing or an experimentally measured dose.
    """
    if type(response)is not dict or type(intervention)is not dict:
        raise ValueError('Two exact source-derived directional records required')
    expected={'response':(('context','intervention'),('expression',)),
              'intervention':(('context','expression'),('intervention',))}
    for d,label in ((response,'response'),(intervention,'intervention')):
        conditions,targets=expected[label]
        if (d.get('direction')!=label or d.get('condition_keys')!=conditions
                or d.get('target_keys')!=targets or d.get('source_context')!=_CONTEXT
                or d.get('original_group')!=_GROUP):
            raise ValueError('Original direction, target exclusion, study context or source group changed')
        for key in('source_shard_sha256','expression_sha256','axis_sha256'):
            value=d.get(key)
            if type(value)is not str or _HEX.fullmatch(value)is None:
                raise ValueError('Complete literal source and expression identity digests required')
        if type(d.get('source_index'))is not int or d['source_index']<0:
            raise ValueError('Exact original row source ordinal required')
        if type(d.get('cell_id'))is not str or not d['cell_id']:
            raise ValueError('Original source cell identity required')
        x=d.get('intervention')
        if (type(x)is not dict or set(x)!={'identity','source_coordinate','units','physical_concentration','is_vehicle'}
                or x['identity']not in _INTERVENTIONS or type(x['is_vehicle'])is not bool
                or x['is_vehicle']!=(x['identity']=='Vehicle')
                or x['physical_concentration']is not None
                or x['units']!='source_scaled_coordinate'
                or type(x['source_coordinate'])not in(int,float)
                or not math.isfinite(x['source_coordinate'])or x['source_coordinate']<0):
            raise ValueError('Unconverted observational treatment category and source dose required')
    same=('source_index','cell_id','source_shard_sha256','source_context',
          'expression_sha256','axis_sha256','intervention','original_group')
    if any(response[k]!=intervention[k]for k in same):
        raise ValueError('Original cell, identity, expression or intervention mismatch across reverse tasks')
    return dict(source_index=response['source_index'],cell_id_hash=_sha(response['cell_id'].encode()),
                source_context=response['source_context'],intervention=response['intervention'],
                expression_sha256=response['expression_sha256'],axis_sha256=response['axis_sha256'],
                group=response['original_group'])


def _direction_signature(adapter,window):
    """Audit original bytes, distinct source target/condition masks and provenance.

    Never call the returned learned objective or project data through the model.
    """
    direction=adapter.direction
    if window.batch.original_split!='train' or window.original_source_rows!=1:
        raise ValueError('One original training cell/task record required')
    row,=window.batch.native_rows.to_pylist()
    raw=row.get('native_payload')
    if (row.get('source_family')!=_FAMILY or row.get('task_name')!=direction
            or row.get('native_schema')!=_SCHEMA or row.get('split')!='train'
            or type(raw)is not bytes or _sha(raw)!=row.get('native_payload_sha256')):
        raise ValueError('Source-native task direction, release and payload digest changed')
    task=ObservationalSciPlexTask.from_bytes(raw)
    side=json.loads(task.task.sidecar_bytes)
    if (set(side)!={'conditions','evaluation','quarantine_reason','targets','task','training_ready','version'}
            or side['task']!=direction or side['training_ready']is not False
            or type(side['quarantine_reason'])is not str or not side['quarantine_reason']
            or side['conditions'].get('context')!=_CONTEXT):
        raise ValueError('Original observational source context or explicit unready status changed')
    metadata=task.input_metadata()
    receipt=json.loads(task.receipt_bytes)
    if (metadata.get('task')!=direction or metadata.get('split')!='train_only'
            or receipt.get('leakage_group')!=_GROUP or receipt.get('split')!='train_only'):
        raise ValueError('Source-admitted observational training view and group must remain exact')
    other='intervention' if direction=='response' else'expression'
    if (set(side['conditions'])!={'context',other}
            or set(side['targets'])!={direction if direction=='intervention' else'expression'}):
        raise ValueError('Original direction leaked target into conditioned source fields')
    expression=(side['targets']if direction=='response' else side['conditions'])['expression']
    treatment=(side['conditions']if direction=='response' else side['targets'])['intervention']
    # The original sidecar has no explicit per-array role key. Its containing
    # conditions/targets field establishes direction, without inventing labels.
    if (type(expression)is not dict or 'role'in expression
            or expression.get('axis_identity')!='source_column_position'
            or expression.get('units')!='source selected-gene count layer'
            or expression.get('status')!='observed'
            or type(expression.get('values'))is not list or len(expression['values'])!=5000
            or type(expression.get('axis_ids'))is not list or len(expression['axis_ids'])!=5000
            or any(type(g)is not str or not g for g in expression['axis_ids'])
            or any(type(x)not in(int,float)or not math.isfinite(x)or x<0 for x in expression['values'])):
        raise ValueError('Complete 5000 original source-column expression values required')
    evaluation=side['evaluation']
    if (type(evaluation.get('source_index'))is not int or evaluation['source_index']<0
            or type(evaluation.get('cell_id'))is not str or not evaluation['cell_id']):
        raise ValueError('Complete source cell position and literal identity required')
    source=dict(direction=direction,condition_keys=tuple(sorted(side['conditions'])),
        target_keys=tuple(sorted(side['targets'])),source_index=evaluation['source_index'],
        cell_id=evaluation['cell_id'],source_shard_sha256=window.batch.source_shard['sha256'],
        source_context=side['conditions']['context'],original_group=receipt['leakage_group'],
        expression_sha256=_sha(canonical(expression['values'])),axis_sha256=_sha(canonical(expression['axis_ids'])),
        intervention=treatment)
    # The exact original adapter prepares a caller-bound model objective but
    # its closure is never called. Preserve the original task direction.
    prepared=adapter.prepare(window)
    count=5000 if direction=='response' else 2
    if prepared.observed_positions!=count or prepared.unavailable_reason is not None or not callable(prepared.evaluate):
        raise ValueError('Source-native prepared target count or objective owner changed')
    if direction=='response':
        if source['condition_keys']!=('context','intervention') or source['target_keys']!=('expression',):
            raise ValueError('Response target exposure or conditioning roles changed')
    else:
        if source['condition_keys']!=('context','expression') or source['target_keys']!=('intervention',):
            raise ValueError('Intervention target exposure or conditioning roles changed')
    return source|dict(original_source_revision=row['source_revision'],
        task_id_sha256=_sha(row['task_id'].encode()),target_positions=count,
        native_payload_sha256=row['native_payload_sha256'],native_payload_bytes=len(raw),
        original_target_role=direction,sidecar_training_ready=False,
        objective_invoked=False)


def audit_original_gperturb_two_direction_pairs(root,*,directions=_DIRECTIONS,max_windows=2):
    """Read exactly two actual cells in BOTH native tasks under BOTH snapshots."""
    if type(directions)is not tuple or directions!=_DIRECTIONS or type(max_windows)is not int or max_windows!=2:
        raise ValueError('Exactly two original GPerturb task directions and two original windows required')
    root=Path(root).resolve()
    if not root.is_dir()or Path.cwd().resolve()!=root:
        raise ValueError('Use the original pinned source checkout as the current working directory')
    context=load_inactive_common_source_rebind(root)
    artifact=document(root,context.evidence)
    candidate=Ref(**artifact['inactive_candidate_snapshot']);common=context.common_snapshot
    if candidate.sha256==common.sha256 or common.sha256!=artifact['common_source_snapshot']['sha256']:
        raise ValueError('Distinct exact candidate and common29-head source commitments required')
    view=load_assets(root)[0]
    if finite_view_policy_sha256(view)!=_POLICY:
        raise ValueError('Changed original sci-Plex2 effective source view policy')
    bound={}
    for direction in _DIRECTIONS:
        key=_FAMILY+':'+direction
        evidence,=tuple(r for r in artifact['route_bindings']if r['task_key']==key)
        if (evidence['common_view_policy_sha256']!=_POLICY
                or evidence['candidate']['max_windows']!=2 or evidence['common']['max_windows']!=2):
            raise ValueError('Exact planned original GPerturb two-window source policy required')
        for label,snapshot in (('candidate',candidate),('common',common)):
            adapter=GPerturbNativeCampaignAdapter(view,snapshot.sha256,direction,1)
            binding=adapter.train_binding()
            actual=dict(task_key=binding.route.task_key,
                initial_cursor_sha256=_sha(binding.route.initial_cursor),
                adapter_sha256=executable_adapter_sha256(binding.prepare,root=Path(__file__).resolve().parents[3]),
                max_windows=2)
            if (actual!=evidence[label] or binding.route.native_view_policy_sha256!=_POLICY
                    or binding.route.source_snapshot_sha256!=snapshot.sha256):
                raise ValueError('Frozen original native GPerturb cursor/adapter policy differs: '+key+'/'+label)
            bound[(direction,label)]=(adapter,binding)

    evidence_rows={}
    for direction in _DIRECTIONS:
        for label in ('candidate','common'):
            adapter,binding=bound[(direction,label)]
            route=binding.route;iterator=route.iter_from(route.initial_cursor)
            rows=[];position=route._cursor_position(route.initial_cursor)
            try:
                for _ in range(2):
                    window=next(iterator)
                    route._view.require_native_rows(window.batch,split='train')
                    after=route._cursor_position(window.cursor_after)
                    _checked_two_window_transition(position,after)
                    rows.append((window,_direction_signature(adapter,window)))
                    position=after
            except StopIteration as exc:
                raise ValueError('Original GPerturb source exhausted before two complete windows')from exc
            finally:
                close=getattr(iterator,'close',None)
                if close is not None:close()
            restarted=_first(route.iter_from(rows[0][0].cursor_after))
            if (restarted.cursor_after!=rows[1][0].cursor_after
                    or restarted.evidence_sha256!=rows[1][0].evidence_sha256
                    or not restarted.batch.native_rows.equals(rows[1][0].batch.native_rows,check_metadata=True)
                    or _direction_signature(adapter,restarted)!=rows[1][1]):
                raise ValueError('Second GPerturb original source row changed after independent cursor restart')
            evidence_rows[(direction,label)]=tuple(rows)

    signatures={};reported=[]
    for direction in _DIRECTIONS:
        left=evidence_rows[(direction,'candidate')];right=evidence_rows[(direction,'common')]
        for n,((a,s),(b,t)) in enumerate(zip(left,right,strict=True)):
            if (a.batch.source_shard!=b.batch.source_shard
                    or a.batch.source_row_ordinals!=b.batch.source_row_ordinals
                    or not a.batch.native_rows.equals(b.batch.native_rows,check_metadata=True)
                    or s!=t or _source_fingerprint(a)!=_source_fingerprint(b)):
                raise ValueError('Original GPerturb cell, task, target or masks differ across source snapshots')
            pos_a=bound[(direction,'candidate')][1].route._cursor_position(a.cursor_after)
            pos_b=bound[(direction,'common')][1].route._cursor_position(b.cursor_after)
            _checked_rebound_position(pos_a,pos_b)
            signatures[(direction,n)]=dict(source=s,ordinal=a.batch.source_row_ordinals[0],
                native_source_fingerprint=_source_fingerprint(a)[0],
                candidate_cursor_sha256=_sha(a.cursor_after),common_cursor_sha256=_sha(b.cursor_after),
                source_position=pos_a)
            if a.cursor_after==b.cursor_after:
                raise ValueError('Distinct original source-snapshot cursor identities required')
            reported.append((direction,n,s['target_positions']))
    result=[]
    for n in range(2):
        iproof=signatures[('intervention',n)];rproof=signatures[('response',n)]
        intervention,iordinal=iproof['source'],iproof['ordinal']
        response,rordinal=rproof['source'],rproof['ordinal']
        pair=require_paired_source_semantics(response,intervention)
        if (iordinal==rordinal or response['original_source_revision']!=intervention['original_source_revision']
                or response['source_shard_sha256']!=intervention['source_shard_sha256']
                or response['native_payload_sha256']==intervention['native_payload_sha256']):
            raise ValueError('Reverse original task directions must be distinct real source records')
        result.append(dict(source_index=pair['source_index'],cell_id_hash=pair['cell_id_hash'],
            original_source_revision=response['original_source_revision'],source_shard_sha256=response['source_shard_sha256'],
            source_context=pair['source_context'],intervention=pair['intervention'],
            intervention_ordinal=iordinal,response_ordinal=rordinal,
            expression_axis_length=5000,response_expression_sha256=response['expression_sha256'],
            intervention_condition_expression_sha256=intervention['expression_sha256'],
            expression_axis_sha256=pair['axis_sha256'],
            source_group=pair['group'],condition_target_roles=dict(
                response=dict(conditions=list(response['condition_keys']),targets=list(response['target_keys'])),
                intervention=dict(conditions=list(intervention['condition_keys']),targets=list(intervention['target_keys']))),
            response_native_payload_sha256=response['native_payload_sha256'],
            intervention_native_payload_sha256=intervention['native_payload_sha256'],
            response_native_source_fingerprint=rproof['native_source_fingerprint'],
            intervention_native_source_fingerprint=iproof['native_source_fingerprint'],
            candidate_response_cursor_sha256=rproof['candidate_cursor_sha256'],
            common_response_cursor_sha256=rproof['common_cursor_sha256'],
            candidate_intervention_cursor_sha256=iproof['candidate_cursor_sha256'],
            common_intervention_cursor_sha256=iproof['common_cursor_sha256'],
            response_next_position=rproof['source_position'],
            intervention_next_position=iproof['source_position'],
            target_sidecar_training_ready=False,observational_train_only=True,
            candidate_common_cursor_positions_equal=True))
    if result[0]['source_index']==result[1]['source_index']:
        raise ValueError('Two distinct original source-cell indices required')
    directions_total={d:sum(num for route,_,num in reported if route==d)for d in _DIRECTIONS}
    return dict(schema='Mora3.GPerturbSourcePairedRoutesTwoWindows/v1',source_family=_FAMILY,
        original_view_policy_sha256=_POLICY,candidate_source_snapshot_sha256=candidate.sha256,
        common_source_snapshot_sha256=common.sha256,source_cells=2,original_route_rows=4,
        direction_targets=directions_total,original_source_targets=sum(directions_total.values()),
        paired_rows=tuple(result),candidate_common_source_equal=True,
        independent_second_window_restarts=True,shared_original_source_cells=True,
        source_group_heldout_accepted=False,physical_concentration_resolved=False,
        paired_causal_treatment_evidence=False,source_train_only_not_full_campaign=True,
        training_admission=False,optimizer_updates=0,actual_campaign_runs=0,
        complete62_route_replay=False,fresh_same_owner_validation_reference=False,
        conclusion='observational paired source-task views, no calibrated dose, no causal effect or globally accepted split')
