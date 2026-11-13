"""Full original Rfam train-route iterator replay and bounded native preparation.

Checks both exact staged N29 candidate/common source binding identities against
original rows. Only first/last window of each shard is prepared; ALL original
train windows are consumed and independently checked. No model/evaluator call,
optimizer, heldout source, admission or counted sweep is executed.
"""
from hashlib import sha256
import json
from pathlib import Path

from mora3.data.admet import canonical
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.research.rfam_goa_registry import FILTER
from mora3.research.rfam_campaign_routes import RfamCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.rfam_original_dual_direction_census import _strict

CENSUS = ('outputs/common-source-rfam-full-dual-20261111/actual-original-rfam-paired-census.json',
          '460e278369f78a285520820211981806084028748272062f66d3d538a995b88d')
REBOUND = ('outputs/primary-campaign-common-source-rebind-20261108-v1/common-source-rebind.json',
           '9da9041ab90959d15a7395a21c3cd90d8a1b55648898b300559b37feb99d6c0b')


def _original_positions(row, direction):
    obj=_strict(row['native_payload'])
    if obj.get('task')!='Rfam_source_family_'+direction+'_prior':
        raise ValueError('Native original task direction mismatch')
    if direction=='prediction':
        if set(obj['conditions'])!={'source_native_sequence'}:
            raise ValueError('Source Rfam prediction target entered its conditions')
        return len(obj['targets']['source_Rfam_family_ids'])
    if set(obj['conditions'])!={'desired_source_Rfam_family_ids','desired_source_clan_ids'}:
        raise ValueError('Native Rfam generation target entered its conditions')
    return len(obj['targets']['source_native_sequence'])+1


def _stream(binding, direction):
    route=binding.route
    if route._batch_size!=1 or len(route._paths)!=72:
        raise ValueError('Exact frozen single-original-record Rfam route required')
    population=sha256(); parents=sha256();samples=sha256()
    rows=0;positions=0;prepared=0;previous_index=-1;previous_ordinal=-1
    first=None;last=None;file_count=0;last_cursor=None

    def check_sample(window):
        nonlocal prepared
        result=binding.prepare(window)
        row,=window.batch.native_rows.to_pylist()
        expected=_original_positions(row,direction)
        if (result.observed_positions!=expected or result.unavailable_reason is not None
            or not callable(result.evaluate)):
            raise ValueError('Native prepared Rfam target count, availability or objective missing')
        samples.update(canonical((window.batch.source_shard['path'],window.batch.source_row_ordinals[0],
                                  row['task_id'],expected))+b'\n')
        prepared+=1

    for window in route.iter_from(route.initial_cursor):
        if window.original_source_rows!=1 or len(window.batch.source_row_ordinals)!=1:
            raise ValueError('Native Rfam original window cardinality changed')
        path=window.batch.source_shard['path']
        try:index=route._paths.index(path)
        except ValueError as exc:raise ValueError('Unknown native original Rfam shard') from exc
        ordinal,=window.batch.source_row_ordinals
        if index<previous_index or (index==previous_index and ordinal<=previous_ordinal):
            raise ValueError('Original Rfam cursor not strictly increasing')
        if index!=previous_index:
            if previous_index!=-1:
                check_sample(first)
                if last.batch.source_row_ordinals!=first.batch.source_row_ordinals:check_sample(last)
            file_count+=1;first=window
        row,=window.batch.native_rows.to_pylist()
        if (row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False
            or row['split']!='train' or row['source_family']!='Rfam15.1-source-family-prior'
            or sha256(row['native_payload']).hexdigest()!=row['native_payload_sha256']):
            raise ValueError('Original Rfam source role/adapter disposition or bytes changed')
        n=_original_positions(row,direction)
        if n<1:raise ValueError('Original Rfam native target observations missing')
        population.update(canonical((path,ordinal,row['task_id'],row['record_id'],
                                     row['native_payload_sha256'],n))+b'\n')
        parents.update(canonical((path,row['record_id']))+b'\n')
        rows+=1;positions+=n
        previous_index=index;previous_ordinal=ordinal;last=window;last_cursor=window.cursor_after
    if first is None:raise ValueError('No original native Rfam train source rows')
    check_sample(first)
    if last.batch.source_row_ordinals!=first.batch.source_row_ordinals:check_sample(last)
    if file_count!=72 or rows!=18423:
        raise ValueError('Original native Rfam source window coverage incomplete')
    # Exhaust the same native reader from its last genuine persisted cursor.
    if tuple(route.iter_from(last_cursor)):
        raise ValueError('Last Rfam cursor repeats a completed source window')
    return dict(rows=rows,observed_positions=positions,original_shards=file_count,
        actual_prepared_boundary_windows=prepared,population_sha256=population.hexdigest(),
        original_parent_order_sha256=parents.hexdigest(),prepared_boundaries_sha256=samples.hexdigest(),
        final_real_window_cursor_sha256=sha256(last_cursor).hexdigest(),
        final_real_window_cursor_position=list(route._cursor_position(last_cursor)),
        terminal_restart_exhausted=True)


def replay_complete_rfam_native_routes(root, *, worktree=None):
    root=Path(root).resolve()
    worktree=Path(worktree).resolve() if worktree is not None else Path(__file__).resolve().parents[3]
    if root!=Path.cwd().resolve():
        raise ValueError('Actual active immutable Rfam source root required')
    raw=(worktree/CENSUS[0]).read_bytes()
    if sha256(raw).hexdigest()!=CENSUS[1]:
        raise ValueError('Prior original whole Rfam census readback changed')
    census=json.loads(raw)
    if (census['parent_observations']!=18423 or census['source_rows']!=36846
        or census['source_snapshot_sha256']!='79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'):
        raise ValueError('Original paired-parent source census differs')
    context=load_inactive_common_source_rebind(root)
    metadata=document(root,Ref(*REBOUND))
    candidate=metadata['inactive_candidate_snapshot']['sha256']
    common=context.common_snapshot.sha256
    view=load_inverse_literal_filtered_native_view(FILTER[0],FILTER[1],root=root)
    if finite_view_policy_sha256(view)!='11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046':
        raise ValueError('Actual original Rfam source policy differs')
    planned={x['task_key']:x for x in metadata['route_bindings']}
    output={};snapshots={'candidate':candidate,'common':common}
    for direction in ('prediction','generation'):
        key='Rfam_source_family_'+direction+'_prior'
        one={}
        for name,snapshot in snapshots.items():
            adapter=RfamCampaignAdapter(view,direction,snapshot,1)
            binding=adapter.train_binding()
            spec=planned[key][name]
            if (binding.route.task_key!=key or spec['max_windows']!=2
                or sha256(binding.route.initial_cursor).hexdigest()!=spec['initial_cursor_sha256']
                or executable_adapter_sha256(binding.prepare,root=worktree)!=spec['adapter_sha256']):
                raise ValueError('Exact original candidate/common N29 Rfam source preparation identity differs')
            one[name]=_stream(binding,direction)
        if any(one['candidate'][k]!=one['common'][k] for k in
               ('rows','observed_positions','original_shards','actual_prepared_boundary_windows',
                'population_sha256','original_parent_order_sha256','prepared_boundaries_sha256')):
            raise ValueError('Candidate and common original native Rfam source preparation differs')
        output[direction]=one
    if output['prediction']['candidate']['original_parent_order_sha256']!=output['generation']['candidate']['original_parent_order_sha256']:
        raise ValueError('Rfam two task directions do not preserve original parent order')
    return dict(schema='Mora3.OriginalRfamNativeFullSourceReplay/v1',
        source_snapshot_sha256=common,candidate_source_snapshot_sha256=candidate,
        source_view_policy_sha256=finite_view_policy_sha256(view),
        source_previous_census_sha256=CENSUS[1],
        rows={k:v['common']['rows'] for k,v in output.items()},
        original_parent_observations=18423,native_preparation_success=True,
        candidate_common_result_match=True,
        restart_checks=4,actual_native_source_windows_consumed=4*18423,
        actual_boundary_preparations=sum(v[n]['actual_prepared_boundary_windows']
            for v in output.values() for n in ('candidate','common')),
        original_native_source_routes=output,
        complete_original_native_route_iteration=True,
        all_original_targets_prepared=False,
        full_model_training_admission=False,optimizer_updates=0,
        actual_proxy_trials=0,actual_target_runs=0,
        heldout_targets_opened=False,
        note='All original native source windows, sampled each shard first/last actual model input preparation; no model execution or training')


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:raise SystemExit('Usage: python -m mora3.research.rfam_native_complete_replay ACTIVE_ROOT NEW_RESULT')
    dest=Path(sys.argv[2]);result=replay_complete_rfam_native_routes(Path(sys.argv[1]))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('rows','actual_native_source_windows_consumed','actual_boundary_preparations','candidate_common_result_match')},sort_keys=True))
