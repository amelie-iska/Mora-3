"""Inactive common-source route/evaluator binding for the native N29 owner."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .primary_source_factory_preflight import Ref, document, preflight_native29_source_inputs, require

_STAGE='outputs/primary-campaign-common-source-rebind-20261108-v1/'
_ARTIFACT=Ref(_STAGE+'common-source-rebind.json','9da9041ab90959d15a7395a21c3cd90d8a1b55648898b300559b37feb99d6c0b')
_REFUSALS=(
    'storage-isolated exact live route/evaluator construction and 5,000-window replay',
    'new matched validation reference using this exact 29-head owner and source identity',
    'completed independent split, owner, capacity and worker admission',
)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CommonTrainBinding:
    task_key: str
    adapter_sha256: str
    initial_cursor_sha256: str
    max_windows: int
    scheduled_updates: int


@dataclass(frozen=True)
class CommonValidationBinding:
    task_key: str
    metadata_adapter_sha256: str
    coordinate_commitment_sha256: str
    original_coordinates: int
    observed_targets: int


@dataclass(frozen=True)
class InactiveCommonSourceRebind:
    evidence: Ref
    common_snapshot: Ref
    native_family_sha256: str
    task_cycle_sha256: str
    train: tuple[CommonTrainBinding, ...]
    validation: tuple[CommonValidationBinding, ...]
    validation_view_metadata_sha256: str

    def require_worker_ready(self):
        raise ValueError('Common source rebind is metadata only. Remaining gates: '+', '.join(_REFUSALS))


def load_inactive_common_source_rebind(root):
    """Authenticate the exact repository-bound result and immutable parents."""
    root=Path(root).resolve()
    plan=preflight_native29_source_inputs(root)
    value=document(root,_ARTIFACT)
    require(value['schema']=='Mora3.InactiveCommonCampaignSourceRebind/v1'
            and Ref(**value['common_source_snapshot'])==plan.owner_snapshot
            and Ref(**value['original_route_snapshot'])==plan.original_route_snapshot
            and Ref(**value['inactive_candidate_snapshot'])==plan.candidate_route_snapshot
            and value['n29_family_sha256']==plan.native_family_sha256
            and value['prior_diagnostic_family_sha256']!=plan.native_family_sha256
            and value['common_reader_replay_completed'] is value['common_matched_reference_measured']
            is value['global_split_accepted'] is value['training_authorized'] is False,
            'Exact inactive common-source lineage and incomplete admission required')
    route_plan=document(root,Ref(**value['original_plan']))
    old=document(root,Ref(**value['candidate_assembly']))
    trace=document(root,Ref(**value['original_replay']))
    prior={row['task_key']:row for row in route_plan['train_routes']}
    previous={row['task_key']:row for row in old['route_bindings']}
    require(value['original_ordered_task_cycle_sha256']==_digest(route_plan['task_cycle'])
            ==old['ordered_task_cycle_sha256'] and len(route_plan['task_cycle'])==value['original_schedule_updates']==5000
            and trace['source_windows_consumed']==value['original_consumed_windows']==5940
            and value['candidate_validation_view_metadata_sha256']==old['candidate_validation_view_metadata_sha256']
            and value['common_validation_view_metadata_sha256']!=old['candidate_validation_view_metadata_sha256'],
            'Exact preserved route schedule and separate common evaluator policy required')
    rows=value['route_bindings']
    keys=tuple(row['task_key'] for row in rows)
    require(keys==tuple(sorted(prior))==tuple(sorted(previous)) and len(keys)==62,
            'Exact ordered common training route inventory required')
    train=[]
    for row in rows:
        key=row['task_key']
        common=row['common']
        require(row['original']==prior[key] and row['candidate']==previous[key]['candidate']['binding']
                and common['task_key']==key and common['max_windows']==prior[key]['max_windows']
                and row['scheduled_updates']==route_plan['required_updates'][key]
                and all(len({row['original'][field],row['candidate'][field],common[field]})==3
                        for field in ('adapter_sha256','initial_cursor_sha256')),
                'Exact new route bindings cannot reuse parent cursors or adapter identities')
        train.append(CommonTrainBinding(**common,scheduled_updates=row['scheduled_updates']))
    projection,=(row['common_projection'] for row in rows if row['common_projection'] is not None)
    policy=document(root,Ref(**projection))
    require(policy['source_snapshot_sha256']==plan.owner_snapshot.sha256
            and policy['original_roles_preserved'] is True and policy['training_authorized'] is False,
            'Exact inactive common M-CSA train policy required')
    validation=[]
    old_validation=old['metadata_only_evaluators']
    require(set(value['validation_bindings'])==set(old_validation)==set(route_plan['validation_evaluator_hashes'])
            and len(old_validation)==5,'Exact matched validation metadata scope required')
    for key,item in sorted(value['validation_bindings'].items()):
        previous=old_validation[key]
        change='common_metadata_evaluator_sha256'
        old_change='candidate_metadata_evaluator_sha256'
        require(item['role']=='validation' and item['scope']=='metadata binding only'
                and item['parent_v6_evaluator_sha256']==route_plan['validation_evaluator_hashes'][key]
                and len({item['parent_v6_evaluator_sha256'],item[change],previous[old_change]})==3
                and {k:v for k,v in item.items() if k!=change}=={k:v for k,v in previous.items() if k!=old_change},
                'Exact protected enrollment commitment must survive evaluator rebinding')
        validation.append(CommonValidationBinding(key,item[change],item['coordinate_commitment_sha256'],
                                                   item['original_coordinates'],item['observed_targets']))
    require(sum(x.original_coordinates for x in validation)==2324 and sum(x.observed_targets for x in validation)==5560367,
            'Exact preserved protected validation denominators required')
    return InactiveCommonSourceRebind(_ARTIFACT,plan.owner_snapshot,plan.native_family_sha256,
        value['original_ordered_task_cycle_sha256'],tuple(train),tuple(validation),value['common_validation_view_metadata_sha256'])


def check_common_runtime_bindings(root, *, context, bindings, validation_view, evaluators):
    """Check constructed bindings against the inactive metadata without opening windows."""
    from .matched_validation import matched_validation_source_policy_sha256
    from .mup_campaign import executable_adapter_sha256
    from .primary_native_routes import NativeRouteBinding

    require(type(context) is InactiveCommonSourceRebind and context==load_inactive_common_source_rebind(root),
            'Exact authenticated common source metadata required')
    require(type(bindings) is dict and set(bindings)=={row.task_key for row in context.train}
            and type(evaluators) is dict and set(evaluators)=={row.task_key for row in context.validation},
            'Complete train and matched validation implementations required')
    for planned in context.train:
        row=bindings[planned.task_key]
        require(type(row) is NativeRouteBinding and row.route.task_key==planned.task_key
                and row.route.source_snapshot_sha256==context.common_snapshot.sha256
                and sha256(row.route.initial_cursor).hexdigest()==planned.initial_cursor_sha256
                and executable_adapter_sha256(row.prepare,root=root)==planned.adapter_sha256,
                'Live route, cursor or adapter differs from common source metadata')
    require(matched_validation_source_policy_sha256(validation_view)==context.validation_view_metadata_sha256
            and all(executable_adapter_sha256(evaluators[row.task_key],root=root)==row.metadata_adapter_sha256
                    for row in context.validation),'Live matched validation view or evaluators differ')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    result=load_inactive_common_source_rebind(parser.parse_args(argv).root)
    print(json.dumps({'common_source_snapshot_sha256':result.common_snapshot.sha256,
        'native_family_sha256':result.native_family_sha256,
        'routes':len(result.train),'matched_metadata_evaluators':len(result.validation),
        'original_coordinates':sum(x.original_coordinates for x in result.validation),
        'observed_targets':sum(x.observed_targets for x in result.validation),
        'training_authorized':False,'remaining_gates':_REFUSALS},sort_keys=True))


if __name__=='__main__':
    main()
