"""Repository-reviewed primary campaign launch boundary.

The diagnostic v6 context is independent of production release authority.
The typed execution adapter uses the repository-owned native source factory
only after exact release and common source-snapshot checks.
"""
import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from mora3.model.full_owner_training_admission import (
    _LAUNCH_RELEASE_SCHEMA, _metadata, load_reviewed_primary_release,
)
from .primary_campaign_assembly import ArtifactReference, PrimaryCampaignInputs, _reference
from .primary_campaign_context import (
    build_primary_campaign_context, check_primary_request_metadata,
)
from .primary_native_routes import NativeMetricIdentity
from .runtime_capacity_gate import RuntimeCapacityRequest

_SCHEMA = 'Mora3.PrimaryCampaignWorkerLaunch/v1'


@dataclass(frozen=True)
class _Launch:
    release: ArtifactReference
    request: ArtifactReference
    inputs: PrimaryCampaignInputs
    route_request: ArtifactReference
    capacity: RuntimeCapacityRequest
    reviewed_capacity_sha256: str
    controller_directory: Path
    campaign_directory: Path
    journal_path: Path
    metric_identity: NativeMetricIdentity
    device: str
    geometry_identifier_axis: tuple[str, ...]
    color_identifier_axis: tuple[str, ...]
    profile_group_manifest: ArtifactReference
    profile_private_seed: int


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _output_path(root, value):
    _require(type(value) is str and value and str(Path(value))==value
             and Path(value).is_relative_to('outputs') and len(Path(value).parts)>1
             and '..' not in Path(value).parts
             and not {'data-intake','data-sources','shards','validation','heldout','held-out','test'}.intersection(Path(value).parts),
             'Exact relative campaign output location required')
    path=root/value
    _require(not any(p.is_symlink() for p in (path,*path.parents)),
             'Campaign paths must be nonsymlink')
    return path


def _identifier_axis(value):
    _require(type(value) is list and value and all(type(v) is str and v and v.strip()==v for v in value)
             and len(set(value))==len(value),'Exact finite identifier axis required')
    return tuple(value)


def preflight_primary_worker_launch(root, *, release_path, release_sha256):
    """Read only pinned review metadata. No source payload or run state."""
    root=Path(root).resolve()
    release_ref,release,inputs,window=load_reviewed_primary_release(
        root,path=release_path,expected_sha256=release_sha256)
    _require(release['schema']==_LAUNCH_RELEASE_SCHEMA,
             'A repository-reviewed release must bind the worker launch request')
    request=_reference(release['launch_request'])
    doc=_metadata(root,request)
    _require(set(doc)=={'schema','route_request','capacity','controller_directory','campaign_directory',
                        'journal_path','run_id','segment_id','device','geometry_identifier_axis',
                        'color_identifier_axis','profile_group_manifest','profile_private_seed',
                        'execution_root','implementation_sha256'}
             and doc['schema']==_SCHEMA and doc['execution_root']==str(root),
             'Exact versioned worker launch fields and execution root required')
    from mora3.model import primary_source_factory, source_assembly_inputs
    from . import primary_campaign_construction62, primary_source_factory_preflight
    modules={'entrypoint':__file__,'route_construction':primary_campaign_construction62.__file__,
             'source_assembly_inputs':source_assembly_inputs.__file__,
             'source_factory':primary_source_factory.__file__,
             'source_factory_preflight':primary_source_factory_preflight.__file__}
    _require(doc['implementation_sha256']=={key:sha256(Path(path).read_bytes()).hexdigest()
                                           for key,path in modules.items()},
             'Live launch, route construction or source assembly implementation differs from review')
    routes=_reference(doc['route_request'])
    route_doc=_metadata(root,routes)
    _require(route_doc.get('schema')==primary_campaign_construction62._SCHEMA
             and route_doc.get('primary_request')==asdict(inputs.request),
             'Route construction must bind the exact reviewed primary request')
    declaration=_metadata(root,inputs.request)
    budget=doc['capacity']
    _require(type(budget) is dict and set(budget)=={'max_identity_metadata_bytes','max_replay_windows'},
             'Exact reviewed window capacity budgets required')
    capacity=RuntimeCapacityRequest(asdict(inputs.source_snapshot),declaration['campaign'],**budget)
    controller=_output_path(root,doc['controller_directory'])
    campaign=_output_path(root,doc['campaign_directory'])
    journal=_output_path(root,doc['journal_path'])
    _require(controller.is_dir() and campaign.is_dir() and journal.parent.is_dir()
             and not journal.exists() and not controller.is_relative_to(campaign)
             and not campaign.is_relative_to(controller) and not journal.is_relative_to(campaign)
             and not journal.is_relative_to(controller),'Exact independent existing run paths and unused journal required')
    device=doc['device']
    _require(type(device) is str and re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]{0,2})',device),
             'Explicit CPU or indexed CUDA device required')
    seed=doc['profile_private_seed']
    _require(type(seed) is int and 0<=seed<2**63,'Exact private profile seed required')
    return _Launch(
        release_ref,request,inputs,routes,capacity,window,controller,campaign,journal,
        NativeMetricIdentity(doc['run_id'],inputs.trial.trial_id,doc['segment_id'],inputs.trial.seed),
        device,_identifier_axis(doc['geometry_identifier_axis']),_identifier_axis(doc['color_identifier_axis']),
        _reference(doc['profile_group_manifest']),seed,
    )


def execute_primary_campaign(root, *, release_path, release_sha256, sources):
    """Execute the single existing worker transaction from exact typed owner inputs."""
    root=Path(root).resolve()
    launch=preflight_primary_worker_launch(root,release_path=release_path,release_sha256=release_sha256)
    import torch
    from mora3.model.full_owner import SourceAssemblyInputs, construct_primary_owner
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
    from mora3.model.mu_optimizer import MuAdamW
    from mora3.model.profile_reward_stage import ProfileRewardStageController
    from .mup_campaign_pin_store import execute_one_pinned_trial, load_pinned_campaign
    from .mup_campaign_schedule import NativeMuWarmupCosine
    from .mup_sweep_selection import MAX_UPDATES
    from .primary_campaign_construction62 import construct_scoped_primary_campaign62
    from .primary_training import NativeMuUpdateEngine
    from .runtime_capacity_gate import require_runtime_window_capacity

    _require(type(sources) is SourceAssemblyInputs,'Exact typed native source assembly required')
    _require(type(sources.reconstruction_source) is dict
             and sources.reconstruction_source.get('observed_group_manifest')==asdict(launch.profile_group_manifest),
             'Profile reward must bind the exact observed source group manifest')
    inputs=launch.inputs
    campaign=inputs.campaign
    trial=inputs.trial
    prefix=load_pinned_campaign(launch.controller_directory,launch.campaign_directory,
                                selection=campaign.selection_contract())
    campaign.require_next_trial(trial.trial_id,tuple(pin.trial_id for pin in prefix.pins))
    route_request=launch.route_request
    scoped=construct_scoped_primary_campaign62(root,route_request.path,route_request.sha256)
    _require(scoped.inputs==inputs and scoped.route_request==route_request
             and scoped.routes.plans==campaign.train_routes,
             'Executable routes differ from the reviewed primary request')
    bindings=dict(scoped.routes.bindings)
    evaluators=dict(scoped.routes.evaluators)
    capacity=require_runtime_window_capacity(root,campaign=campaign,bindings=bindings,request=launch.capacity)
    _require(capacity.sha256==launch.reviewed_capacity_sha256,'Live capacity differs from the reviewed release')

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(trial.seed)
        owner=construct_primary_owner(dict(inputs.configurations)['base'],sources)
        owner.authorize_primary_campaign_training(root,path=launch.release.path,expected_sha256=launch.release.sha256)
        # Verify the live parameter declarations against the reviewed family in
        # MuAdamW. Do not allocate unused delta and transfer-width replicas.
        device=torch.device(launch.device)
        with torch.random.fork_rng(devices=[] if device.type=='cpu' else [device.index]):
            if device.type=='cuda':
                torch.cuda.default_generators[device.index].manual_seed(trial.seed)
            owner.to(device=device)
            owner.require_training_admission(campaign_sha256=campaign.sha256,trial_id=trial.trial_id,
                                             window_capacity_sha256=capacity.sha256)
            roles=full_owner_mup_parameter_roles(owner,geometry_identifier_axis=launch.geometry_identifier_axis,
                                                  color_identifier_axis=launch.color_identifier_axis)
            optimizer=MuAdamW(owner,inputs.family,roles,base_lr=float(trial.learning_rate),
                              weight_decay=.01,decay_aliases=campaign.decay_aliases,member='base')
            scheduler=NativeMuWarmupCosine(optimizer,total_updates=MAX_UPDATES)
            profile=ProfileRewardStageController(
                owner.core.parent.parent.reconstruction,
                source_group_manifest_sha256=launch.profile_group_manifest.sha256,
                private_seed=launch.profile_private_seed,
            )
            engine=NativeMuUpdateEngine(
                owner,optimizer,binding_sha256=campaign.sha256,task_cycle=campaign.task_cycle,
                source_cursors={key:binding.route.initial_cursor for key,binding in bindings.items()},
                scheduler=scheduler,profile_stage_controller=profile,
                gradient_clip_max_norm=campaign.gradient_clip_max_norm,
            )
            return execute_one_pinned_trial(
                launch.controller_directory,launch.campaign_directory,campaign=campaign,trial_id=trial.trial_id,
                engine=engine,bindings=bindings,validation_view=scoped.routes.validation,
                validation_evaluators=evaluators,metric_identity=launch.metric_identity,root=root,
                journal_path=launch.journal_path,evidence_paths={row.key:row.reference.path for row in inputs.evidence},
                capacity_request=launch.capacity,
            )


def _load_repository_source_inputs(root, launch):
    from mora3.model.primary_source_factory import load_reviewed_native29_source_inputs

    return load_reviewed_native29_source_inputs(root,release_path=launch.release.path,
                                              release_sha256=launch.release.sha256,launch=launch)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    subparsers=parser.add_subparsers(dest='action',required=True)
    inspect=subparsers.add_parser('inspect',help='read bounded diagnostic metadata')
    inspect.add_argument('--context',required=True)
    inspect.add_argument('--context-sha256',required=True)
    inspect.add_argument('--primary-request',nargs=2,metavar=('PATH','SHA256'))
    subparsers.add_parser('source-preflight',help='verify native owner input provenance without loading sources')
    execute=subparsers.add_parser('execute',help='require an exact repository-reviewed worker release')
    execute.add_argument('--release',required=True)
    execute.add_argument('--release-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='execute':
            launch=preflight_primary_worker_launch(args.root,release_path=args.release,release_sha256=args.release_sha256)
            sources=_load_repository_source_inputs(args.root,launch)
            execute_primary_campaign(args.root,release_path=args.release,release_sha256=args.release_sha256,sources=sources)
            print(json.dumps({'worker_transaction':'committed'}))
            return 0
        if args.action=='source-preflight':
            from .primary_source_factory_preflight import preflight_native29_source_inputs

            plan=preflight_native29_source_inputs(args.root)
            print(json.dumps(asdict(plan)|{'owner_and_routes_rebound':False,'training_authorized':False},sort_keys=True))
            return 0
        context=build_primary_campaign_context(args.root,args.context,args.context_sha256)
        if args.primary_request:
            check_primary_request_metadata(args.root,*args.primary_request,context=context)
        print(json.dumps({
            'source_snapshot_sha256':context.source_snapshot.sha256,
            'route_count':len(context.train_routes),'observed_updates':context.prior_observed_updates,
            'consumed_windows':context.prior_consumed_windows,'validation_scopes':len(context.baselines),
            'validation_coordinates':sum(len(row.coordinates) for row in context.baselines),
            'validation_targets':sum(row.observed_targets for row in context.baselines),
            'allocation_authorized':False,'training_authorized':False,'unresolved':context.unresolved,
        },sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'Primary campaign refused: {error}',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
